"""
docx_reconstruction.py
======================

Native Microsoft Word reconstruction helpers for app.py.

Design goals
------------
- Keep tables editable in Word.
- Preserve table dimensions from PDF coordinates.
- Preserve cell text, font, size, bold, italic, underline and color when
  those properties are supplied by the extraction layer.
- Preserve borders and cell shading when supplied.
- Support merged cells when the source cell map contains identical/overlapping
  cells.
- Avoid fixed page-specific coordinates in application logic.
- Work with the IR dictionaries produced by app.py.

This module intentionally contains no LLM and no external API.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").replace("\u00ad", "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rgb_tuple(value: Any) -> Optional[tuple[int, int, int]]:
    """
    Accept:
      (r,g,b)
      [r,g,b]
      '#RRGGBB'
      integer 0xRRGGBB
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if re.fullmatch(r"[0-9a-fA-F]{6}", text):
            return (
                int(text[0:2], 16),
                int(text[2:4], 16),
                int(text[4:6], 16),
            )

    if isinstance(value, int):
        return (
            (value >> 16) & 255,
            (value >> 8) & 255,
            value & 255,
        )

    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            return (
                max(0, min(255, int(value[0]))),
                max(0, min(255, int(value[1]))),
                max(0, min(255, int(value[2]))),
            )
        except Exception:
            return None

    return None


def _rgb_hex(value: Any) -> Optional[str]:
    rgb = _rgb_tuple(value)
    if rgb is None:
        return None
    return "".join(f"{v:02X}" for v in rgb)


def _normalize_font_name(name: Any) -> str:
    text = _clean(name).strip()

    # Remove PDF subset prefixes such as ABCDEF+Calibri.
    if "+" in text:
        prefix, rest = text.split("+", 1)
        if len(prefix) == 6 and prefix.isalpha():
            text = rest

    return text or "Arial"


def _get_bbox(item: Any) -> Optional[tuple[float, float, float, float]]:
    if isinstance(item, dict):
        value = item.get("bbox")
    else:
        value = getattr(item, "bbox", None)

    if value is None:
        return None

    try:
        values = tuple(float(v) for v in value)
        if len(values) == 4:
            return values
    except Exception:
        pass

    return None


def _bbox_width(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox) -> float:
    return max(0.0, bbox[3] - bbox[1])


# ---------------------------------------------------------------------------
# Word XML helpers
# ---------------------------------------------------------------------------

def _get_or_add(parent, tag: str):
    element = parent.find(qn(tag))
    if element is None:
        element = OxmlElement(tag)
        parent.append(element)
    return element


def set_cell_width(cell, width_pt: float) -> None:
    """
    Set exact Word table-cell width in points.
    """
    width_pt = max(0.1, float(width_pt))

    tc_pr = cell._tc.get_or_add_tcPr()

    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)

    tc_w.set(
        qn("w:w"),
        str(max(1, int(round(width_pt * 20))),
    ))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_height(cell, height_pt: float) -> None:
    """
    Set exact minimum cell height in points.
    """
    height_pt = max(0.1, float(height_pt))

    tr = cell._tc.getparent()
    if tr is None:
        return

    tr_pr = tr.get_or_add_trPr()

    # Remove old exact-height values from this row.
    for old in list(tr_pr.findall(qn("w:trHeight"))):
        tr_pr.remove(old)

    tr_height = OxmlElement("w:trHeight")
    tr_height.set(
        qn("w:val"),
        str(max(1, int(round(height_pt * 20)))),
    )
    tr_height.set(
        qn("w:hRule"),
        "atLeast",
    )
    tr_pr.append(tr_height)


def set_cell_shading(cell, fill: Any) -> None:
    color = _rgb_hex(fill)
    if not color:
        return

    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))

    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)

    shd.set(qn("w:fill"), color)
    shd.set(qn("w:val"), "clear")


def set_cell_margins(
    cell,
    top: float = 0.0,
    start: float = 0.0,
    bottom: float = 0.0,
    end: float = 0.0,
) -> None:
    """
    Word cell margins are stored in twentieths of a point (twips).
    """
    tc_pr = cell._tc.get_or_add_tcPr()

    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)

    values = {
        "top": top,
        "start": start,
        "bottom": bottom,
        "end": end,
    }

    for side, value in values.items():
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)

        node.set(
            qn("w:w"),
            str(max(0, int(round(float(value) * 20)))),
        )
        node.set(qn("w:type"), "dxa")


def set_cell_vertical_alignment(
    cell,
    value: Any = "top",
) -> None:
    text = _clean(value).lower()

    mapping = {
        "top": WD_ALIGN_VERTICAL.TOP,
        "center": WD_ALIGN_VERTICAL.CENTER,
        "middle": WD_ALIGN_VERTICAL.CENTER,
        "bottom": WD_ALIGN_VERTICAL.BOTTOM,
    }

    cell.vertical_alignment = mapping.get(
        text,
        WD_ALIGN_VERTICAL.TOP,
    )


def set_table_borders(
    table,
    color: Any = (0, 0, 0),
    size: int = 4,
    val: str = "single",
) -> None:
    """
    Apply one border definition to the complete table.
    """
    rgb = _rgb_hex(color) or "000000"

    tbl = table._tbl
    tbl_pr = tbl.tblPr

    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)

    for edge in (
        "top",
        "left",
        "bottom",
        "right",
        "insideH",
        "insideV",
    ):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)

        tag.set(qn("w:val"), val)
        tag.set(qn("w:sz"), str(max(0, int(size))))
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), rgb)


def set_table_layout_fixed(table) -> None:
    tbl_pr = table._tbl.tblPr

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)

    layout.set(qn("w:type"), "fixed")


def set_table_width(table, width_pt: float) -> None:
    width_pt = max(0.1, float(width_pt))

    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))

    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)

    tbl_w.set(
        qn("w:w"),
        str(max(1, int(round(width_pt * 20)))),
    )
    tbl_w.set(qn("w:type"), "dxa")


def set_table_alignment(
    table,
    alignment: Any = "left",
) -> None:
    text = _clean(alignment).lower()

    if text == "center":
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
    elif text == "right":
        table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    else:
        table.alignment = WD_TABLE_ALIGNMENT.LEFT


def set_table_indent(
    table,
    indent_pt: float,
) -> None:
    tbl_pr = table._tbl.tblPr

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)

    tbl_ind.set(
        qn("w:w"),
        str(max(0, int(round(indent_pt * 20)))),
    )
    tbl_ind.set(qn("w:type"), "dxa")


# ---------------------------------------------------------------------------
# Paragraph / run formatting
# ---------------------------------------------------------------------------

def set_paragraph_spacing(
    paragraph,
    before_pt: float = 0.0,
    after_pt: float = 0.0,
    line_spacing: float = 1.0,
) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(max(0.0, before_pt))
    fmt.space_after = Pt(max(0.0, after_pt))

    try:
        fmt.line_spacing = max(0.1, line_spacing)
    except Exception:
        pass


def set_paragraph_alignment(
    paragraph,
    alignment: Any = "left",
) -> None:
    text = _clean(alignment).lower()

    mapping = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "both": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }

    paragraph.alignment = mapping.get(
        text,
        WD_ALIGN_PARAGRAPH.LEFT,
    )


def set_run_font(
    run,
    font_name: Any = "Arial",
    size_pt: Any = 10,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    color: Any = None,
) -> None:
    name = _normalize_font_name(font_name)

    try:
        size = max(1.0, float(size_pt))
    except Exception:
        size = 10.0

    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bool(bold)
    run.font.italic = bool(italic)
    run.font.underline = bool(underline)

    # Explicitly set all Word font families. This matters for non-Latin text.
    rpr = run._r.get_or_add_rPr()

    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)

    for attr in (
        "w:ascii",
        "w:hAnsi",
        "w:eastAsia",
        "w:cs",
    ):
        rfonts.set(qn(attr), name)

    # Word uses half-points for w:sz.
    sz = rpr.find(qn("w:sz"))
    if sz is None:
        sz = OxmlElement("w:sz")
        rpr.append(sz)

    sz.set(
        qn("w:val"),
        str(max(2, int(round(size * 2)))),
    )

    sz_cs = rpr.find(qn("w:szCs"))
    if sz_cs is None:
        sz_cs = OxmlElement("w:szCs")
        rpr.append(sz_cs)

    sz_cs.set(
        qn("w:val"),
        str(max(2, int(round(size * 2)))),
    )

    rgb = _rgb_hex(color)
    if rgb:
        run.font.color.rgb = RGBColor.from_string(
            rgb
        )


def write_cell_text(
    cell,
    value: Any,
    font: Any = "Arial",
    size: Any = 10,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    color: Any = None,
    align: Any = "left",
    valign: Any = "top",
    before_pt: float = 0.0,
    after_pt: float = 0.0,
    line_spacing: float = 1.0,
) -> None:
    cell.text = ""

    paragraph = cell.paragraphs[0]

    set_paragraph_alignment(
        paragraph,
        align,
    )

    set_paragraph_spacing(
        paragraph,
        before_pt,
        after_pt,
        line_spacing,
    )

    set_cell_vertical_alignment(
        cell,
        valign,
    )

    run = paragraph.add_run(
        _clean(value)
    )

    set_run_font(
        run,
        font,
        size,
        bold,
        italic,
        underline,
        color,
    )


# ---------------------------------------------------------------------------
# Cell/row geometry
# ---------------------------------------------------------------------------

def _extract_rows(table_data: dict) -> list:
    rows = table_data.get("rows", [])
    if rows is None:
        return []
    return list(rows)


def _cell_value(cell_data: Any, key: str, default=None):
    if isinstance(cell_data, dict):
        return cell_data.get(key, default)

    return getattr(
        cell_data,
        key,
        default,
    )


def _cell_bbox(cell_data: Any):
    bbox = _cell_value(
        cell_data,
        "bbox",
        None,
    )

    if bbox is None:
        return None

    try:
        return tuple(
            float(v)
            for v in bbox
        )
    except Exception:
        return None


def _calculate_column_widths(
    rows: list,
    table_bbox=None,
) -> list[float]:
    """
    Calculate a stable set of column widths from source cell geometry.

    The source table may have rows with different numbers of cells.
    We determine the maximum column count and use the maximum observed
    width for each column.
    """
    col_count = max(
        (
            len(row)
            for row in rows
            if row is not None
        ),
        default=0,
    )

    if col_count == 0:
        return []

    widths = [0.0] * col_count

    for row in rows:
        for index, cell_data in enumerate(
            row or []
        ):
            bbox = _cell_bbox(cell_data)

            if bbox is not None:
                widths[index] = max(
                    widths[index],
                    _bbox_width(bbox),
                )

    # Fill missing widths using table width.
    table_width = 0.0

    if table_bbox is not None:
        try:
            table_width = _bbox_width(
                tuple(float(v) for v in table_bbox)
            )
        except Exception:
            table_width = 0.0

    known = sum(widths)

    missing = [
        i
        for i, width in enumerate(widths)
        if width <= 0
    ]

    if missing:
        remaining = max(
            1.0,
            table_width - known,
        )

        each = remaining / len(missing)

        for index in missing:
            widths[index] = each

    if sum(widths) <= 0:
        widths = [72.0] * col_count

    return widths


def _calculate_row_heights(
    rows: list,
) -> list[float]:
    heights = []

    for row in rows:
        row_height = 0.0

        for cell_data in row or []:
            bbox = _cell_bbox(cell_data)

            if bbox is not None:
                row_height = max(
                    row_height,
                    _bbox_height(bbox),
                )

        heights.append(
            row_height
            if row_height > 0
            else 18.0
        )

    return heights


# ---------------------------------------------------------------------------
# Merged-cell detection
# ---------------------------------------------------------------------------

def _same_horizontal_span(
    a,
    b,
    tolerance: float = 1.5,
) -> bool:
    return (
        abs(a[0] - b[0]) <= tolerance
        and abs(a[2] - b[2]) <= tolerance
    )


def _same_vertical_span(
    a,
    b,
    tolerance: float = 1.5,
) -> bool:
    return (
        abs(a[1] - b[1]) <= tolerance
        and abs(a[3] - b[3]) <= tolerance
    )


def detect_merge_groups(
    rows: list,
    tolerance: float = 1.5,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Detect simple horizontal/vertical merged-cell groups from geometry.

    This intentionally avoids guessing arbitrary spans from text alone.
    Only geometrically strong cases are merged.
    """
    groups = []

    for r, row in enumerate(rows):
        for c, cell_data in enumerate(row or []):
            bbox = _cell_bbox(cell_data)
            if bbox is None:
                continue

            # Horizontal merge: current cell and next cell share the same
            # vertical boundaries but current cell covers more width.
            if c + 1 < len(row):
                next_bbox = _cell_bbox(row[c + 1])

                if (
                    next_bbox is not None
                    and _same_vertical_span(
                        bbox,
                        next_bbox,
                        tolerance,
                    )
                ):
                    # Do not merge ordinary adjacent cells that merely have
                    # the same height. A merged source cell normally overlaps
                    # the next geometry or has explicit span metadata.
                    span = _cell_value(
                        cell_data,
                        "col_span",
                        None,
                    )

                    if span and int(span) > 1:
                        groups.append(
                            (
                                (r, c),
                                (r, c + int(span) - 1),
                            )
                        )

            # Vertical merge if explicit row_span exists.
            span = _cell_value(
                cell_data,
                "row_span",
                None,
            )

            if span and int(span) > 1:
                groups.append(
                    (
                        (r, c),
                        (
                            r + int(span) - 1,
                            c,
                        ),
                    )
                )

    return groups


def _apply_explicit_merges(
    table,
    rows: list,
) -> None:
    groups = detect_merge_groups(
        rows
    )

    for start, end in groups:
        try:
            r0, c0 = start
            r1, c1 = end

            if (
                r0 < 0
                or c0 < 0
                or r1 >= len(table.rows)
                or c1 >= len(table.columns)
            ):
                continue

            if r0 == r1 and c0 == c1:
                continue

            table.cell(
                r0,
                c0,
            ).merge(
                table.cell(
                    r1,
                    c1,
                )
            )

        except Exception:
            # A malformed source map should not make the entire document
            # unusable.
            continue


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def reconstruct_table_element(
    doc,
    table_data: dict,
):
    """
    Reconstruct one PDF table as an editable Word table.

    Expected input:
    {
        "type": "table",
        "bbox": (x0, y0, x1, y1),
        "rows": [
            [
                {
                    "text": "...",
                    "font": "Arial",
                    "size": 10,
                    "bold": False,
                    "italic": False,
                    "underline": False,
                    "color": (0,0,0),
                    "align": "left",
                    "valign": "top",
                    "bbox": (x0,y0,x1,y1),
                    "fill": (255,255,255),   # optional
                    "col_span": 1,           # optional
                    "row_span": 1            # optional
                }
            ]
        ]
    }
    """
    rows = _extract_rows(
        table_data
    )

    if not rows:
        return None

    col_count = max(
        (
            len(row or [])
            for row in rows
        ),
        default=0,
    )

    if col_count <= 0:
        return None

    table_bbox = table_data.get(
        "bbox"
    )

    table = doc.add_table(
        rows=len(rows),
        cols=col_count,
    )

    table.autofit = False

    set_table_layout_fixed(
        table
    )

    # Table geometry.
    column_widths = _calculate_column_widths(
        rows,
        table_bbox,
    )

    row_heights = _calculate_row_heights(
        rows
    )

    total_width = sum(
        column_widths
    )

    if total_width > 0:
        set_table_width(
            table,
            total_width,
        )

    set_table_alignment(
        table,
        table_data.get(
            "align",
            "left",
        ),
    )

    indent = table_data.get(
        "indent_pt",
        None,
    )

    if indent is not None:
        set_table_indent(
            table,
            _safe_float(
                indent
            ),
        )

    # Default border configuration.
    border_color = table_data.get(
        "border_color",
        (0, 0, 0),
    )

    border_size = table_data.get(
        "border_size",
        4,
    )

    border_value = table_data.get(
        "border_val",
        "single",
    )

    set_table_borders(
        table,
        border_color,
        border_size,
        border_value,
    )

    # Cell geometry and formatting.
    for r, row in enumerate(rows):
        if r >= len(table.rows):
            break

        row_cells = row or []

        for c in range(col_count):
            cell = table.cell(
                r,
                c,
            )

            cell_data = (
                row_cells[c]
                if c < len(row_cells)
                else {}
            )

            width = (
                column_widths[c]
                if c < len(column_widths)
                else 72.0
            )

            set_cell_width(
                cell,
                width,
            )

            height = (
                row_heights[r]
                if r < len(row_heights)
                else 18.0
            )

            set_cell_height(
                cell,
                height,
            )

            set_cell_margins(
                cell,
                top=_safe_float(
                    _cell_value(
                        cell_data,
                        "margin_top",
                        0.0,
                    )
                ),
                start=_safe_float(
                    _cell_value(
                        cell_data,
                        "margin_left",
                        0.0,
                    )
                ),
                bottom=_safe_float(
                    _cell_value(
                        cell_data,
                        "margin_bottom",
                        0.0,
                    )
                ),
                end=_safe_float(
                    _cell_value(
                        cell_data,
                        "margin_right",
                        0.0,
                    )
                ),
            )

            fill = _cell_value(
                cell_data,
                "fill",
                None,
            )

            if fill is not None:
                set_cell_shading(
                    cell,
                    fill,
                )

            write_cell_text(
                cell,
                _cell_value(
                    cell_data,
                    "text",
                    "",
                ),
                font=_cell_value(
                    cell_data,
                    "font",
                    "Arial",
                ),
                size=_cell_value(
                    cell_data,
                    "size",
                    10,
                ),
                bold=bool(
                    _cell_value(
                        cell_data,
                        "bold",
                        False,
                    )
                ),
                italic=bool(
                    _cell_value(
                        cell_data,
                        "italic",
                        False,
                    )
                ),
                underline=bool(
                    _cell_value(
                        cell_data,
                        "underline",
                        False,
                    )
                ),
                color=_cell_value(
                    cell_data,
                    "color",
                    None,
                ),
                align=_cell_value(
                    cell_data,
                    "align",
                    "left",
                ),
                valign=_cell_value(
                    cell_data,
                    "valign",
                    "top",
                ),
                before_pt=_safe_float(
                    _cell_value(
                        cell_data,
                        "space_before",
                        0,
                    )
                ),
                after_pt=_safe_float(
                    _cell_value(
                        cell_data,
                        "space_after",
                        0,
                    )
                ),
                line_spacing=_safe_float(
                    _cell_value(
                        cell_data,
                        "line_spacing",
                        1,
                    ),
                    1,
                ),
            )

    # Apply explicit merge metadata only.
    _apply_explicit_merges(
        table,
        rows,
    )

    # Some source maps can provide per-cell borders.
    for r, row in enumerate(rows):
        for c, cell_data in enumerate(
            row or []
        ):
            if r >= len(table.rows) or c >= col_count:
                continue

            borders = _cell_value(
                cell_data,
                "borders",
                None,
            )

            if isinstance(
                borders,
                dict,
            ):
                _set_cell_borders(
                    table.cell(r, c),
                    borders,
                )

    # Prevent row splitting where possible.
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()

        cant_split = tr_pr.find(
            qn("w:cantSplit")
        )

        if cant_split is None:
            cant_split = OxmlElement(
                "w:cantSplit"
            )
            tr_pr.append(
                cant_split
            )

    # Keep table together as much as Word permits.
    tbl_pr = table._tbl.tblPr
    tbl_overlap = tbl_pr.find(
        qn("w:tblOverlap")
    )

    if tbl_overlap is None:
        tbl_overlap = OxmlElement(
            "w:tblOverlap"
        )
        tbl_pr.append(
            tbl_overlap
        )

    tbl_overlap.set(
        qn("w:val"),
        "never",
    )

    return table


# ---------------------------------------------------------------------------
# Per-cell border support
# ---------------------------------------------------------------------------

def _set_cell_borders(
    cell,
    borders: dict,
) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()

    tc_borders = tc_pr.find(
        qn("w:tcBorders")
    )

    if tc_borders is None:
        tc_borders = OxmlElement(
            "w:tcBorders"
        )
        tc_pr.append(
            tc_borders
        )

    for edge in (
        "top",
        "left",
        "bottom",
        "right",
        "insideH",
        "insideV",
        "start",
        "end",
    ):
        spec = borders.get(edge)

        if spec is None:
            continue

        if not isinstance(spec, dict):
            spec = {
                "color": spec
            }

        node = tc_borders.find(
            qn(f"w:{edge}")
        )

        if node is None:
            node = OxmlElement(
                f"w:{edge}"
            )
            tc_borders.append(node)

        node.set(
            qn("w:val"),
            _clean(
                spec.get(
                    "val",
                    "single",
                )
            ),
        )
        node.set(
            qn("w:sz"),
            str(
                max(
                    0,
                    int(
                        _safe_float(
                            spec.get(
                                "size",
                                4,
                            ),
                            4,
                        )
                    ),
                )
            ),
        )
        node.set(
            qn("w:space"),
            "0",
        )
        node.set(
            qn("w:color"),
            _rgb_hex(
                spec.get(
                    "color",
                    (0, 0, 0),
                )
            )
            or "000000",
        )



# ---------------------------------------------------------------------------
# Rotated PDF-table reconstruction
# ---------------------------------------------------------------------------
def _set_cell_text_direction(cell, value="btLr"):
    """Rotate text inside a Word table cell without rasterizing it."""
    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:textDirection"))
    if node is None:
        node = OxmlElement("w:textDirection")
        tc_pr.append(node)
    node.set(qn("w:val"), value)


def _set_table_position(table, left_pt, top_pt):
    """Anchor a table to page coordinates."""
    tbl_pr = table._tbl.tblPr
    old = tbl_pr.find(qn("w:tblpPr"))
    if old is not None:
        tbl_pr.remove(old)
    pos = OxmlElement("w:tblpPr")
    pos.set(qn("w:leftFromText"), "0")
    pos.set(qn("w:rightFromText"), "0")
    pos.set(qn("w:topFromText"), "0")
    pos.set(qn("w:bottomFromText"), "0")
    pos.set(qn("w:vertAnchor"), "page")
    pos.set(qn("w:horzAnchor"), "page")
    pos.set(qn("w:tblpX"), str(int(round(left_pt * 20))))
    pos.set(qn("w:tblpY"), str(int(round(top_pt * 20))))
    tbl_pr.append(pos)


def reconstruct_rotated_table_element(doc, table_data, target_lang=None,
                                       translator_fn=None, font_override=None):
    """Visually reconstruct a 90-degree PDF table with editable Word text.

    Word has no reliable cross-version API for rotating an entire native table.
    For this unusual PDF construct, the best visual/compatibility compromise is
    a set of absolutely-positioned VML text boxes: every cell's text remains
    editable/selectable, borders remain visible, and the objects do not alter
    normal pagination. Ordinary non-rotated PDF tables continue to use native
    Word tables through reconstruct_table_element().
    """
    rows = table_data.get("rows") or []
    bbox = table_data.get("bbox")
    if not rows or not bbox:
        return None
    n_rows=len(rows); n_cols=max(len(r or []) for r in rows)
    if not n_rows or not n_cols:
        return None
    row_bounds=table_data.get("row_bounds") or []
    col_bounds=table_data.get("col_bounds") or []
    x0,y0,x1,y1=map(float,bbox)
    if len(row_bounds)!=n_rows:
        step=(x1-x0)/n_rows; row_bounds=[(x0+i*step,x0+(i+1)*step) for i in range(n_rows)]
    if len(col_bounds)!=n_cols:
        step=(y1-y0)/n_cols; col_bounds=[(y0+i*step,y0+(i+1)*step) for i in range(n_cols)]

    import html
    anchor=doc.paragraphs[-1]
    for lc,cb in enumerate(col_bounds):
        for lr,rb in enumerate(row_bounds):
            d=rows[lr][lc] if lc<len(rows[lr]) else {}
            if isinstance(d,str): d={"text":d}
            text=str(d.get("text","") or "")
            if not text: continue
            font=_normalize_font_name(d.get("font","Arial"))
            size=max(4.0,min(18.0,_safe_float(d.get("size",8),8)))
            bold=bool(d.get("bold",False)); italic=bool(d.get("italic",False)); underline=bool(d.get("underline",False))
            color=_rgb_hex(d.get("color",(0,0,0))) or "000000"
            # Shape dimensions before rotation are swapped: the narrow physical
            # cell becomes a tall text box, then rotation returns it to PDF space.
            left=cb[0]; top=rb[0]; width=max(2.0,cb[1]-cb[0]); height=max(2.0,rb[1]-rb[0])
            rpr=(f'<w:rFonts w:ascii="{html.escape(font)}" w:hAnsi="{html.escape(font)}" w:eastAsia="{html.escape(font)}" w:cs="{html.escape(font)}"/>'
                 + ('<w:b/><w:bCs/>' if bold else '') + ('<w:i/><w:iCs/>' if italic else '')
                 + ('<w:u w:val="single"/>' if underline else '')
                 + f'<w:sz w:val="{max(2,int(round(size*2)))}"/><w:szCs w:val="{max(2,int(round(size*2)))}"/><w:color w:val="{color}"/>')
            # Use the actual cell rectangle and rotate the text box 90 degrees.
            # VML rotation is honored by Word and LibreOffice and, unlike a
            # floating table, does not consume pagination space.
            xml=(
                '<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:pict xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">'
                f'<v:shape style="position:absolute;left:{left:.2f}pt;top:{top:.2f}pt;'
                f'width:{width:.2f}pt;height:{height:.2f}pt;rotation:270;'
                'mso-wrap-style:none;mso-position-horizontal:absolute;mso-position-horizontal-relative:page;'
                'mso-position-vertical:absolute;mso-position-vertical-relative:page" '
                'stroked="t" strokecolor="#6D6E70" strokeweight="0.5pt" filled="f">'
                '<v:textbox inset="0,0,0,0" style="mso-fit-shape-to-text:f;v-text-anchor:middle">'
                f'<w:txbxContent><w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="0" w:line="200" w:lineRule="exact"/></w:pPr>'
                f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{html.escape(text)}</w:t></w:r>'
                '</w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r>'
            )
            anchor._p.append(OxmlElement("w:r"))
            anchor._p.remove(anchor._p[-1])
            anchor._p.append(__import__('docx').oxml.parse_xml(xml))
    return anchor._p

# ---------------------------------------------------------------------------
# Optional utility for rebuilding an entire table list
# ---------------------------------------------------------------------------

def reconstruct_tables(
    doc,
    tables: Iterable[dict],
) -> list:
    """
    Convenience wrapper for callers that have multiple table dictionaries.
    """
    result = []

    for table_data in tables:
        if not isinstance(
            table_data,
            dict,
        ):
            continue

        table = reconstruct_table_element(
            doc,
            table_data,
        )

        if table is not None:
            result.append(table)

    return result


__all__ = [
    "reconstruct_table_element",
    "reconstruct_rotated_table_element",
    "reconstruct_tables",
    "set_cell_width",
    "set_cell_height",
    "set_cell_shading",
    "set_cell_margins",
    "set_cell_vertical_alignment",
    "set_table_borders",
    "set_table_layout_fixed",
    "set_table_width",
    "set_table_alignment",
    "set_table_indent",
    "set_paragraph_spacing",
    "set_paragraph_alignment",
    "set_run_font",
    "write_cell_text",
]
