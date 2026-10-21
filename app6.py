"""io
PDF -> Editable DOCX Converter -- FULL production pipeline
Matches the client's architecture diagram exactly:

Upload PDF -> PDF Type Detection (Text / Scanned)
  -> Scanned: PDF -> Images -> PaddleOCR
  -> Text: parsed directly
-> PyMuPDF (parsing, font, position, image extraction)
-> DocLayout-YOLO (layout detection: title / body / figure / table / caption / abandon)
-> PP-StructureV3 (table structure recognition)
-> Extract Text + Font + Position
-> Translation Engine (pluggable LLM / NMT hook)
-> Document Reconstruction Engine (python-docx)
-> Translated / Editable DOCX
"""

import io
import os
import re
import html
from html.parser import HTMLParser
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import fitz  # PyMuPDF
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION, WD_ORIENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.text.run import Run

# Table reconstruction engine
from docx_reconstruction import reconstruct_table_element

# Overlap/overflow detection, debug JSON export, and DOCX-vs-PDF render-compare
# validation -- deliberately a separate module (spec section 24).
from validation import (
    detect_element_overlaps,
    page_debug_entry,
    build_debug_json,
    validate_docx_against_pdf,
)

# --------------------------------------------------------------------------- #
# Optional heavy ML deps -- imported lazily inside the cached loader functions
# --------------------------------------------------------------------------- #

def _probe(mod_name):
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False

DOCLAYOUT_AVAILABLE = _probe("doclayout_yolo")
PADDLEOCR_AVAILABLE = _probe("paddleocr")
HF_HUB_AVAILABLE = _probe("huggingface_hub")

DOCLAYOUT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
DOCLAYOUT_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

LAYOUT_ROLE = {
    "title": "heading",
    "plain text": "body",
    "abandon": "skip",
    "figure": "figure",
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "table_footnote": "caption",
    "isolate_formula": "formula",
    "formula_caption": "caption",
}


# --------------------------------------------------------------------------- #
# PDF page-size + application/metadata detection
#
# Reads the PDF's actual MediaBox/page rectangle (via PyMuPDF) for every
# page -- never guesses from the filename -- and the document Info
# dictionary (Creator/Producer/Author/Title/Subject) for "which app made
# this PDF". Pure detection helpers: no Streamlit calls in here so they can
# be unit-tested / reused headlessly.
# --------------------------------------------------------------------------- #

# 1 pt = 1/72 inch. All internal page-size math is done in points, matching
# python-docx's Pt() unit and the PDF's native page-rectangle unit.
_UNIT_TO_PT = {
    "pt": 1.0,
    "mm": 72.0 / 25.4,
    "cm": 72.0 / 2.54,
    "inch": 72.0,
    "in": 72.0,
}

# Standard paper sizes in mm, portrait (width, height). Matched against
# either orientation of a detected page.
_STANDARD_SIZES_MM = {
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
}


def unit_to_pt(value: float, unit: str) -> float:
    """Convert a value in the given unit (mm/cm/inch/pt) to points."""
    return float(value) * _UNIT_TO_PT.get(unit, 1.0)


def pt_to_unit(value_pt: float, unit: str) -> float:
    """Convert a value in points to the given unit (mm/cm/inch/pt)."""
    return float(value_pt) / _UNIT_TO_PT.get(unit, 1.0)


def classify_standard_size(width_pt: float, height_pt: float, tolerance_mm: float = 2.0) -> str:
    """Identify a page rectangle (in pt) as A3/A4/A5/Letter/Legal/Custom.

    Compares both dimensions regardless of orientation (a landscape A4 page
    still reports "A4") within a small tolerance to absorb PDF producers'
    rounding. Returns "Custom" if nothing matches.
    """
    w_mm = pt_to_unit(width_pt, "mm")
    h_mm = pt_to_unit(height_pt, "mm")
    dims = sorted([w_mm, h_mm])
    for name, (sw, sh) in _STANDARD_SIZES_MM.items():
        std = sorted([sw, sh])
        if abs(dims[0] - std[0]) <= tolerance_mm and abs(dims[1] - std[1]) <= tolerance_mm:
            return name
    return "Custom"


def detect_pdf_application(pdf_doc: "fitz.Document") -> dict:
    """Read the PDF Info dictionary and return the creating application's
    metadata fields. Never guesses -- falls back to "Not available" for any
    field the PDF simply doesn't carry."""
    meta = pdf_doc.metadata or {}

    def _field(key):
        val = (meta.get(key) or "").strip()
        return val if val else "Not available"

    return {
        "creator": _field("creator"),
        "producer": _field("producer"),
        "author": _field("author"),
        "title": _field("title"),
        "subject": _field("subject"),
    }


def detect_pdf_page_info(pdf_bytes: bytes) -> dict:
    """Detect per-page width/height/orientation/standard-size (from the
    PDF's real page rectangle -- never assumed from the document name) plus
    the creating application's metadata.

    Returns:
        {
          "application": {...},
          "pages": [{"page_num", "width_pt", "height_pt", "orientation", "standard_size"}, ...],
          "mixed": bool,   # True if pages differ in dimensions
          "n_pages": int,
        }
    """
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        application = detect_pdf_application(pdf_doc)
        pages = []
        for i, page in enumerate(pdf_doc):
            rect = page.rect
            width_pt, height_pt = rect.width, rect.height
            orientation = "Landscape" if width_pt > height_pt else "Portrait"
            pages.append({
                "page_num": i + 1,
                "width_pt": width_pt,
                "height_pt": height_pt,
                "orientation": orientation,
                "standard_size": classify_standard_size(width_pt, height_pt),
            })
        distinct = {(round(p["width_pt"], 1), round(p["height_pt"], 1)) for p in pages}
        return {
            "application": application,
            "pages": pages,
            "mixed": len(distinct) > 1,
            "n_pages": len(pages),
        }
    finally:
        pdf_doc.close()


# --------------------------------------------------------------------------- #
# Cached model loaders
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading DocLayout-YOLO layout model (first run only)...")
def get_layout_model():
    from huggingface_hub import hf_hub_download
    from doclayout_yolo import YOLOv10
    weight_path = hf_hub_download(repo_id=DOCLAYOUT_REPO, filename=DOCLAYOUT_FILENAME)
    return YOLOv10(weight_path)


@st.cache_resource(show_spinner="Loading PaddleOCR text recognition model (first run only)...")
def get_ocr_engine(lang: str = "en"):
    from paddleocr import PaddleOCR
    # enable_mkldnn=False works around a PaddlePaddle 3.3.x bug where CPU
    # oneDNN inference raises:
    #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
    #   support [pir::ArrayAttribute<pir::DoubleAttribute>]
    # See: https://github.com/PaddlePaddle/Paddle/issues/77340
    try:
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except TypeError:
        # Older paddleocr builds don't accept enable_mkldnn as a kwarg here.
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )


@st.cache_resource(show_spinner="Loading PP-StructureV3 table recognition model (first run only)...")
def get_table_engine():
    from paddleocr import PPStructureV3
    # Same oneDNN/PIR workaround as get_ocr_engine() above.
    try:
        return PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_chart_recognition=False,
            use_formula_recognition=False,
            enable_mkldnn=False,
        )
    except TypeError:
        return PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_chart_recognition=False,
            use_formula_recognition=False,
        )


# --------------------------------------------------------------------------- #
# Noise filtering & font mappings
# --------------------------------------------------------------------------- #

_NOISE_STRINGS = {"separator", "spacer", "divider"}


def _is_noise_text(text: str) -> bool:
    return text.strip().lower() in _NOISE_STRINGS


FONT_SUBSTITUTES = {
    # Standard Sans-Serif
    "arial": "Arial", "arialmt": "Arial", "helvetica": "Arial", "helveticaneue": "Arial",
    "arialblack": "Arial Black", "ariblk": "Arial Black",
    "calibri": "Calibri", "calibrilight": "Calibri Light",
    "aptos": "Aptos", "aptosdisplay": "Aptos Display",
    "segoeui": "Segoe UI", "segoeuisemibold": "Segoe UI Semibold", "segoeuilight": "Segoe UI Light",
    "tahoma": "Tahoma", "verdana": "Verdana", "trebuchetms": "Trebuchet MS",
    "centurygothic": "Century Gothic", "gothic": "Century Gothic", "futura": "Century Gothic",
    "franklingothic": "Franklin Gothic Medium", "franklingothicbook": "Franklin Gothic Book",
    "candara": "Candara", "corbel": "Corbel",
    "bahnschrift": "Bahnschrift", "din": "Bahnschrift", "dinpro": "Bahnschrift",
    "lucidasans": "Lucida Sans", "lucidagrande": "Lucida Sans",
    "gillsans": "Gill Sans MT", "gillsansmt": "Gill Sans MT",
    "myriadpro": "Arial", "myriad": "Arial", "proximanova": "Arial", "gotham": "Arial", "avenir": "Arial", "avenirnext": "Arial",

    # Modern Web / Universal Sans-Serif
    "opensans": "Open Sans", "roboto": "Roboto", "lato": "Lato", "montserrat": "Montserrat",
    "poppins": "Poppins", "inter": "Inter", "nunito": "Nunito", "sourcesanspro": "Source Sans Pro",
    "sourcesans3": "Source Sans Pro", "cabin": "Open Sans",

    # Narrow / Condensed Sans-Serif
    "arialnarrow": "Arial Narrow", "arialnarrowmt": "Arial Narrow",
    "helveticanarrow": "Arial Narrow", "helveticacondensed": "Arial Narrow",
    "arialcondensed": "Arial Narrow", "arialnarrowbold": "Arial Narrow",
    "liberationsansnarrow": "Arial Narrow", "robotocondensed": "Arial Narrow",

    # Standard & Traditional Serif
    "times": "Times New Roman", "timesnewroman": "Times New Roman", "timesnewromanpsmt": "Times New Roman",
    "timesroman": "Times New Roman", "tmsrmn": "Times New Roman",
    "georgia": "Georgia", "cambria": "Cambria", "garamond": "Garamond", "ebgaramond": "Garamond",
    "palatino": "Palatino Linotype", "palatinolinotype": "Palatino Linotype", "bookantiqua": "Book Antiqua",
    "baskerville": "Baskerville Old Face", "baskervilleoldface": "Baskerville Old Face",
    "bodoni": "Bodoni MT", "bodonimt": "Bodoni MT",
    "centuryschoolbook": "Century Schoolbook", "newschoolbook": "Century Schoolbook",
    "constantia": "Constantia", "minion": "Minion Pro", "minionpro": "Minion Pro",
    "didot": "Didot", "bookman": "Bookman Old Style", "bookmanoldstyle": "Bookman Old Style",
    "rockwell": "Rockwell",

    # Monospaced / Code / Fixed Width
    "consolas": "Consolas", "courier": "Courier New", "couriernew": "Courier New",
    "couriernewps": "Courier New", "couriernewpsmt": "Courier New", "courierps": "Courier New",
    "courierstd": "Courier New", "courierbold": "Courier New", "couriernewbold": "Courier New",
    "lucidaconsole": "Lucida Console", "sourcecodepro": "Consolas",
    "menlo": "Consolas", "monaco": "Consolas", "inconsolata": "Consolas",
    "firacode": "Consolas",

    # CJK & Complex Scripts
    "simsun": "SimSun", "nsimsun": "NSimSun", "songti": "SimSun",
    "simhei": "SimHei", "heiti": "SimHei",
    "microsoftyahei": "Microsoft YaHei", "msyahei": "Microsoft YaHei",
    "kaiti": "KaiTi", "fangsong": "FangSong",
    "msgothic": "MS Gothic", "mspgothic": "MS PGothic", "msuigothic": "MS UI Gothic",
    "msmincho": "MS Mincho", "mspmincho": "MS PMincho",
    "meiryo": "Meiryo", "yugothic": "Yu Gothic",
    "malgungothic": "Malgun Gothic", "batang": "Batang", "gulim": "Gulim", "dotum": "Dotum",
    "nirmalaui": "Nirmala UI", "mangal": "Mangal", "arialunicodems": "Arial Unicode MS",
    "notosans": "Noto Sans", "notoserif": "Noto Serif",

    # Symbols
    "symbol": "Symbol", "wingdings": "Wingdings", "wingdings2": "Wingdings 2",
    "wingdings3": "Wingdings 3", "webdings": "Webdings",
}

# Any of these substrings appearing in the raw font name mean "condensed/narrow" --
# checked BEFORE the bold/italic-style suffix stripping below, because narrow/
# condensed is a real geometric variant of the font (not a weight/style like Bold)
# and must be preserved rather than silently dropped.
_CONDENSED_MARKERS = ("narrow", "condensed", "cond", "compressed")

# Matches generic PDF internal font resource ids that are NOT real font
# names: "F1", "F12", "C0", "TT2", "T1_0", short all-caps/digit codes, etc.
_RAW_FONT_ID_RE = re.compile(r"^[a-z]{1,3}\d{1,3}(_\d+)?$", re.I)


def _looks_like_raw_font_resource_id(name: str) -> bool:
    if not name:
        return True
    return bool(_RAW_FONT_ID_RE.match(name.strip()))


# --------------------------------------------------------------------------- #
# Real embedded-font-name resolution
# --------------------------------------------------------------------------- #

_embedded_font_family_cache = {}


def _resolve_embedded_font_family(pdf_doc, xref):
    """Reads the real family name out of an embedded font file's own name
    table. Checks IDs 16 (Typographic Family), 21 (WWS Family), 1 (Font Family),
    4 (Full Name), 6 (PostScript Name) across Windows, Unicode, and Mac platforms."""
    cache_key = (id(pdf_doc), xref)
    if cache_key in _embedded_font_family_cache:
        return _embedded_font_family_cache[cache_key]

    family = None
    try:
        from fontTools.ttLib import TTFont
        buf = pdf_doc.extract_font(xref)
        font_bytes = buf[3] if buf else None
        if font_bytes:
            tt = TTFont(io.BytesIO(font_bytes), lazy=True, fontNumber=0)
            name_table = tt.get("name")
            if name_table:
                # Check IDs in priority order: 16 (Preferred Family), 21 (WWS Family), 1 (Family), 4 (Full Name)
                raw_name = (
                    name_table.getDebugName(16)
                    or name_table.getDebugName(21)
                    or name_table.getDebugName(1)
                    or name_table.getDebugName(4)
                )
                if not raw_name:
                    for name_record in name_table.names:
                        if name_record.nameID in (16, 21, 1, 4):
                            try:
                                raw_name = name_record.toUnicode()
                                if raw_name:
                                    break
                            except Exception:
                                pass
                if raw_name:
                    raw_name = raw_name.split("+")[-1].strip()
                    is_narrow = is_condensed_font(raw_name)
                    clean_family = re.sub(
                        r"\s*[-,_]?\s*(Bold|Italic|Oblique|Regular|Light|Medium|Black|"
                        r"SemiBold|Semi Bold|ExtraBold|Thin|Heavy|Demi)\b.*",
                        "", raw_name, flags=re.I
                    ).strip() or raw_name
                    if is_narrow and not is_condensed_font(clean_family):
                        clean_family = clean_family + " Narrow"
                    family = clean_family
    except Exception:
        family = None

    _embedded_font_family_cache[cache_key] = family
    return family


def get_pdf_page_font_map(page: "fitz.Page"):
    """Maps this page's internal PDF font resource names ('F1', 'F2', ...) and
    PostScript basefont names to their REAL embedded/resolved family name."""
    mapping = {}
    try:
        pdf_doc = page.parent
        for f in page.get_fonts(full=True):
            xref, ext, ftype, basefont, resource_name = f[0], f[1], f[2], f[3], f[4]
            family = _resolve_embedded_font_family(pdf_doc, xref)
            if not family and basefont:
                clean_base = basefont.split("+")[-1]
                family = map_font(clean_base)
            if family:
                if resource_name:
                    mapping[resource_name] = family
                if basefont:
                    mapping[basefont] = family
                    mapping[basefont.split("+")[-1]] = family
    except Exception:
        pass
    return mapping


def resolve_font(detected_font_name: str, font_override=None, flags: int = 0) -> str:
    if font_override and font_override.get("font"):
        return font_override["font"]
    return map_font(detected_font_name, flags=flags)


def resolve_size(detected_size_pt: float, font_override=None) -> float:
    if font_override and font_override.get("size"):
        return float(font_override["size"])
    return detected_size_pt


def is_condensed_font(raw_font: str) -> bool:
    """True if the PDF's font name indicates a narrow/condensed geometric variant."""
    if not raw_font:
        return False
    low = raw_font.lower()
    return any(m in low for m in _CONDENSED_MARKERS)


def classify_font_fallback(clean_name: str, flags: int = 0) -> str:
    """Classifies an unrecognized font into a suitable standard Word font based on typography clues."""
    low = clean_name.lower()

    # 1. Condensed / Narrow check (by name)
    if is_condensed_font(clean_name):
        return "Arial Narrow"

    # 2. Monospace check (by name)
    if "courier" in low:
        return "Courier New"
    if any(k in low for k in ("mono", "code", "console", "typewriter")):
        return "Consolas"

    # 3. Serif check (by name)
    if any(k in low for k in ("serif", "roman", "antiqua", "mincho", "song", "times", "garamond", "cambria", "georgia", "baskerville", "palatino")):
        return "Times New Roman"

    # 4. Script / Calligraphic check (by name)
    if any(k in low for k in ("script", "hand", "brush", "calligraph")):
        return "Segoe Script"

    # 5. Preserve the embedded font's own family name whenever it looks like
    # a real name (e.g. "LabGrotesque-Medium", "Montserrat", "Bahnschrift").
    # This MUST come before the PDF-flags-based guess below: PyMuPDF's
    # per-span `flags` bitmask (bit 2 = "serifed") is derived from each
    # font's own FontDescriptor, and many custom/subset brand fonts
    # (LabGrotesque, Montserrat, Gotham, ...) have that bit set incorrectly
    # despite being plain sans-serif faces. Trusting the flag over an
    # actual, legible family name silently swapped clearly-sans headings
    # for "Times New Roman" in the Rhenus brochure -- this is why headings
    # rendered as serif even though the source PDF is fully sans-serif.
    if not _looks_like_raw_font_resource_id(clean_name) and len(clean_name) >= 3:
        return clean_name

    # 6. Last resort -- the font name itself is meaningless (a raw PDF font
    # resource id such as "F1" / "C1_0"), so fall back to the flags-based
    # heuristic since there is nothing better to go on.
    if flags & (1 << 3):
        return "Consolas"
    if flags & (1 << 2):
        return "Times New Roman"

    return "Calibri"


def map_font(raw_font: str, flags: int = 0) -> str:
    if not raw_font:
        return "Calibri"
    clean = raw_font.split("+")[-1].strip()
    condensed = is_condensed_font(clean)

    # Clean style suffixes while preserving base name
    clean_base = re.sub(
        r"[-,_]?\s*(Bold|Italic|Oblique|Regular|Light|Medium|Black|Semibold|SemiBold|Thin|Heavy|ExtraBold|Demi)\b.*",
        "", clean, flags=re.I
    ).strip() or clean

    key = re.sub(r"[^a-z0-9]", "", clean_base.lower())
    if key in FONT_SUBSTITUTES:
        return FONT_SUBSTITUTES[key]

    full_key = re.sub(r"[^a-z0-9]", "", clean.lower())
    if full_key in FONT_SUBSTITUTES:
        return FONT_SUBSTITUTES[full_key]

    if condensed:
        base_key = re.sub(r"(narrow|condensed|cond|compressed)", "", key)
        if base_key in ("times", "timesnewroman", "georgia", "cambria", "garamond", "serif"):
            return "Times New Roman"
        return "Arial Narrow"

    return classify_font_fallback(clean_base, flags=flags)


# --------------------------------------------------------------------------- #
# List detection & Hyperlinks
# --------------------------------------------------------------------------- #

_BULLET_RE = re.compile(r"^[\u2022\u2023\u25E6\u2043\u2219\u00B7\-\*\u25CF\u25AA]\s+")
_NUMBER_RE = re.compile(r"^(\d{1,3}|[a-zA-Z])[.\)]\s+")


def detect_list_marker(text: str):
    if not text:
        return None, text
    if _BULLET_RE.match(text):
        return "List Bullet", _BULLET_RE.sub("", text, count=1)
    if _NUMBER_RE.match(text):
        return "List Number", _NUMBER_RE.sub("", text, count=1)
    return None, text


def extract_hyperlinks(page: "fitz.Page"):
    out = []
    try:
        for link in page.get_links():
            uri = link.get("uri")
            frm = link.get("from")
            if uri and frm:
                out.append({"bbox": tuple(frm), "uri": uri})
    except Exception:
        pass
    return out


def find_hyperlink_for_bbox(bbox, hyperlinks):
    if not hyperlinks or not bbox:
        return None
    x0, y0, x1, y1 = bbox
    best_uri, best_overlap = None, 0.0
    for link in hyperlinks:
        lx0, ly0, lx1, ly1 = link["bbox"]
        ix0, iy0 = max(x0, lx0), max(y0, ly0)
        ix1, iy1 = min(x1, lx1), min(y1, ly1)
        overlap = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if overlap > best_overlap:
            best_overlap, best_uri = overlap, link["uri"]
    return best_uri


def add_hyperlink_run(paragraph, text, url, font_name="Calibri", size_pt=11,
                       bold=False, italic=False, color_rgb=None):
    font_name = map_font(font_name)
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rFonts.set(qn("w:eastAsia"), font_name)
    rFonts.set(qn("w:cs"), font_name)
    rPr.append(rFonts)
    if bold:
        rPr.append(OxmlElement("w:b"))
        rPr.append(OxmlElement("w:bCs"))
    if italic:
        rPr.append(OxmlElement("w:i"))
        rPr.append(OxmlElement("w:iCs"))
    sz_half = str(max(2, int(round(size_pt * 2))))
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), sz_half)
    rPr.append(sz)
    szCs = OxmlElement("w:szCs")
    szCs.set(qn("w:val"), sz_half)
    rPr.append(szCs)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1" if not color_rgb else "%02X%02X%02X" % color_rgb)
    rPr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    new_run.append(rPr)

    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return Run(new_run, paragraph)


def order_reading_flow(items, col_gap_min=24.0, bbox_key=lambda it: it.get("bbox")):
    scored = [it for it in items if bbox_key(it)]
    unscored = [it for it in items if not bbox_key(it)]
    if not scored:
        return items
    by_x = sorted(scored, key=lambda it: bbox_key(it)[0])
    bands = []
    for it in by_x:
        x0, x1 = bbox_key(it)[0], bbox_key(it)[2]
        band = next((b for b in bands if not (x1 < b["x0"] - col_gap_min or x0 > b["x1"] + col_gap_min)), None)
        if band is None:
            bands.append({"x0": x0, "x1": x1, "items": [it]})
        else:
            band["x0"] = min(band["x0"], x0)
            band["x1"] = max(band["x1"], x1)
            band["items"].append(it)
    bands.sort(key=lambda b: b["x0"])
    ordered = []
    for b in bands:
        ordered.extend(sorted(b["items"], key=lambda it: bbox_key(it)[1]))
    return ordered + unscored


def clamp_bbox_to_page(bbox, page_width, page_height, min_size=4.0):
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, page_width - min_size))
    y0 = max(0.0, min(y0, page_height - min_size))
    x1 = max(x0 + min_size, min(x1, page_width))
    y1 = max(y0 + min_size, min(y1, page_height))
    return (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# Step 1: PDF type detection & OCR
# --------------------------------------------------------------------------- #

def is_scanned_page(page: "fitz.Page", text_threshold: int = 20) -> bool:
    text = page.get_text("text").strip()
    return len(text) < text_threshold


def render_page_image(page: "fitz.Page", dpi: int = 200) -> Image.Image:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def detect_layout_regions(page_img: Image.Image, dpi: int, model, imgsz=1024, conf=0.25, device="cpu"):
    img_arr = np.array(page_img)
    results = model.predict(img_arr, imgsz=imgsz, conf=conf, device=device, verbose=False)
    px_to_pt = 72.0 / dpi
    regions = []
    if not results:
        return regions
    r = results[0]
    names = r.names
    boxes = r.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        label = names[cls_id]
        x1, y1, x2, y2 = [float(v) * px_to_pt for v in boxes.xyxy[i].tolist()]
        regions.append({
            "label": label,
            "role": LAYOUT_ROLE.get(label, "body"),
            "conf": float(boxes.conf[i].item()),
            "bbox": (x1, y1, x2, y2),
        })
    regions = order_reading_flow(regions)
    return regions


def bbox_center_inside(inner_bbox, outer_bbox) -> bool:
    cx = (inner_bbox[0] + inner_bbox[2]) / 2
    cy = (inner_bbox[1] + inner_bbox[3]) / 2
    return outer_bbox[0] <= cx <= outer_bbox[2] and outer_bbox[1] <= cy <= outer_bbox[3]


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_native_table_regions(regions, native_table_bboxes, iou_threshold: float = 0.15):
    """Cross-reference PyMuPDF's own vector/text-based table finder
    (page.find_tables(), already used for the preview panel and the
    absolute-position pipeline) against the DocLayout-YOLO regions.

    DocLayout-YOLO occasionally fails to tag a genuine tabular block as
    "table" -- e.g. brochure-style grids that use thin rules instead of a
    full ruled border (as with the Rhenus "LGK" product tables). When that
    happens the block falls through to ordinary body-text handling, which
    interleaves the visually-separate columns into one garbled paragraph
    instead of a real Word table.

    Any native table bbox that doesn't already overlap a YOLO region
    tagged "table" is added as a synthetic table region here, so it goes
    through the same reconstruct_table_element() path as every other
    detected table.
    """
    if not native_table_bboxes:
        return regions

    existing_table_bboxes = [r["bbox"] for r in regions if r.get("role") == "table"]
    merged = list(regions)
    for nb in native_table_bboxes:
        if any(_bbox_iou(nb, tb) >= iou_threshold for tb in existing_table_bboxes):
            continue
        merged.append({"label": "table", "role": "table", "conf": 1.0, "bbox": tuple(nb)})
        existing_table_bboxes.append(tuple(nb))
    return merged


def _rasterize_bbox(page: "fitz.Page", bbox, zoom: float = 3.0):
    try:
        rect = fitz.Rect(bbox)
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            return None
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None


def _estimate_ocr_line_props(page_img: Image.Image, poly_arr: np.ndarray, px_to_pt: float):
    """Estimates font bounding box, point size, and boldness from image ink density."""
    x1, y1 = poly_arr[:, 0].min(), poly_arr[:, 1].min()
    x2, y2 = poly_arr[:, 0].max(), poly_arr[:, 1].max()
    bbox_pt = (x1 * px_to_pt, y1 * px_to_pt, x2 * px_to_pt, y2 * px_to_pt)
    est_size = max(6.0, round((y2 - y1) * px_to_pt * 0.75, 1))

    bold = False
    try:
        w, h = page_img.size
        crop_box = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
        if crop_box[2] > crop_box[0] + 4 and crop_box[3] > crop_box[1] + 4:
            crop = page_img.crop(crop_box).convert("L")
            arr = np.array(crop)
            ink_pixels = np.sum(arr < 140)
            density = ink_pixels / max(1, arr.size)
            if density > 0.27 or est_size >= 16:
                bold = True
    except Exception:
        bold = False

    return bbox_pt, est_size, bold


def extract_native_lines(page: "fitz.Page"):
    raw = page.get_text("dict")
    lines_out = []
    images_out = []
    font_map = get_pdf_page_font_map(page)

    for block in raw.get("blocks", []):
        if block.get("type") == 1:
            bbox = block.get("bbox")
            png_bytes = _rasterize_bbox(page, bbox) if bbox else None
            if png_bytes:
                images_out.append({"bytes": png_bytes, "bbox": bbox, "ext": "png"})
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(s.get("text", "") for s in spans).strip()
            if not text or _is_noise_text(text):
                continue
            first = spans[0]
            flags = first.get("flags", 0)
            raw_font_str = first.get("font", "Calibri")
            clean_font_str = raw_font_str.split("+")[-1].strip()
            bold = bool(flags & 2**4) or "bold" in raw_font_str.lower()
            italic = bool(flags & 2**1) or "italic" in raw_font_str.lower()
            color_int = first.get("color", 0)
            color = ((color_int >> 16) & 255, (color_int >> 8) & 255, color_int & 255)

            resolved_font = (
                font_map.get(raw_font_str)
                or font_map.get(clean_font_str)
                or map_font(clean_font_str, flags=flags)
            )

            lines_out.append({
                "text": text, "size": first.get("size", 11), "font": resolved_font,
                "bold": bold, "italic": italic, "color": color, "bbox": line.get("bbox"),
                "flags": flags,
            })

    try:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                bbox = tuple(rects[0])
                png_bytes = _rasterize_bbox(page, bbox)
                if png_bytes:
                    images_out.append({"bytes": png_bytes, "bbox": bbox, "ext": "png"})
            except Exception:
                continue
    except Exception:
        pass

    return lines_out, images_out


def ocr_lines(page_img: Image.Image, dpi: int, ocr_engine):
    img_arr = np.array(page_img)
    px_to_pt = 72.0 / dpi
    lines_out = []

    results = ocr_engine.predict(img_arr)
    for res in results:
        data = res.json if hasattr(res, "json") else res
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        texts = data.get("rec_texts", []) if isinstance(data, dict) else []
        polys = data.get("rec_polys", data.get("rec_boxes", [])) if isinstance(data, dict) else []
        scores = data.get("rec_scores", []) if isinstance(data, dict) else []

        for idx, text in enumerate(texts):
            if not text or not text.strip() or _is_noise_text(text):
                continue
            poly = polys[idx] if idx < len(polys) else None
            if poly is None:
                continue
            poly_arr = np.array(poly).reshape(-1, 2)
            bbox_pt, est_size, bold = _estimate_ocr_line_props(page_img, poly_arr, px_to_pt)
            lines_out.append({
                "text": text.strip(), "size": est_size, "font": "Arial" if bold else "Calibri",
                "bold": bold, "italic": False, "color": (0, 0, 0), "bbox": bbox_pt,
                "score": scores[idx] if idx < len(scores) else None,
            })
    return lines_out


# --------------------------------------------------------------------------- #
# Row-span / column-span aware HTML table parsing
#
# PP-StructureV3 returns each recognized table as an HTML string
# (<table><tr><td rowspan=.. colspan=..>...). The previous implementation
# ran this through pandas.read_html(), which forward-fills merged cells --
# every position a rowspan/colspan covers ends up with the *same duplicated
# text*, and the span information itself is thrown away. Downstream, the
# DOCX writer had no way to tell a genuinely merged cell apart from three
# coincidentally-identical cells, so it always wrote a plain n_rows x n_cols
# grid with the text repeated in every covered position -- never a real
# merged Word cell.
#
# The parser below preserves the HTML's rowspan/colspan exactly and lays
# every cell out onto a dense grid the same shape as the visual table: each
# spanning cell is recorded once at its "origin" position with its
# rowspan/colspan, and every grid position it covers is marked as covered
# (with a pointer back to the origin) instead of holding a text copy. The
# DOCX writer (docx_reconstruction.reconstruct_table) reads this grid and
# performs real cell merges, so the exported table matches the detected
# structure cell-for-cell instead of flattening it.
# --------------------------------------------------------------------------- #

def _safe_int(value, default: int = 1) -> int:
    try:
        v = int(float(value))
        return v if v > 0 else default
    except Exception:
        return default


class _HTMLTableSpanParser(HTMLParser):
    """Parses a single <table> (as emitted by PP-StructureV3) into a list
    of raw rows, where each row is a list of cell dicts:
        {"text": str, "rowspan": int, "colspan": int}
    in original HTML order -- no grid layout yet, no duplication.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self._cur_row = None
        self._cur_cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        a = dict(attrs)
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = {
                "text": "",
                "rowspan": _safe_int(a.get("rowspan"), 1),
                "colspan": _safe_int(a.get("colspan"), 1),
            }
        elif tag == "br" and self._cur_cell is not None:
            # Keep wrapped lines from gluing together into one word.
            self._cur_cell["text"] += " "

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._cur_cell is not None:
            self._cur_cell["text"] = " ".join(self._cur_cell["text"].split())
            if self._cur_row is not None:
                self._cur_row.append(self._cur_cell)
            self._cur_cell = None
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell["text"] += data


def _layout_html_rows_to_grid(raw_rows):
    """Lay HTML rows (with rowspan/colspan, in source order) onto a dense
    visual grid.

    Returns a list of rows, each a list of grid cells (one per visual
    column, every row the same width). A cell is either:
      - an "origin" cell:  {"text", "rowspan", "colspan", "is_origin": True}
      - a "covered" cell:  {"is_origin": False, "origin": (row, col)}
        pointing back at the origin cell that spans over this position.
    """
    if not raw_rows:
        return []

    n_source_rows = len(raw_rows)
    # col -> (rows_still_reserved_after_current_row, origin_row, origin_col)
    tracker = {}
    grid_rows = [dict() for _ in range(n_source_rows)]

    for r in range(n_source_rows):
        row_cells = list(raw_rows[r])
        cell_idx = 0
        col = 0
        new_entries = {}

        while cell_idx < len(row_cells) or col in tracker:
            if col in tracker:
                _remaining, o_r, o_c = tracker[col]
                grid_rows[r][col] = {"is_origin": False, "origin": (o_r, o_c)}
                col += 1
                continue

            cell = row_cells[cell_idx]
            cell_idx += 1
            rowspan = max(1, cell.get("rowspan", 1))
            colspan = max(1, cell.get("colspan", 1))
            origin_coord = (r, col)
            origin_dict = {
                "text": cell.get("text", ""),
                "rowspan": rowspan,
                "colspan": colspan,
                "is_origin": True,
            }

            for dc in range(colspan):
                cc = col + dc
                grid_rows[r][cc] = origin_dict if dc == 0 else {
                    "is_origin": False, "origin": origin_coord,
                }
                if rowspan > 1:
                    new_entries[cc] = (rowspan - 1, origin_coord[0], origin_coord[1])

            col += colspan

        # Age reservations that were already active coming into this row;
        # entries created *this* row (new_entries) start counting next row.
        aged = {}
        for c, (remaining, o_r, o_c) in tracker.items():
            if c in new_entries:
                continue
            nr = remaining - 1
            if nr > 0:
                aged[c] = (nr, o_r, o_c)
        tracker = {**aged, **new_entries}

    n_cols = max((max(row.keys()) + 1 for row in grid_rows if row), default=0)
    if n_cols == 0:
        return []

    dense = []
    for row in grid_rows:
        dense_row = [
            row.get(c) or {"text": "", "rowspan": 1, "colspan": 1, "is_origin": True}
            for c in range(n_cols)
        ]
        dense.append(dense_row)
    return dense


def parse_html_table_with_spans(html_str: str):
    """Parse a PP-StructureV3 HTML table into a dense row/col grid that
    mirrors its merged-cell structure exactly (see module note above)."""
    parser = _HTMLTableSpanParser()
    try:
        parser.feed(html_str)
    except Exception:
        return []
    return _layout_html_rows_to_grid(parser.rows)


def extract_tables_pp_structure(page_img: Image.Image, dpi: int, table_engine, region_bboxes=None):
    import pandas as pd
    px_to_pt = 72.0 / dpi
    tables = []

    def crop_for(bbox_pt):
        if bbox_pt is None:
            return page_img
        dpi_scale = dpi / 72.0
        x1, y1, x2, y2 = [int(v * dpi_scale) for v in bbox_pt]
        pad = 4
        w, h = page_img.size
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return page_img.crop((x1, y1, x2, y2))

    targets = region_bboxes if region_bboxes else [None]

    for bbox_pt in targets:
        img_arr = np.array(crop_for(bbox_pt))
        try:
            output = table_engine.predict(img_arr)
        except Exception:
            continue
        for res in output:
            data = res.json if hasattr(res, "json") else res
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            table_res_list = data.get("table_res_list", []) if isinstance(data, dict) else []
            for t in table_res_list:
                html_str = t.get("pred_html") or t.get("html")
                if not html_str:
                    continue

                # Primary path: span-aware parse -- preserves rowspan/colspan
                # so the DOCX writer can perform real cell merges.
                rows = parse_html_table_with_spans(html_str)

                if not rows:
                    # Fallback for HTML pandas can read but our parser
                    # couldn't (e.g. unexpected markup) -- degrades to the
                    # old flattened behavior rather than dropping the table.
                    try:
                        dfs = pd.read_html(io.StringIO(html_str))
                    except Exception:
                        continue
                    if not dfs:
                        continue
                    df = dfs[0].fillna("")
                    rows = [list(map(str, df.columns))] + df.astype(str).values.tolist()

                tables.append({"bbox": bbox_pt, "rows": rows})

    return tables


def translate_text(text: str, target_lang: str, translator_fn=None) -> str:
    if not target_lang or target_lang == "None (keep original)":
        return text
    if translator_fn is None:
        return text
    try:
        return translator_fn(text, target_lang)
    except Exception:
        return text


# --------------------------------------------------------------------------- #
# Page-level orchestration (Reflow mode)
# --------------------------------------------------------------------------- #

def group_lines_into_paragraphs(lines, gap_multiplier=1.6):
    if not lines:
        return []
    lines_sorted = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    groups = [[lines_sorted[0]]]
    for prev, ln in zip(lines_sorted, lines_sorted[1:]):
        prev_bottom = prev["bbox"][3]
        line_height = max(1.0, prev["bbox"][3] - prev["bbox"][1])
        gap = ln["bbox"][1] - prev_bottom
        is_list_boundary = (detect_list_marker(ln["text"])[0] is not None or
                             detect_list_marker(prev["text"])[0] is not None)
        if gap > line_height * gap_multiplier or is_list_boundary:
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


def build_table_from_lines(lines, y_tol=6.0):
    if not lines:
        return None
    xs = sorted(set(round(l["bbox"][0]) for l in lines))
    if len(xs) < 2:
        return None
    gaps = sorted(((xs[i + 1] - xs[i], xs[i]) for i in range(len(xs) - 1)), reverse=True)
    biggest_gap, gap_start = gaps[0]
    if biggest_gap < 15:
        return None

    threshold = gap_start + biggest_gap / 2
    left = [l for l in lines if l["bbox"][0] <= threshold]
    right = [l for l in lines if l["bbox"][0] > threshold]
    if not left or not right:
        return None

    bands = []
    for ln in sorted(lines, key=lambda l: l["bbox"][1]):
        y0 = ln["bbox"][1]
        band = next((b for b in bands if abs(b["y"] - y0) <= y_tol), None)
        if band is None:
            bands.append({"y": y0, "lines": [ln]})
        else:
            band["lines"].append(ln)

    rows = []
    for band in bands:
        left_text = " ".join(l["text"] for l in band["lines"] if l["bbox"][0] <= threshold)
        right_text = " ".join(l["text"] for l in band["lines"] if l["bbox"][0] > threshold)
        if left_text or right_text:
            rows.append([left_text, right_text])
    return rows or None


def _collect_lines_in_bbox(lines, bbox, used_line_idx):
    collected = []
    for idx, ln in enumerate(lines):
        if idx in used_line_idx:
            continue
        if bbox_center_inside(ln["bbox"], bbox):
            collected.append(ln)
            used_line_idx.add(idx)
    return collected


def build_page_elements(lines, images, regions, tables_from_layout_tables,
                         page_height=None, hyperlinks=None, report=None):
    elements = []
    used_line_idx = set()
    report = report if report is not None else []
    tables_by_bbox = {t["bbox"]: t["rows"] for t in tables_from_layout_tables if t.get("bbox") is not None}

    def hyperlink_for(bbox):
        return find_hyperlink_for_bbox(bbox, hyperlinks) if hyperlinks else None

    if regions:
        for region in regions:
            role = region["role"]
            bbox = region["bbox"]

            if role == "skip":
                region_lines = _collect_lines_in_bbox(lines, bbox, used_line_idx)
                text = " ".join(ln["text"] for ln in region_lines).strip()
                if not text:
                    continue
                if page_height:
                    top_y = bbox[1]
                    bottom_y = bbox[3]
                    if top_y < page_height * 0.15:
                        elements.append({"type": "header_footer", "zone": "header", "text": text, "bbox": bbox})
                        continue
                    if bottom_y > page_height * 0.85:
                        elements.append({"type": "header_footer", "zone": "footer", "text": text, "bbox": bbox})
                        continue
                report.append(f"Marginal note kept as body text (didn't match header/footer position): \"{text[:60]}\"")
                elements.append({"type": "text", "role": "body", "bbox": bbox, "lines": region_lines})
                continue

            if role in ("figure", "formula"):
                elements.append({"type": "image_region", "role": role, "bbox": bbox})
                report.append(f"{role.capitalize()} region on page could not be extracted as a picture -- "
                               f"placeholder inserted (see original PDF).")
                continue

            if role == "table":
                region_lines = _collect_lines_in_bbox(lines, bbox, used_line_idx)
                rows = tables_by_bbox.get(bbox) or build_table_from_lines(region_lines)
                if rows:
                    elements.append({"type": "table", "role": "table", "bbox": bbox, "rows": rows})
                else:
                    report.append("A detected table region could not be reconstructed as a Word table -- "
                                   "its text was kept as separate lines instead.")
                    for ln in region_lines:
                        elements.append({"type": "text", "role": "body", "bbox": ln["bbox"], "lines": [ln],
                                          "uri": hyperlink_for(ln["bbox"])})
                continue

            region_lines = _collect_lines_in_bbox(lines, bbox, used_line_idx)
            for group in group_lines_into_paragraphs(region_lines):
                elements.append({"type": "text", "role": role, "bbox": group[0]["bbox"], "lines": group,
                                  "uri": hyperlink_for(group[0]["bbox"])})

        leftover = [ln for i, ln in enumerate(lines) if i not in used_line_idx]
        for group in group_lines_into_paragraphs(leftover):
            elements.append({"type": "text", "role": "body", "bbox": group[0]["bbox"], "lines": group,
                              "uri": hyperlink_for(group[0]["bbox"])})
    else:
        body_lines = lines
        if page_height:
            body_lines = []
            for ln in lines:
                y0, y1 = ln["bbox"][1], ln["bbox"][3]
                if y1 < page_height * 0.10 and ln["text"].strip():
                    elements.append({"type": "header_footer", "zone": "header",
                                      "text": ln["text"], "bbox": ln["bbox"]})
                elif y0 > page_height * 0.90 and ln["text"].strip():
                    elements.append({"type": "header_footer", "zone": "footer",
                                      "text": ln["text"], "bbox": ln["bbox"]})
                else:
                    body_lines.append(ln)
        for group in group_lines_into_paragraphs(body_lines):
            elements.append({"type": "text", "role": "body", "bbox": group[0]["bbox"], "lines": group,
                              "uri": hyperlink_for(group[0]["bbox"])})

    for img in images:
        elements.append({"type": "image_bytes", "role": "figure", "bbox": img["bbox"],
                          "bytes": img["bytes"], "ext": img["ext"]})

    header_footer_els = [el for el in elements if el["type"] == "header_footer"]
    body_els = [el for el in elements if el["type"] != "header_footer"]
    body_els = order_reading_flow(body_els)
    return body_els + header_footer_els


def apply_docx_run_font(run, font_name: str = "Calibri", size_pt: float = 11.0,
                        bold: bool = False, italic: bool = False,
                        color_rgb: tuple = None, underline: bool = False):
    """Applies high-fidelity Word run formatting with full quad-script rFonts and szCs."""
    resolved = map_font(font_name)
    try:
        run.font.name = resolved
    except Exception:
        pass
    try:
        run.font.size = Pt(max(1.0, float(size_pt)))
    except Exception:
        pass
    run.font.bold = bool(bold)
    run.font.italic = bool(italic)
    run.font.underline = bool(underline)
    if color_rgb and color_rgb != (0, 0, 0):
        try:
            run.font.color.rgb = RGBColor(*color_rgb)
        except Exception:
            pass

    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), resolved)
    rFonts.set(qn("w:hAnsi"), resolved)
    rFonts.set(qn("w:eastAsia"), resolved)
    rFonts.set(qn("w:cs"), resolved)

    sz_half = str(max(2, int(round(size_pt * 2))))
    sz = rPr.find(qn("w:sz"))
    if sz is None:
        sz = OxmlElement("w:sz")
        rPr.append(sz)
    sz.set(qn("w:val"), sz_half)

    szCs = rPr.find(qn("w:szCs"))
    if szCs is None:
        szCs = OxmlElement("w:szCs")
        rPr.append(szCs)
    szCs.set(qn("w:val"), sz_half)

    if bold:
        if rPr.find(qn("w:bCs")) is None:
            rPr.append(OxmlElement("w:bCs"))
    if italic:
        if rPr.find(qn("w:iCs")) is None:
            rPr.append(OxmlElement("w:iCs"))


def build_docx(all_page_elements, target_lang=None, translator_fn=None, report=None,
                font_override=None, page_dims_pt=None, custom_page_size_pt=None,
                pdf_bytes=None, trace_mode=False) -> bytes:
    """
    page_dims_pt: optional list of (width_pt, height_pt), one per source PDF
        page, aligned 1:1 with all_page_elements. When given (and no custom
        override), each generated Word page keeps its own source page's
        true dimensions -- a new Word section is started whenever the size
        changes, exactly like the Exact Visual Layout pipeline.
    custom_page_size_pt: optional (width_pt, height_pt). When given, this
        size wins over page_dims_pt and is applied uniformly to every page.
    Both are optional and default to the previous behaviour (python-docx's
    default page size) when omitted, so existing callers are unaffected.
    pdf_bytes / trace_mode: when trace_mode is True, pdf_bytes must be the
        original source PDF. Every generated Word page gets a faded,
        washed-out image of its own original PDF page as a true background
        watermark -- floating behind the editable text on that SAME page
        (not a separate page) -- so the untouched source is always visible
        right through the converted content. Ignored when False/omitted.
    """
    doc = Document()

    trace_pdf_doc = None
    if trace_mode and pdf_bytes:
        try:
            trace_pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception:
            trace_pdf_doc = None

    first_dims = custom_page_size_pt or (page_dims_pt[0] if page_dims_pt else None)
    if first_dims:
        section = doc.sections[0]
        section.page_width = Pt(first_dims[0])
        section.page_height = Pt(first_dims[1])
        section.orientation = WD_ORIENT.LANDSCAPE if first_dims[0] > first_dims[1] else WD_ORIENT.PORTRAIT

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    report = report if report is not None else []

    n_pages = len(all_page_elements)
    counts = {"header": {}, "footer": {}}
    for elements in all_page_elements:
        for el in elements:
            if el["type"] != "header_footer":
                continue
            norm = re.sub(r"\d+", "#", el["text"]).strip()
            entry = counts[el["zone"]].setdefault(norm, [0, el["text"]])
            entry[0] += 1

    header_norm = footer_norm = None
    if n_pages >= 2:
        min_repeats = max(2, (n_pages + 1) // 2)
        best_header = max(counts["header"].items(), key=lambda kv: kv[1][0], default=None)
        best_footer = max(counts["footer"].items(), key=lambda kv: kv[1][0], default=None)
        if best_header and best_header[1][0] >= min_repeats:
            header_norm = best_header[0]
            doc.sections[0].header.paragraphs[0].text = best_header[1][1]
        if best_footer and best_footer[1][0] >= min_repeats:
            footer_norm = best_footer[0]
            doc.sections[0].footer.paragraphs[0].text = best_footer[1][1]

    current_size = (round(first_dims[0]), round(first_dims[1])) if first_dims else None

    for page_idx, elements in enumerate(all_page_elements):
        if page_idx > 0:
            if (not custom_page_size_pt) and page_dims_pt:
                idx = min(page_idx, len(page_dims_pt) - 1)
                pd = page_dims_pt[idx]
                page_size = (round(pd[0]), round(pd[1]))
                if page_size != current_size:
                    section = doc.add_section(WD_SECTION.NEW_PAGE)
                    section.page_width = Pt(pd[0])
                    section.page_height = Pt(pd[1])
                    section.orientation = WD_ORIENT.LANDSCAPE if pd[0] > pd[1] else WD_ORIENT.PORTRAIT
                    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
                        setattr(section, m, Inches(0.75))
                    current_size = page_size
                else:
                    doc.add_page_break()
            else:
                # Custom uniform size (or no detected dims at all) -- a
                # simple page break keeps the whole document in one section.
                doc.add_page_break()

        if trace_pdf_doc is not None and page_idx < len(trace_pdf_doc):
            if custom_page_size_pt:
                wm_w, wm_h = custom_page_size_pt
            elif page_dims_pt:
                pd = page_dims_pt[min(page_idx, len(page_dims_pt) - 1)]
                wm_w, wm_h = pd
            else:
                src_page = trace_pdf_doc[page_idx]
                wm_w, wm_h = src_page.rect.width, src_page.rect.height
            try:
                trace_png = render_pdf_page_trace_image(trace_pdf_doc[page_idx])
                wm_anchor = doc.add_paragraph()
                wm_pr = wm_anchor._p.get_or_add_pPr()
                wm_pr.append(parse_xml(
                    '<w:spacing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                    'w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
                ))
                add_page_watermark_image(wm_anchor, trace_png, wm_w, wm_h)
            except Exception as e:
                report.append(f"Page {page_idx + 1}: watermark could not be generated: {e}")

        for el in elements:
            if el["type"] == "header_footer":
                norm = re.sub(r"\d+", "#", el["text"]).strip()
                promoted_norm = header_norm if el["zone"] == "header" else footer_norm
                if norm == promoted_norm:
                    continue
                p = doc.add_paragraph()
                run = p.add_run(sanitize_xml_text(el["text"]))
                apply_docx_run_font(run, "Calibri", size_pt=9.0)
                continue

            if el["type"] == "text":
                full_text = sanitize_xml_text(" ".join(ln["text"] for ln in el["lines"]))
                translated = sanitize_xml_text(translate_text(full_text, target_lang, translator_fn))
                first = el["lines"][0]
                flags = first.get("flags", 0)
                font_name = resolve_font(first["font"], font_override, flags=flags)
                size_pt = resolve_size(max(6, min(72, first["size"])), font_override)
                uri = el.get("uri")

                list_style, cleaned = (None, translated) if el["role"] in ("heading", "caption") \
                    else detect_list_marker(translated)

                if el["role"] == "heading":
                    p = doc.add_heading(level=1)
                    # 35. A heading must never be stranded alone at the
                    # bottom of a page, separated from the text that follows.
                    set_keep_with_next(p)
                    set_keep_lines_together(p)
                    if uri:
                        add_hyperlink_run(p, cleaned, uri, font_name=font_name, size_pt=size_pt, bold=True)
                    else:
                        run = p.add_run(cleaned)
                        apply_docx_run_font(run, font_name=font_name, size_pt=size_pt, bold=True, color_rgb=first.get("color"))
                elif el["role"] == "caption":
                    p = doc.add_paragraph()
                    caption_size_pt = size_pt if font_override else max(6, min(72, first["size"] * 0.9))
                    if uri:
                        add_hyperlink_run(p, cleaned, uri, font_name=font_name, size_pt=caption_size_pt, italic=True)
                    else:
                        run = p.add_run(cleaned)
                        apply_docx_run_font(run, font_name=font_name, size_pt=caption_size_pt, italic=True, color_rgb=first.get("color"))
                else:
                    p = doc.add_paragraph(style=list_style) if list_style else doc.add_paragraph()
                    if uri:
                        add_hyperlink_run(p, cleaned, uri, font_name=font_name, size_pt=size_pt,
                                          bold=first["bold"], italic=first["italic"], color_rgb=first.get("color"))
                    else:
                        run = p.add_run(cleaned)
                        apply_docx_run_font(run, font_name=font_name, size_pt=size_pt,
                                            bold=first["bold"], italic=first["italic"], color_rgb=first.get("color"))

            elif el["type"] == "table":
                # Delegate table creation/reconstruction to the dedicated
                # document reconstruction module.  The module receives the
                # complete table element so it can preserve structure and
                # apply translation/font overrides consistently.
                try:
                    reconstruct_table_element(
                        doc=doc,
                        el=el,
                        target_lang=target_lang,
                        translator_fn=translator_fn,
                        font_override=font_override,
                    )
                except Exception as e:
                    report.append(f"Table reconstruction failed: {e}")

            elif el["type"] == "image_bytes":
                try:
                    bbox = el["bbox"]
                    width_pt = bbox[2] - bbox[0] if bbox else None
                    width_in = (width_pt / 72.0) if width_pt else 3
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = p.add_run()
                    run.add_picture(io.BytesIO(el["bytes"]), width=Inches(min(max(width_in, 0.5), 6.5)))
                except Exception as e:
                    report.append(f"An image could not be embedded (unsupported codec) and was skipped: {e}")
                    continue

            elif el["type"] == "image_region":
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(f"[{el['role'].capitalize()} region detected -- see original PDF page]")
                run.font.italic = True
                run.font.size = Pt(9)

    _append_report_page(doc, report)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def _append_report_page(doc, report):
    if not report:
        return
    doc.add_page_break()
    doc.add_heading("Processing Report", level=1)
    doc.add_paragraph(
        "The following items could not be converted with full fidelity and were "
        "handled with a documented fallback instead of being silently dropped:"
    )
    seen = set()
    for line in report:
        if line in seen:
            continue
        seen.add(line)
        doc.add_paragraph(sanitize_xml_text(line), style="List Bullet")


def process_pdf_full(pdf_bytes: bytes, dpi=200, layout_conf=0.25, device="cpu",
                      run_tables=True, ocr_lang="en", progress_cb=None, report=None):
    layout_model = get_layout_model() if DOCLAYOUT_AVAILABLE and HF_HUB_AVAILABLE else None
    ocr_engine = get_ocr_engine(ocr_lang) if PADDLEOCR_AVAILABLE else None
    table_engine = get_table_engine() if (PADDLEOCR_AVAILABLE and run_tables) else None
    report = report if report is not None else []

    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_elements = []
    page_meta = []

    for i, page in enumerate(pdf_doc):
        if progress_cb:
            progress_cb(i + 1, len(pdf_doc))

        scanned = is_scanned_page(page)
        page_img = render_page_image(page, dpi=dpi)
        hyperlinks = extract_hyperlinks(page)

        regions = []
        if layout_model is not None:
            regions = detect_layout_regions(page_img, dpi, layout_model, conf=layout_conf, device=device)

        # Safety net: add any table PyMuPDF's own vector/text-based table
        # finder detects but that DocLayout-YOLO missed, so genuine grids
        # (e.g. rule-separated product tables without a full border) still
        # reconstruct as real Word tables instead of garbled flow text.
        try:
            native_table_bboxes = _detect_native_tables(page)
        except Exception:
            native_table_bboxes = []
        if native_table_bboxes:
            regions = merge_native_table_regions(regions, native_table_bboxes)

        images = []
        if scanned:
            if ocr_engine is not None:
                lines = ocr_lines(page_img, dpi, ocr_engine)
            else:
                lines = [{"text": "[PaddleOCR not installed -- see requirements-full.txt]",
                          "size": 11, "font": "Calibri", "bold": False, "italic": False,
                          "color": (0, 0, 0), "bbox": (0, 0, 0, 0)}]
                report.append(f"Page {i + 1} is scanned but PaddleOCR isn't installed -- text could not be recovered.")
        else:
            lines, images = extract_native_lines(page)

        table_bboxes = [r["bbox"] for r in regions if r["role"] == "table"]
        tables = []
        if table_engine is not None:
            tables = extract_tables_pp_structure(
                page_img, dpi, table_engine,
                region_bboxes=table_bboxes if table_bboxes else None,
            )
        elif table_bboxes:
            for bbox in table_bboxes:
                tables.append({"bbox": bbox, "rows": [["[Table detected -- install "
                                                        "PP-StructureV3 to extract cell data]"]]})
                report.append(f"Page {i + 1}: a table region was detected but PP-StructureV3 isn't "
                               f"installed -- cell data could not be extracted.")

        elements = build_page_elements(lines, images, regions, tables,
                                        page_height=page.rect.height, hyperlinks=hyperlinks, report=report)
        all_elements.append(elements)
        page_meta.append({"scanned": scanned, "n_regions": len(regions), "n_tables": len(tables)})

    pdf_doc.close()
    return all_elements, page_meta, report


# =============================================================================
# EXACT VISUAL LAYOUT MODE
#
# Position-based layout matching the source PDF:
# Every background band, border, image, and text line is placed at its exact
# original coordinates.
# =============================================================================

def _is_boxing_hint_color(r: int, g: int, b: int) -> bool:
    """
    Classify an RGB color as a "boxing hint" annotation color (blue outline
    used purely to mark which paragraph lines should be grouped into a
    single reconstructed table -- see add_absolute_paragraph_table's
    hint_boxes support). These rectangles are never drawn in the output;
    they are only used to decide grouping + width/indentation.

    Heuristic: blue channel clearly dominant over red and green, and bright
    enough not to be confused with near-black borders. Covers common
    "annotation blue" shades (e.g. Word's default shape outline #4472C4,
    hyperlink blue #0563C1, PowerPoint accent blues, etc).
    """
    return b > 90 and b > r + 25 and b > g + 15


def extract_drawings(page: "fitz.Page", include_drawings: bool = True):
    """
    Returns (rects, hint_boxes):
      - rects: normal fills/border-lines to be rendered visually, unchanged
        from before.
      - hint_boxes: unfilled, blue-stroked, fully rectangular annotation
        boxes (4 real sides) that are NEVER rendered. They exist purely as
        a layout signal: any paragraph lines falling inside one of these
        boxes must be reconstructed as a SINGLE table matching the box's
        width and left indentation (see build_docx_exact_layout).
    """
    if not include_drawings:
        return [], []
    rects = []
    hint_boxes = []
    page_w, page_h = page.rect.width, page.rect.height

    for d in page.get_drawings():
        color = d.get("color")
        fill = d.get("fill")
        rect = d.get("rect")
        width = d.get("width") or 0
        path_type = d.get("type") or ""
        fill_opacity = d.get("fill_opacity")

        has_real_fill = (
            fill is not None
            and "f" in path_type
            and (fill_opacity is None or fill_opacity > 0)
        )
        if rect is None:
            continue

        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        rw, rh = x1 - x0, y1 - y0

        if rw <= 0.5 and rh <= 0.5:
            continue

        if has_real_fill:
            r, g, b = [max(0, min(255, int(round(c * 255)))) for c in fill]

            # Preserve section banners and header bars (width >= 80pt, height 8..60pt)
            is_banner_bar = (rw >= 80 and 8 <= rh <= 60)

            # Filter ONLY huge full-page near-black dark masks
            is_huge_dark_mask = (r < 25 and g < 25 and b < 25 and rw > page_w * 0.7 and rh > page_h * 0.4)

            if is_huge_dark_mask and not is_banner_bar:
                continue

            rects.append({"bbox": (x0, y0, x1, y1), "color_hex": f"{r:02X}{g:02X}{b:02X}"})

        elif color is not None and width and width > 0.3:
            r, g, b = [max(0, min(255, int(round(c * 255)))) for c in color]
            hex_color = f"{r:02X}{g:02X}{b:02X}"

            # A real 2-D unfilled rectangle outline (both sides present, not
            # just a single ruled line) drawn in a "boxing hint" blue is a
            # grouping annotation, not a visible design element -- capture
            # it separately and skip normal rendering entirely.
            is_real_box = rw > 4 and rh > 4
            if is_real_box and _is_boxing_hint_color(r, g, b):
                hint_boxes.append({"bbox": (x0, y0, x1, y1), "color_hex": hex_color})
                continue

            if abs(y1 - y0) < 0.5 and (x1 - x0) > 0.5:
                cy = (y0 + y1) / 2
                rects.append({"bbox": (x0, cy - width / 2, x1, cy + width / 2), "color_hex": hex_color})
            elif abs(x1 - x0) < 0.5 and (y1 - y0) > 0.5:
                cx = (x0 + x1) / 2
                rects.append({"bbox": (cx - width / 2, y0, cx + width / 2, y1), "color_hex": hex_color})

    return rects, hint_boxes


def extract_lines_multirun(page: "fitz.Page"):
    raw = page.get_text("dict")
    lines_out = []
    images_out = []
    seen_image_bboxes = set()

    # Resolve this page's font resource ids ("F1", "F2"...) to their real
    # embedded family names up front -- see get_pdf_page_font_map for why
    # this matters (generic resource ids otherwise get written into the docx
    # verbatim and cause width-estimate/actual-render mismatches -> text
    # wrapping/overlap). Falls back to the old resource-id string for any
    # font this can't resolve, so behavior is unchanged where it can't help.
    font_map = get_pdf_page_font_map(page)

    def _bbox_key(bbox):
        return tuple(round(v, 1) for v in bbox)

    for block in raw.get("blocks", []):
        if block.get("type") == 1:
            bbox = block.get("bbox")
            if bbox and _bbox_key(bbox) not in seen_image_bboxes:
                seen_image_bboxes.add(_bbox_key(bbox))
                png_bytes = _rasterize_bbox(page, bbox)
                if png_bytes:
                    images_out.append({"bytes": png_bytes, "bbox": bbox, "ext": "png"})
            continue

        for line in block.get("lines", []):
            spans_raw = line.get("spans", [])
            if not spans_raw:
                continue

            spans_out = []
            for idx, s in enumerate(spans_raw):
                text = s.get("text", "")
                if not text or _is_noise_text(text):
                    continue

                flags = s.get("flags", 0)
                bold = bool(flags & 2**4) or "bold" in s.get("font", "").lower()
                italic = bool(flags & 2**1) or "italic" in s.get("font", "").lower()
                color_int = s.get("color", 0)
                color_hex = f"{(color_int >> 16) & 255:02X}{(color_int >> 8) & 255:02X}{color_int & 255:02X}"
                raw_font_str = s.get("font", "Calibri")
                clean_font_str = raw_font_str.split("+")[-1].strip()

                # Prefer the REAL family name read from the embedded font file
                font = (
                    font_map.get(raw_font_str)
                    or font_map.get(clean_font_str)
                    or map_font(clean_font_str, flags=flags)
                )

                # Word space separation check: Insert space between adjacent spans if missing
                if spans_out:
                    prev_text = spans_out[-1]["text"]
                    if prev_text and not prev_text[-1].isspace() and not text[0].isspace():
                        if prev_text[-1].isalnum() and (text[0].isalnum() or text[0] in "({[$\"'"):
                            spans_out[-1]["text"] += " "

                spans_out.append({
                    "text": text, "size": s.get("size", 11), "font": font,
                    "bold": bold, "italic": italic, "color_hex": color_hex,
                    "bbox": s.get("bbox", line.get("bbox")),
                    "flags": flags,
                })

            if spans_out:
                lines_out.append({"bbox": line.get("bbox"), "spans": spans_out})

    try:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for r in rects:
                    bbox = tuple(r)
                    if _bbox_key(bbox) in seen_image_bboxes:
                        continue
                    seen_image_bboxes.add(_bbox_key(bbox))
                    png_bytes = _rasterize_bbox(page, bbox)
                    if png_bytes:
                        images_out.append({"bytes": png_bytes, "bbox": bbox, "ext": "png"})
            except Exception:
                continue
    except Exception:
        pass

    return lines_out, images_out


def ocr_lines_multirun(page_img: Image.Image, dpi: int, ocr_engine):
    img_arr = np.array(page_img)
    px_to_pt = 72.0 / dpi
    lines_out = []

    results = ocr_engine.predict(img_arr)
    for res in results:
        data = res.json if hasattr(res, "json") else res
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        texts = data.get("rec_texts", []) if isinstance(data, dict) else []
        polys = data.get("rec_polys", data.get("rec_boxes", [])) if isinstance(data, dict) else []
        scores = data.get("rec_scores", []) if isinstance(data, dict) else []

        for idx, text in enumerate(texts):
            if not text or not text.strip() or _is_noise_text(text):
                continue
            poly = polys[idx] if idx < len(polys) else None
            if poly is None:
                continue
            poly_arr = np.array(poly).reshape(-1, 2)
            bbox_pt, est_size, bold = _estimate_ocr_line_props(page_img, poly_arr, px_to_pt)
            lines_out.append({
                "bbox": bbox_pt,
                "spans": [{"text": text.strip(), "size": est_size, "font": "Arial" if bold else "Calibri",
                           "bold": bold, "italic": False, "color_hex": "000000"}],
            })
    return lines_out


_XML_INVALID_CHARS_RE = re.compile(
    "[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFE\uFFFF]"
)


def sanitize_xml_text(text: str) -> str:
    if not text:
        return text
    return _XML_INVALID_CHARS_RE.sub("", text)


def add_absolute_rect(paragraph, left_pt, top_pt, width_pt, height_pt, fill_hex):
    """Creates a background VML rectangle shape.
    Sets z-index:-1000 and o:behindDoc="1" so Word strictly renders the shape in the
    background layer behind all document text and tables."""
    if width_pt <= 0 or height_pt <= 0:
        return
    xml = (
        '<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:pict xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">'
        f'<v:rect style="position:absolute;z-index:-1000;left:{left_pt}pt;top:{top_pt}pt;'
        f'width:{width_pt}pt;height:{height_pt}pt;'
        'mso-position-horizontal:absolute;mso-position-horizontal-relative:page;'
        'mso-position-vertical:absolute;mso-position-vertical-relative:page" '
        f'fillcolor="#{fill_hex}" stroked="f" o:behindDoc="1" o:allowincell="f"/></w:pict></w:r>'
    )
    paragraph._p.append(parse_xml(xml))



# --------------------------------------------------------------------------- #
# Accurate text-width measurement
# --------------------------------------------------------------------------- #

from PIL import ImageFont as _ImageFont

_ARIAL_NARROW_SCALE = 0.82
_font_cache = {}


def _find_system_font(font_name: str, bold: bool = False):
    """Finds a matching TrueType / OpenType font file on Windows or Linux."""
    key = font_name.lower().strip()
    is_bold = bold

    # Windows Fonts directory
    win_dir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = os.path.join(win_dir, "Fonts")

    win_font_candidates = {
        "arial": ("arialbd.ttf" if is_bold else "arial.ttf"),
        "arial narrow": ("arialbd.ttf" if is_bold else "arial.ttf"),  # scaled
        "calibri": ("calibrib.ttf" if is_bold else "calibri.ttf"),
        "times new roman": ("timesbd.ttf" if is_bold else "times.ttf"),
        "georgia": ("georgiab.ttf" if is_bold else "georgia.ttf"),
        "cambria": ("cambriab.ttf" if is_bold else "cambria.ttc"),
        "segoe ui": ("segoeuib.ttf" if is_bold else "segoeui.ttf"),
        "tahoma": ("tahomabd.ttf" if is_bold else "tahoma.ttf"),
        "verdana": ("verdanab.ttf" if is_bold else "verdana.ttf"),
        "consolas": ("consolab.ttf" if is_bold else "consola.ttf"),
        "courier new": ("courbd.ttf" if is_bold else "cour.ttf"),
    }

    if key in win_font_candidates:
        fn = win_font_candidates[key]
        full_path = os.path.join(fonts_dir, fn)
        if os.path.exists(full_path):
            return full_path

    # Try generic file search in Windows Fonts
    if os.path.isdir(fonts_dir):
        for cand in (f"{key}{'bd' if is_bold else ''}.ttf", f"{key}.ttf", f"{key}.ttc"):
            cand_path = os.path.join(fonts_dir, cand)
            if os.path.exists(cand_path):
                return cand_path

    # Linux fallback paths
    linux_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if is_bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if is_bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for lp in linux_paths:
        if os.path.exists(lp):
            return lp

    return win_font_candidates.get(key, "arial.ttf")


def _get_metric_font(font_name: str, bold: bool, size_pt: float):
    key = (font_name.lower().strip(), bold, round(size_pt * 4))
    if key in _font_cache:
        return _font_cache[key]
    font_target = _find_system_font(font_name, bold)
    target_size = max(4, int(round(size_pt * 4)))
    try:
        font = _ImageFont.truetype(font_target, target_size)
    except Exception:
        try:
            font = _ImageFont.truetype("arial.ttf", target_size)
        except Exception:
            font = _ImageFont.load_default()
    _font_cache[key] = font
    return font


def measure_text_width_pt(text: str, font_name: str, size_pt: float, bold: bool = False) -> float:
    """Real glyph-metric width of `text` set in `font_name` at `size_pt`, in points."""
    if not text:
        return 0.0
    try:
        font = _get_metric_font(font_name, bold, size_pt)
        width = font.getlength(text) / 4.0
        if font_name.lower().strip() == "arial narrow":
            width *= _ARIAL_NARROW_SCALE
        return width
    except Exception:
        ratio = 0.60 if bold else 0.52
        return len(text) * size_pt * ratio


def estimate_text_width_pt(text: str, size_pt: float, bold: bool = False) -> float:
    return measure_text_width_pt(text, "arial", size_pt, bold)


def fit_scale_percent(measured_width_pt: float, available_width_pt: float,
                       floor_percent: int = 70) -> int:
    """Word horizontal character-scale percentage (<w:w>) needed to make a line
    that measures `measured_width_pt` fit inside `available_width_pt`, clamped
    so text never gets squeezed illegibly thin. Returns 100 when it already fits."""
    if measured_width_pt <= 0 or available_width_pt <= 0:
        return 100
    if measured_width_pt <= available_width_pt:
        return 100
    pct = int((available_width_pt / measured_width_pt) * 100)
    return max(floor_percent, min(100, pct))


def add_absolute_textbox_multirun(paragraph, spans, left_pt, top_pt, width_pt, height_pt,
                                   hyperlink_uri=None, font_override=None):
    if not spans:
        return
    runs_xml = []
    est_width = 0.0
    for sp in spans:
        text_esc = html.escape(sanitize_xml_text(sp["text"]))
        flags = sp.get("flags", 0)
        bold_xml = "<w:b/><w:bCs/>" if sp["bold"] else ""
        italic_xml = "<w:i/><w:iCs/>" if sp["italic"] else ""
        # A hyperlink must stay clickable WITHOUT being forced blue/underlined --
        # only apply that styling when the source PDF span itself was already
        # underlined/blue. Otherwise keep the run's own detected color/underline.
        underline_xml = '<w:u w:val="single"/>' if sp.get("underline") else ""
        size_pt = resolve_size(sp["size"], font_override)
        sz_half = max(2, int(round(size_pt * 2)))
        resolved_font_name = resolve_font(sp["font"], font_override, flags=flags)
        font = html.escape(sanitize_xml_text(resolved_font_name))
        color_hex = sp["color_hex"]
        est_width += measure_text_width_pt(sp["text"], resolved_font_name, size_pt, sp["bold"])
        runs_xml.append(
            f'<w:r><w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>'
            f'{bold_xml}{italic_xml}{underline_xml}'
            f'<w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/><w:color w:val="{color_hex}"/></w:rPr>'
            f'<w:t xml:space="preserve">{text_esc}</w:t></w:r>'
        )
    run_block = "".join(runs_xml)
    r_ns = ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    if hyperlink_uri:
        r_id = paragraph.part.relate_to(hyperlink_uri, RT.HYPERLINK, is_external=True)
        run_block = f'<w:hyperlink{r_ns} r:id="{r_id}">{run_block}</w:hyperlink>'
    safe_width_pt = max(width_pt, est_width * 1.25)
    xml = (
        '<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:pict xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">'
        f'<v:shape style="position:absolute;left:{left_pt}pt;top:{top_pt}pt;'
        f'width:{safe_width_pt}pt;height:{height_pt}pt;mso-wrap-style:none;'
        'mso-position-horizontal:absolute;mso-position-horizontal-relative:page;'
        'mso-position-vertical:absolute;mso-position-vertical-relative:page" o:allowincell="f" '
        'stroked="f" filled="f"><v:textbox style="mso-fit-shape-to-text:f" inset="0,0,0,0">'
        f'<w:txbxContent><w:p><w:pPr><w:spacing w:before="0" w:after="0" '
        f'w:line="240" w:lineRule="auto"/></w:pPr>{run_block}</w:p></w:txbxContent>'
        "</v:textbox></v:shape></w:pict></w:r>"
    )
    paragraph._p.append(parse_xml(xml))



def add_absolute_column_textbox(anchor_p, lines, left_pt, top_pt, width_pt, page_height_pt,
                                hyperlinks=None, font_override=None):
    """Place one complete PDF reading column inside ONE Word VML textbox.

    Unlike the old one-row-per-band table approach, the textbox owns the whole
    vertical column. Text therefore continues downward inside the left/right
    column independently instead of making Word start the next block on a new
    table row. The text remains native Word text and can be edited.
    """
    if not lines:
        return

    lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    blocks = group_lines_into_paragraph_blocks(lines)
    if not blocks:
        return

    # Keep the box inside the PDF page. A small safety margin prevents Word's
    # textbox border/rounding from creating a page spill.
    safe_width = max(20.0, min(width_pt, 1000.0))
    safe_height = max(20.0, page_height_pt - max(0.0, top_pt) - 34.0)

    content_xml = _build_paragraph_xml_for_lines(
        lines, left_pt, safe_width, hyperlinks, font_override, anchor_p, top_y=top_pt
    )
    if not content_xml:
        return

    xml = (
        '<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:pict xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">'
        f'<v:shape style="position:absolute;left:{left_pt}pt;top:{top_pt}pt;'
        f'width:{safe_width}pt;height:{safe_height}pt;mso-wrap-style:none;'
        'mso-position-horizontal:absolute;mso-position-horizontal-relative:page;'
        'mso-position-vertical:absolute;mso-position-vertical-relative:page" '
        'o:allowincell="f" stroked="f" filled="f">'
        '<v:textbox style="mso-fit-shape-to-text:f;inset:0,0,0,0">'
        f'<w:txbxContent>{content_xml}</w:txbxContent>'
        '</v:textbox></v:shape></w:pict></w:r>'
    )
    anchor_p._p.append(parse_xml(xml))


def _bbox_intersection_ratio(a, b):
    """Return intersection area divided by area of bbox a."""
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    ia = (ix1 - ix0) * (iy1 - iy0)
    aa = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return ia / aa


def _is_bbox_inside_any_table(bbox, table_bboxes, threshold=0.45):
    return any(_bbox_intersection_ratio(bbox, tb) >= threshold for tb in table_bboxes)


def _clean_table_text(text):
    """Remove PP/PyMuPDF table-finder separator artifacts without changing real text."""
    if not text:
        return ""
    parts = []
    for line in re.split(r"\r?\n", text):
        line = line.strip()
        if not line or _is_noise_text(line):
            continue
        parts.append(line)
    return "\n".join(parts)


def _cell_fill_from_pdf(page, cell_bbox):
    """Sample the PDF background color near the top-left interior of a cell."""
    try:
        x0, y0, x1, y1 = cell_bbox
        # Avoid borders/text. The source document uses flat fills, so a small
        # interior sample is more reliable than trying to infer fills from line drawings.
        samples = []
        for dx, dy in ((4, 3), (8, 3), (4, 6), (8, 6)):
            x = min(max(x0 + dx, x0 + 1), x1 - 1)
            y = min(max(y0 + dy, y0 + 1), y1 - 1)
            samples.append((x, y))
        scale = 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        rgb = []
        for x, y in samples:
            px = min(max(int(round(x * scale)), 0), pix.width - 1)
            py = min(max(int(round(y * scale)), 0), pix.height - 1)
            idx = (py * pix.width + px) * 3
            if idx + 2 < len(pix.samples):
                rgb.append(tuple(pix.samples[idx:idx + 3]))
        if not rgb:
            return "FFFFFF"
        # Median is robust against a sample landing on a glyph/border.
        arr = np.array(rgb, dtype=np.int16)
        med = np.median(arr, axis=0).astype(int)
        return "{:02X}{:02X}{:02X}".format(*med)
    except Exception:
        return "FFFFFF"


def _page_text_spans(page):
    """Return de-duplicated text spans from the complete PDF page."""
    out = []
    seen = set()
    try:
        raw = page.get_text("dict")
    except Exception:
        return out
    font_map = get_pdf_page_font_map(page)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                text = s.get("text", "")
                if not text or _is_noise_text(text):
                    continue
                bbox = tuple(s.get("bbox", line.get("bbox", (0, 0, 0, 0))))
                key = (text.strip(), tuple(round(v, 2) for v in bbox), s.get("font", ""), s.get("size", 0))
                if key in seen:
                    continue
                seen.add(key)
                flags = s.get("flags", 0)
                font_raw = s.get("font", "Calibri")
                clean_font = font_raw.split("+")[-1].strip()
                font = (
                    font_map.get(font_raw)
                    or font_map.get(clean_font)
                    or map_font(clean_font, flags=flags)
                )
                color_int = s.get("color", 0)
                out.append({
                    "text": text,
                    "size": float(s.get("size", 10)),
                    "font": font,
                    "bold": bool(flags & 2**4) or "bold" in font_raw.lower(),
                    "italic": bool(flags & 2**1) or "italic" in font_raw.lower(),
                    "color_hex": f"{(color_int >> 16) & 255:02X}{(color_int >> 8) & 255:02X}{color_int & 255:02X}",
                    "bbox": bbox,
                    "flags": flags,
                })
    return out


def _spans_for_cell(all_spans, cell_bbox):
    """Assign complete-page PDF spans to a cell using span center coordinates."""
    x0, y0, x1, y1 = cell_bbox
    result = []
    for sp in all_spans:
        sx0, sy0, sx1, sy1 = sp["bbox"]
        cx = (sx0 + sx1) / 2.0
        cy = (sy0 + sy1) / 2.0
        if x0 - 1.0 <= cx <= x1 + 1.0 and y0 - 2.0 <= cy <= y1 + 2.0:
            result.append(sp)
    result.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    return result



def _extract_page_vector_segments(page):
    """Extracts horizontal and vertical line segments from page drawings."""
    h_segments = []
    v_segments = []

    for d in page.get_drawings():
        color = d.get("color") or d.get("fill")
        rect = d.get("rect")
        width = d.get("width") or 1.0

        if color is not None:
            r, g, b = [max(0, min(255, int(round(c * 255)))) for c in color]
            color_hex = f"{r:02X}{g:02X}{b:02X}"
        else:
            color_hex = "D9D9D9"

        sz_val = max(2, min(24, int(round(width * 8))))

        if rect:
            rx0, ry0, rx1, ry1 = rect.x0, rect.y0, rect.x1, rect.y1
            rw = rx1 - rx0
            rh = ry1 - ry0

            if rh <= 3.0 and rw >= 4.0:
                h_segments.append({
                    "y": (ry0 + ry1) / 2.0,
                    "x0": rx0,
                    "x1": rx1,
                    "color": color_hex,
                    "sz": sz_val
                })
            elif rw <= 3.0 and rh >= 4.0:
                v_segments.append({
                    "x": (rx0 + rx1) / 2.0,
                    "y0": ry0,
                    "y1": ry1,
                    "color": color_hex,
                    "sz": sz_val
                })

        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= 1.5 and abs(p1.x - p2.x) >= 4.0:
                    h_segments.append({
                        "y": (p1.y + p2.y) / 2.0,
                        "x0": min(p1.x, p2.x),
                        "x1": max(p1.x, p2.x),
                        "color": color_hex,
                        "sz": sz_val
                    })
                elif abs(p1.x - p2.x) <= 1.5 and abs(p1.y - p2.y) >= 4.0:
                    v_segments.append({
                        "x": (p1.x + p2.x) / 2.0,
                        "y0": min(p1.y, p2.y),
                        "y1": max(p1.y, p2.y),
                        "color": color_hex,
                        "sz": sz_val
                    })

    return h_segments, v_segments


def _has_horizontal_border_between(h_segments, x0, x1, y_top, y_bot, tol=3.0):
    """Returns True if an actual horizontal vector border separates y_top and y_bot."""
    for seg in h_segments:
        if (y_top - tol) <= seg["y"] <= (y_bot + tol):
            overlap = min(x1, seg["x1"]) - max(x0, seg["x0"])
            if overlap >= (x1 - x0) * 0.4:
                return True
    return False


def _detect_cell_borders(h_segments, v_segments, cell_bbox, tol=3.0):
    """Detects border properties (top, bottom, left, right) for a cell."""
    x0, y0, x1, y1 = cell_bbox
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)

    borders = {
        "top": None,
        "bottom": None,
        "left": None,
        "right": None
    }

    for seg in h_segments:
        if abs(seg["y"] - y0) <= tol:
            overlap = min(x1, seg["x1"]) - max(x0, seg["x0"])
            if overlap >= w * 0.35:
                borders["top"] = {"color": seg["color"], "sz": seg["sz"]}
        if abs(seg["y"] - y1) <= tol:
            overlap = min(x1, seg["x1"]) - max(x0, seg["x0"])
            if overlap >= w * 0.35:
                borders["bottom"] = {"color": seg["color"], "sz": seg["sz"]}

    for seg in v_segments:
        if abs(seg["x"] - x0) <= tol:
            overlap = min(y1, seg["y1"]) - max(y0, seg["y0"])
            if overlap >= h * 0.35:
                borders["left"] = {"color": seg["color"], "sz": seg["sz"]}
        if abs(seg["x"] - x1) <= tol:
            overlap = min(y1, seg["y1"]) - max(y0, seg["y0"])
            if overlap >= h * 0.35:
                borders["right"] = {"color": seg["color"], "sz": seg["sz"]}

    return borders


def _detect_line_alignment(line_x0, line_x1, cell_x0, cell_x1):
    """Detect paragraph horizontal justification/alignment from geometry."""
    cell_w = max(1.0, cell_x1 - cell_x0)
    line_w = max(1.0, line_x1 - line_x0)
    left_space = line_x0 - cell_x0
    right_space = cell_x1 - line_x1

    if line_w >= cell_w - 6.0:
        return "left"

    if abs(left_space - right_space) <= max(3.0, cell_w * 0.08) and left_space > 3.0 and right_space > 3.0:
        return "center"
    elif right_space < 4.0 and left_space > 8.0:
        return "right"
    else:
        return "left"


def _add_table_cell_xml(cell_xml, cell_bbox, cell_spans, borders=None, font_override=None,
                        target_lang=None, translator_fn=None, grid_span=None, fill_hex="FFFFFF"):
    """Populate one native Word table cell from already-assigned PDF spans."""
    x0, y0, x1, y1 = cell_bbox
    width_dxa = max(1, int(round((x1 - x0) * 20)))

    left_dxa = 0
    right_dxa = 0
    if cell_spans:
        min_sx0 = min(s["bbox"][0] for s in cell_spans)
        max_sx1 = max(s["bbox"][2] for s in cell_spans)
        left_pad_pt = max(0.0, min(8.0, min_sx0 - x0))
        right_pad_pt = max(0.0, min(8.0, x1 - max_sx1))
        left_dxa = int(round(left_pad_pt * 20))
        right_dxa = int(round(right_pad_pt * 20))

    tc_borders = ['<w:tcBorders>']
    for side in ('top', 'left', 'bottom', 'right'):
        b_info = borders.get(side) if borders else None
        if b_info:
            sz = b_info.get("sz", 4)
            col = b_info.get("color", "D9D9D9")
            tc_borders.append(f'<w:{side} w:val="single" w:sz="{sz}" w:space="0" w:color="{col}"/>')
        else:
            tc_borders.append(f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="D9D9D9"/>')
    tc_borders.append('</w:tcBorders>')

    tc_pr = [
        f'<w:tcW w:w="{width_dxa}" w:type="dxa"/>',
        f'<w:tcMar><w:top w:w="0" w:type="dxa"/><w:left w:w="{left_dxa}" w:type="dxa"/>'
        f'<w:bottom w:w="0" w:type="dxa"/><w:right w:w="{right_dxa}" w:type="dxa"/></w:tcMar>',
        '<w:vAlign w:val="top"/>',
        ''.join(tc_borders),
        f'<w:shd w:val="clear" w:color="auto" w:fill="{fill_hex}"/>'
    ]
    if grid_span:
        tc_pr.append(f'<w:gridSpan w:val="{int(grid_span)}"/>')

    line_groups = []
    for sp in cell_spans:
        if not line_groups:
            line_groups.append([sp])
        else:
            prev_line = line_groups[-1]
            prev_cy = (prev_line[0]["bbox"][1] + prev_line[0]["bbox"][3]) / 2.0
            curr_cy = (sp["bbox"][1] + sp["bbox"][3]) / 2.0
            curr_h = max(4.0, sp["bbox"][3] - sp["bbox"][1])
            if abs(curr_cy - prev_cy) <= curr_h * 0.45 or abs(sp["bbox"][1] - prev_line[0]["bbox"][1]) <= 2.0:
                line_groups[-1].append(sp)
            else:
                line_groups.append([sp])

    for group in line_groups:
        group.sort(key=lambda s: s["bbox"][0])

    p_xml = []
    prev_y1 = None
    for group in line_groups:
        line_text = "".join(s["text"] for s in group).strip()
        if not line_text:
            continue
        before_pt = 0.0
        if prev_y1 is not None:
            before_pt = max(0.0, min(8.0, group[0]["bbox"][1] - prev_y1))
        prev_y1 = max(s["bbox"][3] for s in group)

        line_x0 = min(s["bbox"][0] for s in group)
        line_x1 = max(s["bbox"][2] for s in group)
        jc_val = _detect_line_alignment(line_x0, line_x1, x0, x1)

        if target_lang and target_lang != "None (keep original)" and translator_fn:
            translated = sanitize_xml_text(translate_text(line_text, target_lang, translator_fn))
            sp = group[0]
            flags = sp.get("flags", 0)
            font_name = resolve_font(sp["font"], font_override, flags=flags)
            size_pt = resolve_size(sp["size"], font_override)
            sz_half = max(2, int(round(size_pt * 2)))
            rpr = (
                f'<w:rFonts w:ascii="{html.escape(font_name)}" w:hAnsi="{html.escape(font_name)}" '
                f'w:eastAsia="{html.escape(font_name)}" w:cs="{html.escape(font_name)}"/>'
                + ('<w:b/><w:bCs/>' if sp["bold"] else '') + ('<w:i/><w:iCs/>' if sp["italic"] else '')
                + f'<w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/><w:color w:val="{sp["color_hex"]}"/><w:w w:val="100"/>'
            )
            runs_xml = f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{html.escape(translated)}</w:t></w:r>'
        else:
            runs = []
            for sp in group:
                text = sanitize_xml_text(sp["text"])
                if not text:
                    continue
                flags = sp.get("flags", 0)
                font_name = resolve_font(sp["font"], font_override, flags=flags)
                size_pt = resolve_size(sp["size"], font_override)
                sz_half = max(2, int(round(size_pt * 2)))
                rpr = (
                    f'<w:rFonts w:ascii="{html.escape(font_name)}" w:hAnsi="{html.escape(font_name)}" '
                    f'w:eastAsia="{html.escape(font_name)}" w:cs="{html.escape(font_name)}"/>'
                    + ('<w:b/><w:bCs/>' if sp["bold"] else '')
                    + ('<w:i/><w:iCs/>' if sp["italic"] else '')
                    + f'<w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/><w:color w:val="{sp["color_hex"]}"/><w:w w:val="100"/>'
                )
                runs.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{html.escape(text)}</w:t></w:r>')
            runs_xml = ''.join(runs)

        p_xml.append(
            f'<w:p><w:pPr><w:jc w:val="{jc_val}"/><w:spacing w:before="{int(round(before_pt * 20))}" w:after="0" '
            'w:line="200" w:lineRule="auto"/><w:widowControl w:val="0"/></w:pPr>'
            + runs_xml + '</w:p>'
        )

    if not p_xml:
        p_xml = ['<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:p>']

    cell_xml.append('<w:tc>' + f'<w:tcPr>{"".join(tc_pr)}</w:tcPr>' + ''.join(p_xml) + '</w:tc>')


def add_absolute_native_pdf_table(anchor_p, page, table, font_override=None,
                                   target_lang=None, translator_fn=None):
    """Rebuild one detected PDF table as a fixed-layout native Word table.
    Preserves exact row/cell geometry, prevents row overlap, and merges multi-line detector rows."""
    rows = table.rows
    if not rows:
        return None

    h_segments, v_segments = _extract_page_vector_segments(page)
    all_spans = _page_text_spans(page)

    num_cols = getattr(table, "col_count", None) or max((len(r.cells) for r in rows), default=1)

    def padded_cells(row):
        cells = list(row.cells)
        if len(cells) < num_cols:
            cells += [None] * (num_cols - len(cells))
        return cells[:num_cols]

    def row_text(row, col):
        cells = padded_cells(row)
        cell = cells[col] if col < len(cells) else None
        if not cell:
            return ""
        texts = _spans_for_cell(all_spans, cell)
        return " ".join(s["text"].strip() for s in texts if s["text"].strip())

    def row_texts(row):
        return [_clean_table_text(row_text(row, c)) for c in range(num_cols)]

    def row_is_full_width(row):
        cells = padded_cells(row)
        c0 = cells[0]
        rest_empty = all(c is None for c in cells[1:])
        return bool(c0 and c0[0] <= table.bbox[0] + 1 and c0[2] >= table.bbox[2] - 1 and rest_empty)

    normalized = []
    i = 0
    while i < len(rows):
        row = rows[i]
        cells = padded_cells(row)
        texts = row_texts(row)
        c0_text = texts[0]

        if i + 1 < len(rows) and row_is_full_width(row) and not c0_text:
            next_row = rows[i + 1]
            next_cells = padded_cells(next_row)
            next_text = row_texts(next_row)[0]
            if row_is_full_width(next_row) and next_text:
                bbox = (cells[0][0], cells[0][1], cells[0][2], next_cells[0][3])
                normalized.append({"cells": [bbox] + [None] * (num_cols - 1),
                                    "height": bbox[3] - bbox[1], "merged": True})
                i += 2
                continue

        if normalized and tuple(texts) == tuple(normalized[-1].get("_texts", ())):
            i += 1
            continue

        if not any(texts):
            i += 1
            continue

        item = {"cells": cells, "height": row.bbox[3] - row.bbox[1], "merged": False,
                "_texts": tuple(texts)}

        while i + 1 < len(rows):
            nr = rows[i + 1]
            next_cells = padded_cells(nr)
            next_texts = row_texts(nr)

            if not any(next_texts):
                i += 1
                continue

            nonempty_next = [c for c, t in enumerate(next_texts) if t]
            can_merge = False
            merge_col = None

            cur_bot_y = max(c[3] for c in item["cells"] if c is not None)
            nr_top_y = min(c[1] for c in next_cells if c is not None)
            has_h_border = _has_horizontal_border_between(
                h_segments, table.bbox[0], table.bbox[2], cur_bot_y, nr_top_y, tol=2.5
            )

            if not has_h_border:
                if len(nonempty_next) == 1:
                    col = nonempty_next[0]
                    cur_c = item["cells"][col]
                    nxt_c = next_cells[col]
                    if cur_c and nxt_c:
                        cx0, cy0, cx1, cy1 = cur_c
                        nx0, ny0, nx1, ny1 = nxt_c
                        x_overlap = max(0.0, min(cx1, nx1) - max(cx0, nx0))
                        max_w = max(cx1 - cx0, nx1 - nx0, 1.0)
                        x_ratio = x_overlap / max_w
                        y_gap = ny0 - cy1
                        cur_h = max(4.0, cy1 - cy0)

                        same_col = (x_ratio >= 0.70)
                        y_contiguous = (y_gap <= max(14.0, cur_h * 1.6) and ny1 > cy0)

                        print(f"ROW MERGE CHECK:")
                        print(f"  current bbox = {cur_c}")
                        print(f"  next bbox    = {nxt_c}")
                        print(f"  same column = {same_col} ({x_ratio*100:.1f}%)")
                        print(f"  y gap = {y_gap:.1f} pt")
                        print(f"  horizontal border between = {has_h_border}")

                        if same_col and y_contiguous:
                            can_merge = True
                            merge_col = col
                            print(f"  decision = MERGE")
                        else:
                            print(f"  decision = DO NOT MERGE")

            if can_merge and merge_col is not None:
                col = merge_col
                cur_c = item["cells"][col]
                nxt_c = next_cells[col]
                merged_cells = list(item["cells"])
                merged_cells[col] = (
                    min(cur_c[0], nxt_c[0]),
                    min(cur_c[1], nxt_c[1]),
                    max(cur_c[2], nxt_c[2]),
                    max(cur_c[3], nxt_c[3])
                )
                item["cells"] = merged_cells
                item["height"] = max(c[3] for c in merged_cells if c) - min(c[1] for c in merged_cells if c)
                texts[col] = f"{texts[col]} {next_texts[col]}".strip()
                item["_texts"] = tuple(texts)
                i += 1
            else:
                break

        normalized.append(item)
        i += 1

    for item in normalized:
        item.pop("_texts", None)

    x0, y0, x1, y1 = table.bbox
    page_width = page.rect.width
    safe_width = min(max(10.0, x1 - x0), page_width - x0 - 1.0)

    col_bounds = None
    for row in rows:
        cells = padded_cells(row)
        if all(c is not None for c in cells):
            col_bounds = [(c[0], c[2]) for c in cells]
            break
    if col_bounds is None:
        col_w = safe_width / num_cols
        col_bounds = [(x0 + j * col_w, x0 + (j + 1) * col_w) for j in range(num_cols)]

    col_dxas = [max(1, int(round((cx1 - cx0) * 20))) for cx0, cx1 in col_bounds]
    width_dxa = sum(col_dxas)

    grid_cols_xml = ''.join(f'<w:gridCol w:w="{w}"/>' for w in col_dxas)
    tbl_xml = [
        '<w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">',
        '<w:tblPr>',
        f'<w:tblpPr w:leftFromText="0" w:rightFromText="0" w:topFromText="0" w:bottomFromText="0" '
        f'w:vertAnchor="page" w:horzAnchor="page" w:tblpX="{int(round(x0 * 20))}" w:tblpY="{int(round(y0 * 20))}"/>',
        f'<w:tblW w:w="{width_dxa}" w:type="dxa"/>',
        '<w:tblLayout w:type="fixed"/>',
        '<w:tblOverlap w:val="overlap"/>',
        '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>',
        '</w:tblPr>',
        f'<w:tblGrid>{grid_cols_xml}</w:tblGrid>'
    ]

    for item in normalized:
        cells = item["cells"]
        row_top = min((c[1] for c in cells if c), default=y0)
        row_pdf_h = max(4.0, item["height"])
        row_bottom = row_top + row_pdf_h

        cell_content_heights = []
        for cell_bbox in cells:
            if not cell_bbox:
                continue
            spans = _spans_for_cell(all_spans, cell_bbox)
            if not spans:
                continue
            lg = []
            for sp in spans:
                if not lg or abs(sp["bbox"][1] - lg[-1][0]["bbox"][1]) > 2.0:
                    lg.append([sp])
                else:
                    lg[-1].append(sp)
            c_h = 0.0
            prev_y = None
            for g in lg:
                b_pt = max(0.0, min(8.0, g[0]["bbox"][1] - prev_y)) if prev_y is not None else 0.0
                max_sz = max(s.get("size", 10.0) for s in g)
                line_h = max(8.0, max_sz * 1.25)
                c_h += line_h + b_pt
                prev_y = max(s["bbox"][3] for s in g)
            cell_content_heights.append(c_h)

        required_content_h = max(cell_content_heights, default=0.0)
        final_row_h = max(row_pdf_h, required_content_h, 6.0)

        row_xml = [f'<w:tr><w:trPr><w:cantSplit/><w:trHeight w:val="{int(round(final_row_h * 20))}" w:hRule="atLeast"/></w:trPr>']

        if item["merged"]:
            bbox = cells[0]
            fill = _cell_fill_from_pdf(page, bbox)
            spans = _spans_for_cell(all_spans, bbox)
            borders = _detect_cell_borders(h_segments, v_segments, bbox)
            _add_table_cell_xml(row_xml, bbox, spans, borders=borders, font_override=font_override,
                                target_lang=target_lang, translator_fn=translator_fn,
                                grid_span=num_cols, fill_hex=fill)
        else:
            c_idx = 0
            while c_idx < num_cols:
                cell_bbox = cells[c_idx] if c_idx < len(cells) else None
                if cell_bbox is None:
                    cx0, cx1 = col_bounds[c_idx]
                    cell_bbox = (cx0, row_top, cx1, row_bottom)
                    fill = "FFFFFF"
                    borders = _detect_cell_borders(h_segments, v_segments, cell_bbox)
                    _add_table_cell_xml(row_xml, cell_bbox, [], borders=borders, font_override=font_override,
                                        target_lang=target_lang, translator_fn=translator_fn,
                                        grid_span=None, fill_hex=fill)
                    c_idx += 1
                    continue

                span = 1
                while (c_idx + span < num_cols and cells[c_idx + span] is None
                       and col_bounds[c_idx + span][1] <= cell_bbox[2] + 2.0):
                    span += 1

                fill = _cell_fill_from_pdf(page, cell_bbox)
                spans = _spans_for_cell(all_spans, cell_bbox)
                borders = _detect_cell_borders(h_segments, v_segments, cell_bbox)
                _add_table_cell_xml(row_xml, cell_bbox, spans, borders=borders, font_override=font_override,
                                    target_lang=target_lang, translator_fn=translator_fn,
                                    grid_span=(span if span > 1 else None), fill_hex=fill)
                c_idx += span
        row_xml.append('</w:tr>')
        tbl_xml.append(''.join(row_xml))
    tbl_xml.append('</w:tbl>')

    tbl_elem = parse_xml(''.join(tbl_xml))
    target = anchor_p._p if hasattr(anchor_p, "_p") else anchor_p
    target.addnext(tbl_elem)
    return tbl_elem



def are_same_visual_line(line_a, line_b, log_diag=False):
    """Determines whether two text fragments share the same vertical baseline/line position.
    Uses geometry-aware vertical center distance and overlap relative to font/line height."""
    ay0, ay1 = line_a["bbox"][1], line_a["bbox"][3]
    by0, by1 = line_b["bbox"][1], line_b["bbox"][3]
    acy = (ay0 + ay1) / 2.0
    bcy = (by0 + by1) / 2.0
    ah = max(4.0, ay1 - ay0)
    bh = max(4.0, by1 - by0)
    min_h = min(ah, bh)
    y_diff = abs(acy - bcy)
    v_overlap = min(ay1, by1) - max(ay0, by0)
    y_tol = min_h * 0.45

    is_same = (y_diff <= y_tol) or (v_overlap >= min_h * 0.5)

    if log_diag:
        txt_a = " ".join(s["text"] for s in line_a["spans"]).strip()
        txt_b = " ".join(s["text"] for s in line_b["spans"]).strip()
        print(f"  PREV: \"{txt_a}\" | bbox: ({line_a['bbox'][0]:.1f}, {ay0:.1f}, {line_a['bbox'][2]:.1f}, {ay1:.1f})")
        print(f"  CURR: \"{txt_b}\" | bbox: ({line_b['bbox'][0]:.1f}, {by0:.1f}, {line_b['bbox'][2]:.1f}, {by1:.1f})")
        print(f"  Y diff: {y_diff:.2f} pt | Y tol: {y_tol:.2f} pt | Same visual line: {is_same}")

    return is_same


def merge_fragments_into_visual_line(line_fragments):
    """Merges sorted same-baseline fragments into a single unified visual line object."""
    if len(line_fragments) == 1:
        return line_fragments[0]

    # Sort left to right by x0
    frags_sorted = sorted(line_fragments, key=lambda l: l["bbox"][0])
    
    all_spans = []
    prev_frag = None

    for frag in frags_sorted:
        frag_spans = [dict(s) for s in frag["spans"]]
        if prev_frag is not None and frag_spans:
            # Check horizontal gap between previous fragment and this fragment
            prev_x1 = prev_frag["bbox"][2]
            curr_x0 = frag["bbox"][0]
            gap_x = curr_x0 - prev_x1
            if gap_x > 3.0:
                first_sp = frag_spans[0]
                prev_sp = all_spans[-1] if all_spans else None
                if prev_sp and not prev_sp["text"].endswith(" ") and not first_sp["text"].startswith(" "):
                    font_sz = first_sp.get("size", 11)
                    n_spaces = max(1, min(12, int(round(gap_x / max(2.0, font_sz * 0.28)))))
                    prev_sp["text"] += " " * n_spaces

        all_spans.extend(frag_spans)
        prev_frag = frag

    min_x0 = min(f["bbox"][0] for f in frags_sorted)
    min_y0 = min(f["bbox"][1] for f in frags_sorted)
    max_x1 = max(f["bbox"][2] for f in frags_sorted)
    max_y1 = max(f["bbox"][3] for f in frags_sorted)

    return {
        "bbox": (min_x0, min_y0, max_x1, max_y1),
        "spans": all_spans
    }


def partition_into_columns_if_present(lines, gutter_min_pt=15.0):
    """If lines form distinct side-by-side reading columns, partition them into
    column groups to protect multi-column reading order and prevent cross-column merging."""
    if len(lines) < 6:
        return [lines]

    x0_all = min(l["bbox"][0] for l in lines)
    x1_all = max(l["bbox"][2] for l in lines)
    total_w = x1_all - x0_all
    if total_w < 100.0:
        return [lines]

    mid_start = x0_all + total_w * 0.25
    mid_end = x0_all + total_w * 0.75

    best_gutter = None
    best_score = 0

    for x_split in range(int(mid_start), int(mid_end), 5):
        left_count = 0
        right_count = 0
        span_count = 0
        left_max_x = x0_all
        right_min_x = x1_all

        left_lines = []
        right_lines = []
        for l in lines:
            lx0, _, lx1, _ = l["bbox"]
            if lx1 <= x_split:
                left_count += 1
                left_lines.append(l)
                if lx1 > left_max_x:
                    left_max_x = lx1
            elif lx0 >= x_split:
                right_count += 1
                right_lines.append(l)
                if lx0 < right_min_x:
                    right_min_x = lx0
            else:
                span_count += 1

        if left_count >= 3 and right_count >= 3 and span_count <= len(lines) * 0.25:
            actual_gutter = right_min_x - left_max_x
            if actual_gutter >= gutter_min_pt:
                # Check if lines across the candidate gutter share horizontal baselines.
                # If multiple lines are horizontally aligned, they are rows of a table / key-value grid,
                # NOT independent sequential reading columns.
                matching_horizontal_pairs = 0
                for ll in left_lines:
                    for rl in right_lines:
                        if are_same_visual_line(ll, rl):
                            matching_horizontal_pairs += 1
                            break

                if matching_horizontal_pairs >= 2:
                    continue

                score = (left_count + right_count) * 10 - span_count * 20 + actual_gutter
                if score > best_score:
                    best_score = score
                    best_gutter = (left_max_x, right_min_x)

    if best_gutter is None:
        return [lines]

    g_left, g_right = best_gutter
    left_lines = []
    right_lines = []
    span_top_lines = []
    span_bot_lines = []

    two_col_y0 = min(l["bbox"][1] for l in lines if l["bbox"][2] <= g_left or l["bbox"][0] >= g_right)

    for l in lines:
        lx0, ly0, lx1, ly1 = l["bbox"]
        if lx1 <= g_left + 2.0:
            left_lines.append(l)
        elif lx0 >= g_right - 2.0:
            right_lines.append(l)
        elif ly1 <= two_col_y0 + 2.0:
            span_top_lines.append(l)
        else:
            span_bot_lines.append(l)

    groups = []
    if span_top_lines:
        groups.append(span_top_lines)
    if left_lines:
        groups.append(left_lines)
    if right_lines:
        groups.append(right_lines)
    if span_bot_lines:
        groups.append(span_bot_lines)

    return groups if groups else [lines]


def _lines_share_row_baseline(line_a, line_b, tol_pt=3.0):
    """Strict test for 'these two lines sit on the same table/grid row':
    close vertical CENTER distance only. Deliberately stricter than
    are_same_visual_line (which also treats large vertical overlap as a
    match, since it exists to merge same-line text fragments) -- reused
    here for cross-column row detection, that overlap-based rule flags
    almost any two adjacent lines from two densely-packed columns as
    'aligned', even when they are unrelated paragraph lines. A real
    table/key-value row has near-identical baselines; incidental overlap
    between two different prose lines does not."""
    acy = (line_a["bbox"][1] + line_a["bbox"][3]) / 2.0  
    bcy = (line_b["bbox"][1] + line_b["bbox"][3]) / 2.0
    return abs(acy - bcy) <= tol_pt


def find_column_split(lines, x0_all, x1_all, gutter_min_pt=15.0):
    """Searches actual PDF line geometry for a single vertical gutter that
    cleanly separates `lines` into an independent left group and right group
    (>=3 lines each, minimal cross-gutter spanning lines, no shared
    horizontal baselines -- which would mean a table/key-value row rather
    than two reading columns). Returns (left_lines, right_lines, left_edge_x,
    right_edge_x) for the best-scoring split, or None if no reliable split
    exists. This never assumes a fixed 50/50 split -- the boundary is
    whatever the text bounding boxes actually show."""
    total_w = x1_all - x0_all
    if len(lines) < 6 or total_w < 100.0:
        return None

    mid_start = x0_all + total_w * 0.2
    mid_end = x0_all + total_w * 0.8
    best = None
    best_score = 0

    for x_split in range(int(mid_start), int(mid_end), 5):
        left, right, span = [], [], 0
        left_max_x = x0_all
        right_min_x = x1_all
        for l in lines:
            lx0, _, lx1, _ = l["bbox"]
            if lx1 <= x_split:
                left.append(l)
                if lx1 > left_max_x:
                    left_max_x = lx1
            elif lx0 >= x_split:
                right.append(l)
                if lx0 < right_min_x:
                    right_min_x = lx0
            else:
                span += 1

        if len(left) >= 3 and len(right) >= 3 and span <= len(lines) * 0.25:
            gutter = right_min_x - left_max_x
            if gutter >= gutter_min_pt:
                # Lines that share a horizontal baseline across the candidate
                # gutter are a table/key-value row, not two reading columns --
                # but two independent prose columns will often have a FEW
                # coincidental baseline matches just from similar line pitch,
                # so only disqualify the split when a MAJORITY of lines on
                # the smaller side align (a real grid/table aligns most/all
                # rows, not just a couple by chance).
                pairs = 0
                for ll in left:
                    for rl in right:
                        if _lines_share_row_baseline(ll, rl):
                            pairs += 1
                            break
                min_side = min(len(left), len(right))
                if pairs >= max(3, int(round(min_side * 0.5))):
                    continue

                score = (len(left) + len(right)) * 10 - span * 20 + gutter
                if score > best_score:
                    best_score = score
                    best = (left, right, left_max_x, right_min_x)

    return best



def detect_full_page_columns_for_reconstruction(lines, page_width_pt, page_height_pt,
                                                gutter_min_pt=8.0):
    """Detect continuous side-by-side reading columns for Word reconstruction.

    The old implementation split the page into many horizontal bands first.
    Every band became a new one-row Word table, so a tall left cell pushed the
    following right-column block underneath it.  This function instead finds
    one stable vertical gutter for the page and creates one continuous pair of
    regions.  That is the correct structure for journal/article pages such as
    the supplied Dolder PDF.
    """
    if not lines or len(lines) < 8:
        return None

    # Running headers/footers should not influence the gutter calculation.
    body = []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        if y1 <= 50.0 or y0 >= page_height_pt - 35.0:
            continue
        body.append(ln)
    if len(body) < 8:
        return None

    x0_all = min(l["bbox"][0] for l in body)
    x1_all = max(l["bbox"][2] for l in body)
    if x1_all - x0_all < 150:
        return None

    # Use the largest stable empty x-gap rather than the previous baseline
    # heuristic.  In two-column prose, lines on opposite sides naturally have
    # similar baselines, so baseline matching incorrectly rejected real
    # columns on pages 2, 3, 6, 9, 10 and 11 of the sample.
    best = None
    for i in range(20, 81):
        split_x = x0_all + (x1_all - x0_all) * i / 100.0
        left = [l for l in body if l["bbox"][2] <= split_x]
        right = [l for l in body if l["bbox"][0] >= split_x]
        spanning = len(body) - len(left) - len(right)
        if len(left) < 8 or len(right) < 8:
            continue
        if spanning > max(10, int(len(body) * 0.12)):
            continue

        left_edge = max(l["bbox"][2] for l in left)
        right_edge = min(l["bbox"][0] for l in right)
        gutter = right_edge - left_edge
        if gutter < gutter_min_pt:
            continue

        # Reject a sidebar accidentally paired with a huge figure/legend area.
        balance = min(len(left), len(right)) / max(len(left), len(right))
        if balance < 0.25:
            continue

        # Prefer a large physical gutter and reasonably balanced content.
        score = gutter * 10.0 + balance * 20.0 - spanning * 3.0
        if best is None or score > best[0]:
            best = (score, split_x, left, right, spanning, left_edge, right_edge)

    if best is None:
        return None

    _, split_x, left, right, spanning_count, left_edge, right_edge = best
    spanning = []
    for ln in body:
        lx0, _, lx1, _ = ln["bbox"]
        if lx0 < split_x < lx1:
            spanning.append(ln)

    left.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    right.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    spanning.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))

    return {
        "split_x": split_x,
        "gutter": right_edge - left_edge,
        "left": left,
        "right": right,
        "spanning": spanning,
        "left_bbox": [min(l["bbox"][0] for l in left), min(l["bbox"][1] for l in left),
                      max(l["bbox"][2] for l in left), max(l["bbox"][3] for l in left)],
        "right_bbox": [min(l["bbox"][0] for l in right), min(l["bbox"][1] for l in right),
                       max(l["bbox"][2] for l in right), max(l["bbox"][3] for l in right)],
    }

def detect_page_regions(lines, page_width_pt, page_height_pt, gutter_min_pt=15.0, max_columns=3):
    """Dynamically detects whether a page is genuinely multi-column (1, 2, or
    3 columns) purely from PDF text bounding-box geometry -- never from text
    length and never a hard-coded 50/50 split.

    Returns (span_top_lines, column_regions, span_bot_lines):
      - column_regions: left-to-right ordered list of
            {"bbox": [x0, y0, x1, y1], "lines": [...]}
        one per detected reading column, each with its own real geometric
        boundaries (so left/right column widths need not be equal).
      - span_top_lines / span_bot_lines: content that spans the full width
        above/below the column block (e.g. a page-wide title or a footer
        strip that isn't part of either column).

    Returns ([], [], []) when no reliable multi-column split is found --
    callers should treat that as "this is a single-column page" and keep
    using the existing single-column reconstruction path unchanged.
    """
    if not lines:
        return [], [], []

    x0_all = min(l["bbox"][0] for l in lines)
    x1_all = max(l["bbox"][2] for l in lines)

    split = find_column_split(lines, x0_all, x1_all, gutter_min_pt)
    if split is None:
        return [], [], []

    left_lines, right_lines, g_left, g_right = split
    col_ids = {id(l) for l in left_lines} | {id(l) for l in right_lines}

    two_col_y0 = min(
        (l["bbox"][1] for l in lines if id(l) in col_ids), default=None
    )
    span_top, span_bot = [], []
    for l in lines:
        if id(l) in col_ids:
            continue
        ly1 = l["bbox"][3]
        if two_col_y0 is not None and ly1 <= two_col_y0 + 2.0:
            span_top.append(l)
        else:
            span_bot.append(l)

    columns = [left_lines, right_lines]

    # Optionally try one more split (3-column support), capped at max_columns.
    if max_columns >= 3:
        for idx in (0, 1):
            col_lines = columns[idx]
            if len(col_lines) < 6:
                continue
            cx0 = min(l["bbox"][0] for l in col_lines)
            cx1 = max(l["bbox"][2] for l in col_lines)
            sub = find_column_split(col_lines, cx0, cx1, gutter_min_pt)
            if sub:
                sub_left, sub_right, _, _ = sub
                columns[idx:idx + 1] = [sub_left, sub_right]
                break

    regions = []
    for col_lines in columns:
        if not col_lines:
            continue
        cx0 = min(l["bbox"][0] for l in col_lines)
        cy0 = min(l["bbox"][1] for l in col_lines)
        cx1 = max(l["bbox"][2] for l in col_lines)
        cy1 = max(l["bbox"][3] for l in col_lines)
        regions.append({"bbox": [cx0, cy0, cx1, cy1], "lines": col_lines})

    regions.sort(key=lambda r: r["bbox"][0])
    return span_top, regions, span_bot


def _lines_bbox(lines):
    x0 = min(l["bbox"][0] for l in lines)
    y0 = min(l["bbox"][1] for l in lines)
    x1 = max(l["bbox"][2] for l in lines)
    y1 = max(l["bbox"][3] for l in lines)
    return [x0, y0, x1, y1]


def find_all_column_splits(lines, gutter_min_pt=15.0, max_columns=6):
    """Recursively widens a single find_column_split() call into an
    arbitrary number of side-by-side reading columns (not capped at a fixed
    2 or 3). Repeatedly tries to split every current group again until no
    group can be split further or max_columns is reached -- max_columns is
    a safety ceiling against pathological over-splitting, never an assumed
    layout. Returns left-to-right ordered line groups (1 group = single
    column / no split found)."""
    groups = [lines]
    while len(groups) < max_columns:
        did_split = False
        next_groups = []
        for g in groups:
            if len(g) < 6:
                next_groups.append(g)
                continue
            gx0 = min(l["bbox"][0] for l in g)
            gx1 = max(l["bbox"][2] for l in g)
            split = find_column_split(g, gx0, gx1, gutter_min_pt)
            if split is None:
                next_groups.append(g)
                continue
            left, right, _, _ = split
            # Only keep the split if it doesn't drop lines (span lines
            # between the two groups are folded into whichever side is
            # closer, so no content is lost when column count grows).
            covered = {id(l) for l in left} | {id(l) for l in right}
            missing = [l for l in g if id(l) not in covered]
            if missing:
                # attach spanning leftovers to the nearer column by x-center
                for l in missing:
                    cx = (l["bbox"][0] + l["bbox"][2]) / 2.0
                    l_cx = sum((x["bbox"][0] + x["bbox"][2]) / 2.0 for x in left) / len(left)
                    r_cx = sum((x["bbox"][0] + x["bbox"][2]) / 2.0 for x in right) / len(right)
                    (left if abs(cx - l_cx) <= abs(cx - r_cx) else right).append(l)
            next_groups.append(left)
            next_groups.append(right)
            did_split = True
        groups = next_groups
        if not did_split:
            break
    groups = [g for g in groups if g]
    groups.sort(key=lambda g: min(l["bbox"][0] for l in g))
    return groups


def _segment_into_row_bands(lines, gap_multiplier=2.2):
    """Stage 1 of recursive region detection (spec section 2: "PAGE ->
    VERTICAL SEGMENTATION -> REGION DETECTION -> LAYOUT DETECTION FOR EACH
    REGION"). Groups lines into ordered top-to-bottom horizontal bands by
    looking for vertical whitespace gaps meaningfully larger than the
    surrounding line pitch -- e.g. the gap between a title and the body
    text below it, or between a 2-column block and a full-width table
    beneath it. A single global column-split test across the WHOLE page
    is unreliable when different bands have different column counts (a
    3-column block's middle column shows up as "spanning" lines to any
    left/right test run against the entire page); segmenting into bands
    first, then detecting columns independently inside each band, avoids
    that failure mode.
    """
    if not lines:
        return []
    lines_sorted = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    heights = [max(1.0, l["bbox"][3] - l["bbox"][1]) for l in lines_sorted]
    heights.sort()
    median_h = heights[len(heights) // 2]

    bands = [[lines_sorted[0]]]
    band_bottom = lines_sorted[0]["bbox"][3]
    for ln in lines_sorted[1:]:
        gap = ln["bbox"][1] - band_bottom
        if gap > median_h * gap_multiplier:
            bands.append([ln])
        else:
            bands[-1].append(ln)
        band_bottom = max(band_bottom, ln["bbox"][3])
    return bands


def detect_page_regions_recursive(lines, page_width_pt, page_height_pt, gutter_min_pt=15.0,
                                   max_columns=6, row_gap_multiplier=2.2):
    """Generalizes detect_page_regions() from a single page-level 2/3-column
    guess into a full VERTICAL SEGMENTATION + PER-REGION column detection
    pass (spec sections 2 and 4): a page can stack any number of bands
    top-to-bottom (full-width title -> 2-column body -> 3-column table ->
    full-width footer, etc.), and each band's column count is detected
    independently from that band's own geometry only -- never assumed to
    match its neighbors, and never capped at a hard-coded 2 or 3.

    Returns an ordered (top-to-bottom) list of bands:
        {"bbox": [x0,y0,x1,y1],
         "column_count": int,
         "columns": [{"bbox": [...], "lines": [...]}, ...]}
    A single-column band still has exactly one entry in "columns", so
    callers can treat every band uniformly.
    """
    if not lines:
        return []

    row_bands = _segment_into_row_bands(lines, row_gap_multiplier)

    bands = []
    for band_lines in row_bands:
        col_groups = find_all_column_splits(band_lines, gutter_min_pt, max_columns)
        columns = [{"bbox": _lines_bbox(g), "lines": g} for g in col_groups if g]
        if not columns:
            continue
        band_bbox = [
            min(c["bbox"][0] for c in columns), min(c["bbox"][1] for c in columns),
            max(c["bbox"][2] for c in columns), max(c["bbox"][3] for c in columns),
        ]
        bands.append({"bbox": band_bbox, "column_count": len(columns), "columns": columns})

    return bands


def set_keep_with_next(paragraph):
    """Marks a paragraph 'keep with next' (spec 35) -- e.g. a heading
    should never be stranded alone at the bottom of a page, separated from
    the text that follows it."""
    p_pr = paragraph._p.get_or_add_pPr()
    if p_pr.find(qn("w:keepNext")) is None:
        p_pr.append(OxmlElement("w:keepNext"))


def set_keep_lines_together(paragraph):
    """Marks a paragraph 'keep lines together' (spec 35) so Word won't
    split it across a page boundary mid-paragraph."""
    p_pr = paragraph._p.get_or_add_pPr()
    if p_pr.find(qn("w:keepLines")) is None:
        p_pr.append(OxmlElement("w:keepLines"))


def reconstruct_visual_lines_in_group(lines):
    """Stage 1: Reconstructs same-baseline text fragments into unified visual lines within a reading group."""
    if not lines:
        return []

    lines_sorted = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    
    visual_lines = []
    current_line_frags = [lines_sorted[0]]

    for curr in lines_sorted[1:]:
        base_line = current_line_frags[0]
        if are_same_visual_line(base_line, curr):
            current_line_frags.append(curr)
        else:
            visual_lines.append(merge_fragments_into_visual_line(current_line_frags))
            current_line_frags = [curr]

    if current_line_frags:
        visual_lines.append(merge_fragments_into_visual_line(current_line_frags))

    return visual_lines


def group_lines_into_paragraph_blocks(lines, gap_multiplier=1.4):
    """Groups multirun lines into unified paragraph blocks.
    
    STAGE 1: Detects column partitions if present and unifies same-baseline text fragments
    (e.g., 'Goal Met  □ Yes  ☑ No') into single visual lines based on vertical geometry.
    
    STAGE 2: Groups sequential visual lines into paragraph blocks using line pitch,
    list boundaries, and paragraph spacing."""
    if not lines:
        return []

    # Stage 1: Partition into reading columns if present, then reconstruct visual lines
    col_groups = partition_into_columns_if_present(lines)
    all_visual_lines = []
    for col_lines in col_groups:
        all_visual_lines.extend(reconstruct_visual_lines_in_group(col_lines))

    if not all_visual_lines:
        return []

    # Stage 2: Group consecutive visual lines into paragraphs
    blocks = [[all_visual_lines[0]]]

    for prev, curr in zip(all_visual_lines, all_visual_lines[1:]):
        prev_x0, prev_y0, prev_x1, prev_y1 = prev["bbox"]
        curr_x0, curr_y0, curr_x1, curr_y1 = curr["bbox"]

        prev_height = max(6.0, prev_y1 - prev_y0)
        gap = curr_y0 - prev_y1

        prev_text = " ".join(s["text"] for s in prev["spans"])
        curr_text = " ".join(s["text"] for s in curr["spans"])

        is_list_boundary = (detect_list_marker(curr_text)[0] is not None or
                            detect_list_marker(prev_text)[0] is not None)

        horizontal_aligned = not (curr_x0 > prev_x1 + 40 or curr_x1 < prev_x0 - 40)

        # Key-value / row-aligned boundary: when curr line starts back at the left margin
        is_row_start = (
            curr_x0 <= prev_x0 + 4.0 and
            (prev_x1 > curr_x0 + 100.0 or abs(curr_x0 - prev_x0) <= 2.0) and
            len(curr.get("spans", [])) >= 1 and
            curr_y0 > prev_y0 + 6.0
        )

        if gap > prev_height * gap_multiplier or is_list_boundary or not horizontal_aligned or gap < -2.0 or is_row_start:
            blocks.append([curr])
        else:
            blocks[-1].append(curr)

    return blocks


def _bbox_center_inside(inner_bbox, outer_bbox, y_slack_pt=1.0):
    """True if inner_bbox's center point falls inside outer_bbox (with a
    small vertical slack so lines that just touch a hint box's top/bottom
    edge still count as inside it)."""
    ix0, iy0, ix1, iy1 = inner_bbox
    ox0, oy0, ox1, oy1 = outer_bbox
    cx = (ix0 + ix1) / 2.0
    cy = (iy0 + iy1) / 2.0
    return (ox0 - 2.0) <= cx <= (ox1 + 2.0) and (oy0 - y_slack_pt) <= cy <= (oy1 + y_slack_pt)


def partition_lines_by_hint_boxes(lines, hint_boxes):
    """
    Splits `lines` into:
      - boxed_groups: a list of (hint_box, [lines...]) pairs, one entry per
        hint box that actually contains lines, in top-to-bottom reading order.
      - leftover: lines that don't fall inside any hint box, to go through
        the normal gap-based auto-grouping unchanged.

    A hint box is a blue annotation rectangle from the source PDF (see
    extract_drawings) that marks "these lines must be reconstructed as ONE
    single table, sized to this box's width and left indentation" -- it is
    never rendered itself.
    """
    if not hint_boxes:
        return [], list(lines)

    boxes_sorted = sorted(hint_boxes, key=lambda h: (h["bbox"][1], h["bbox"][0]))
    grouped = {id(h): [] for h in boxes_sorted}
    leftover = []

    for ln in lines:
        placed = False
        for h in boxes_sorted:
            if _bbox_center_inside(ln["bbox"], h["bbox"]):
                grouped[id(h)].append(ln)
                placed = True
                break
        if not placed:
            leftover.append(ln)

    boxed_groups = [
        (h, sorted(grouped[id(h)], key=lambda l: (l["bbox"][1], l["bbox"][0])))
        for h in boxes_sorted
        if grouped[id(h)]
    ]
    return boxed_groups, leftover


def add_absolute_page_paragraph_table(anchor_p, ref_elem, para_blocks_with_meta, page_width_pt=612.0,
                                      page_height_pt=792.0, hyperlinks=None, font_override=None):
    """Creates a SINGLE absolutely-positioned borderless Word table for the entire page's
    normal text, where each paragraph block is rendered as exactly ONE table row (<w:tr>).

    Page
    └── ONE Word table (<w:tbl>)
         ├── Row 1 (<w:tr>) = Paragraph 1
         ├── Row 2 (<w:tr>) = Paragraph 2
         ├── Row 3 (<w:tr>) = Paragraph 3
         └── Row N (<w:tr>) = Paragraph N

    Preserves exact font resolution, font sizes, bold/italic formatting, hyperlinks,
    per-line indentations, inter-line/inter-paragraph spacing, and auto-fit character scaling.
    """
    if not para_blocks_with_meta:
        return ref_elem

    # Flatten all lines across blocks to compute the bounding page-level table geometry
    all_lines = []
    for entry in para_blocks_with_meta:
        lines = entry if isinstance(entry, list) else entry[0]
        all_lines.extend(lines)

    if not all_lines:
        return ref_elem

    p_x0 = min(ln["bbox"][0] for ln in all_lines)
    p_y0 = min(ln["bbox"][1] for ln in all_lines)
    content_x1 = max(ln["bbox"][2] for ln in all_lines)
    content_y1 = max(ln["bbox"][3] for ln in all_lines)

    # Measure all lines to ensure the page table width accommodates font rendering
    max_est_width = 0.0
    for ln in all_lines:
        line_est = sum(
            measure_text_width_pt(sp["text"], resolve_font(sp["font"], font_override, flags=sp.get("flags", 0)),
                                   resolve_size(sp["size"], font_override), sp["bold"])
            for sp in ln["spans"]
        )
        if line_est > max_est_width:
            max_est_width = line_est

    max_allowed_width = max(10.0, page_width_pt - p_x0 - 2.0)
    raw_width = content_x1 - p_x0
    safe_width_pt = min(max(raw_width + 2.0, max_est_width + 4.0), max_allowed_width)

    safe_top_pt = max(0.0, min(p_y0, page_height_pt - 10.0))
    left_dxa = int(round(p_x0 * 20))
    top_dxa = int(round(safe_top_pt * 20))
    width_dxa = int(round(safe_width_pt * 20))

    rows_xml = []
    prev_para_bottom_y = safe_top_pt

    for block_idx, entry in enumerate(para_blocks_with_meta):
        if isinstance(entry, tuple):
            para_lines, override_bbox = entry
        else:
            para_lines, override_bbox = entry, None

        if not para_lines:
            continue

        curr_para_y0 = min(ln["bbox"][1] for ln in para_lines)
        curr_para_y1 = max(ln["bbox"][3] for ln in para_lines)

        # Calculate exact vertical row height from PDF geometry:
        # Row 0 spans from safe_top_pt to curr_para_y1.
        # Row N spans from prev_para_bottom_y to curr_para_y1 when sequential.
        # For horizontally overlapping / parallel blocks in Y, do not double-count the height.
        if block_idx == 0:
            row_gap_pt = max(0.0, curr_para_y0 - safe_top_pt)
            row_height_pt = max(curr_para_y1 - safe_top_pt, 4.0)
        else:
            if curr_para_y0 >= prev_para_bottom_y:
                row_gap_pt = curr_para_y0 - prev_para_bottom_y
                row_height_pt = max(curr_para_y1 - prev_para_bottom_y, 4.0)
            else:
                # Parallel or overlapping block in Y: row gap is 0, height is only new vertical extent
                row_gap_pt = 0.0
                if curr_para_y1 > prev_para_bottom_y:
                    row_height_pt = max(curr_para_y1 - prev_para_bottom_y, 4.0)
                else:
                    row_height_pt = 4.0

        height_dxa = int(round(row_height_pt * 20))

        # Measure lines within this paragraph for auto-fit scaling
        line_widths = {}
        for ln_idx, ln in enumerate(para_lines):
            line_est = sum(
                measure_text_width_pt(sp["text"], resolve_font(sp["font"], font_override, flags=sp.get("flags", 0)),
                                       resolve_size(sp["size"], font_override), sp["bold"])
                for sp in ln["spans"]
            )
            line_widths[ln_idx] = line_est

        paragraphs_xml = []
        for ln_idx, ln in enumerate(para_lines):
            spans = ln["spans"]
            uri = find_hyperlink_for_bbox(ln["bbox"], hyperlinks) if hyperlinks else None

            line_left = ln["bbox"][0]
            rel_indent_pt = max(0.0, line_left - p_x0)
            indent_xml = f'<w:ind w:left="{int(round(rel_indent_pt * 20))}"/>' if rel_indent_pt > 1.5 else ""

            available_pt = max(4.0, safe_width_pt - rel_indent_pt)
            fit_pct = fit_scale_percent(line_widths.get(ln_idx, 0.0), available_pt)
            fit_xml = f'<w:w w:val="{fit_pct}"/>' if fit_pct < 100 else ""

            runs_xml = []
            for sp_idx, sp in enumerate(spans):
                text_esc = html.escape(sanitize_xml_text(sp["text"]))
                flags = sp.get("flags", 0)
                bold_xml = "<w:b/><w:bCs/>" if sp["bold"] else ""
                italic_xml = "<w:i/><w:iCs/>" if sp["italic"] else ""
                # Keep the hyperlink clickable without forcing the classic
                # blue/underlined look -- only render underline/blue when the
                # source PDF span was actually styled that way.
                underline_xml = '<w:u w:val="single"/>' if sp.get("underline") else ""
                size_pt = resolve_size(sp["size"], font_override)
                sz_half = max(2, int(round(size_pt * 2)))
                resolved_font_name = resolve_font(sp["font"], font_override, flags=flags)
                font = html.escape(sanitize_xml_text(resolved_font_name))
                color_hex = sp["color_hex"]

                runs_xml.append(
                    f'<w:r><w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>'
                    f'{bold_xml}{italic_xml}{underline_xml}'
                    f'{fit_xml}<w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/><w:color w:val="{color_hex}"/></w:rPr>'
                    f'<w:t xml:space="preserve">{text_esc}</w:t></w:r>'
                )

            run_block = "".join(runs_xml)
            r_ns = ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
            if uri:
                r_id = anchor_p.part.relate_to(uri, RT.HYPERLINK, is_external=True)
                run_block = f'<w:hyperlink{r_ns} r:id="{r_id}">{run_block}</w:hyperlink>'

            # Maintain vertical spacing between paragraphs and lines
            if ln_idx == 0:
                space_before_pt = int(round(row_gap_pt * 20))
            else:
                prev_y1 = para_lines[ln_idx - 1]["bbox"][3]
                curr_y0 = ln["bbox"][1]
                space_before_pt = max(0, int(round((curr_y0 - prev_y1) * 20)))

            paragraphs_xml.append(
                f'<w:p><w:pPr>{indent_xml}<w:spacing w:before="{space_before_pt}" w:after="0" '
                f'w:line="200" w:lineRule="auto"/><w:widowControl w:val="0"/></w:pPr>{run_block}</w:p>'
            )

        prev_para_bottom_y = max(prev_para_bottom_y, curr_para_y1)
        cell_content_xml = "".join(paragraphs_xml)

        rows_xml.append(
            '<w:tr>'
            f'<w:trPr><w:cantSplit/><w:trHeight w:val="{height_dxa}" w:hRule="exact"/></w:trPr>'
            '<w:tc>'
            f'<w:tcPr><w:tcW w:w="{width_dxa}" w:type="dxa"/>'
            '<w:shd w:val="clear" w:color="auto" w:fill="auto"/>'
            '<w:tcMar>'
            '<w:top w:w="0" w:type="dxa"/>'
            '<w:left w:w="0" w:type="dxa"/>'
            '<w:bottom w:w="0" w:type="dxa"/>'
            '<w:right w:w="0" w:type="dxa"/>'
            '</w:tcMar></w:tcPr>'
            f'{cell_content_xml}'
            '</w:tc>'
            '</w:tr>'
        )

    all_rows_xml = "".join(rows_xml)

    tbl_xml = (
        '<w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:tblPr>'
        f'<w:tblpPr w:leftFromText="0" w:rightFromText="0" w:topFromText="0" w:bottomFromText="0" '
        f'w:vertAnchor="page" w:horzAnchor="page" w:tblpX="{left_dxa}" w:tblpY="{top_dxa}"/>'
        f'<w:tblW w:w="{width_dxa}" w:type="dxa"/>'
        '<w:tblOverlap w:val="overlap"/>'
        '<w:tblBorders>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>'
        '<w:shd w:val="clear" w:color="auto" w:fill="auto"/>'
        '<w:tblCellMar>'
        '<w:top w:w="0" w:type="dxa"/>'
        '<w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/>'
        '<w:right w:w="0" w:type="dxa"/>'
        '</w:tblCellMar>'
        '</w:tblPr>'
        f'<w:tblGrid><w:gridCol w:w="{width_dxa}"/></w:tblGrid>'
        f'{all_rows_xml}'
        '</w:tbl>'
    )

    tbl_elem = parse_xml(tbl_xml)
    target = ref_elem._p if hasattr(ref_elem, "_p") else ref_elem
    target.addnext(tbl_elem)
    return tbl_elem


def _build_paragraph_xml_for_lines(lines, region_x0, region_width_pt, hyperlinks, font_override,
                                    anchor_p, top_y=None):
    """Builds a run of <w:p> XML paragraphs for lines confined to a single
    column region, using the same font/size/bold/italic/hyperlink/auto-fit
    logic as add_absolute_page_paragraph_table -- but emitting plain
    paragraphs meant to live inside one native table cell, not page-anchored
    table rows. `region_x0`/`region_width_pt` are the column's own left edge
    and width, so per-line indentation stays relative to that column only."""
    if not lines:
        return ""

    blocks = group_lines_into_paragraph_blocks(lines)
    if not blocks:
        return ""

    paragraphs_xml = []
    prev_bottom = top_y if top_y is not None else min(ln["bbox"][1] for ln in lines)

    for block in blocks:
        block_y0 = min(ln["bbox"][1] for ln in block)
        block_gap_pt = max(0.0, block_y0 - prev_bottom)

        for ln_idx, ln in enumerate(block):
            spans = ln["spans"]
            uri = find_hyperlink_for_bbox(ln["bbox"], hyperlinks) if hyperlinks else None

            line_left = ln["bbox"][0]
            rel_indent_pt = max(0.0, line_left - region_x0)
            indent_xml = f'<w:ind w:left="{int(round(rel_indent_pt * 20))}"/>' if rel_indent_pt > 1.5 else ""

            line_est = sum(
                measure_text_width_pt(sp["text"], resolve_font(sp["font"], font_override, flags=sp.get("flags", 0)),
                                       resolve_size(sp["size"], font_override), sp["bold"])
                for sp in spans
            )
            available_pt = max(4.0, region_width_pt - rel_indent_pt)
            fit_pct = fit_scale_percent(line_est, available_pt)
            fit_xml = f'<w:w w:val="{fit_pct}"/>' if fit_pct < 100 else ""

            runs_xml = []
            for sp in spans:
                text_esc = html.escape(sanitize_xml_text(sp["text"]))
                flags = sp.get("flags", 0)
                bold_xml = "<w:b/><w:bCs/>" if sp["bold"] else ""
                italic_xml = "<w:i/><w:iCs/>" if sp["italic"] else ""
                underline_xml = '<w:u w:val="single"/>' if sp.get("underline") else ""
                size_pt = resolve_size(sp["size"], font_override)
                sz_half = max(2, int(round(size_pt * 2)))
                resolved_font_name = resolve_font(sp["font"], font_override, flags=flags)
                font = html.escape(sanitize_xml_text(resolved_font_name))
                color_hex = sp["color_hex"]
                runs_xml.append(
                    f'<w:r><w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>'
                    f'{bold_xml}{italic_xml}{underline_xml}'
                    f'{fit_xml}<w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/><w:color w:val="{color_hex}"/></w:rPr>'
                    f'<w:t xml:space="preserve">{text_esc}</w:t></w:r>'
                )

            run_block = "".join(runs_xml)
            if uri:
                r_ns = ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
                r_id = anchor_p.part.relate_to(uri, RT.HYPERLINK, is_external=True)
                run_block = f'<w:hyperlink{r_ns} r:id="{r_id}">{run_block}</w:hyperlink>'

            if ln_idx == 0:
                space_before_pt = int(round(block_gap_pt * 20))
            else:
                prev_y1 = block[ln_idx - 1]["bbox"][3]
                curr_y0 = ln["bbox"][1]
                space_before_pt = max(0, int(round((curr_y0 - prev_y1) * 20)))

            paragraphs_xml.append(
                f'<w:p><w:pPr>{indent_xml}<w:spacing w:before="{space_before_pt}" w:after="0" '
                f'w:line="200" w:lineRule="auto"/><w:widowControl w:val="0"/></w:pPr>{run_block}</w:p>'
            )

        prev_bottom = max(ln["bbox"][3] for ln in block)

    return "".join(paragraphs_xml)


def add_absolute_multi_column_table(anchor_p, ref_elem, column_regions, page_width_pt=612.0,
                                     page_height_pt=792.0, hyperlinks=None, font_override=None):
    """Builds a REAL native Word table for a genuinely multi-column PDF
    region: one row, one cell per detected column (side by side), each cell
    holding only that column's own paragraphs. Left/right content can never
    flow or overlap into each other because they live in separate table
    cells, not shared indentation. A thin borderless spacer cell between
    each pair of columns preserves the actual PDF gutter width (never a
    fixed/equal-width guess). Column widths come directly from each
    region's own PDF bounding box.

    Uses <w:tblLayout w:type="fixed"/> and never relies on Word's autofit,
    per the fixed-column-width requirement for multi-column reconstruction.
    """
    if not column_regions:
        return ref_elem

    all_x0 = min(r["bbox"][0] for r in column_regions)
    all_x1 = max(r["bbox"][2] for r in column_regions)
    all_y0 = min(r["bbox"][1] for r in column_regions)

    left_dxa = int(round(all_x0 * 20))
    top_dxa = int(round(max(0.0, all_y0) * 20))
    width_dxa = int(round(max(20.0, all_x1 - all_x0) * 20))

    grid_cols_dxa = []
    cells_xml = []
    for idx, region in enumerate(column_regions):
        rx0, ry0, rx1, ry1 = region["bbox"]
        col_width_pt = max(20.0, rx1 - rx0)
        col_width_dxa = int(round(col_width_pt * 20))
        grid_cols_dxa.append(col_width_dxa)

        cell_content_xml = _build_paragraph_xml_for_lines(
            region["lines"], rx0, col_width_pt, hyperlinks, font_override, anchor_p, top_y=ry0
        )
        # If this column's content starts lower on the page than the
        # tallest column (e.g. a sidebar starting below the main title),
        # a leading spacer paragraph nudges it down -- native table cells
        # can't be independently y-positioned, so this keeps columns close
        # to their original relative vertical alignment.
        cell_top_offset_pt = max(0.0, ry0 - all_y0)
        if cell_top_offset_pt > 2.0:
            lead_dxa = max(20, int(round(cell_top_offset_pt * 20)))
            cell_content_xml = (
                f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="{lead_dxa}" '
                f'w:lineRule="exact"/></w:pPr></w:p>{cell_content_xml}'
            )
        if not cell_content_xml:
            cell_content_xml = "<w:p/>"

        cells_xml.append(
            '<w:tc>'
            f'<w:tcPr><w:tcW w:w="{col_width_dxa}" w:type="dxa"/>'
            '<w:shd w:val="clear" w:color="auto" w:fill="auto"/>'
            '<w:tcMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
            '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tcMar>'
            '<w:vAlign w:val="top"/></w:tcPr>'
            f'{cell_content_xml}'
            '</w:tc>'
        )

        if idx < len(column_regions) - 1:
            next_x0 = column_regions[idx + 1]["bbox"][0]
            gutter_pt = max(4.0, next_x0 - rx1)
            gutter_dxa = int(round(gutter_pt * 20))
            grid_cols_dxa.append(gutter_dxa)
            cells_xml.append(
                '<w:tc>'
                f'<w:tcPr><w:tcW w:w="{gutter_dxa}" w:type="dxa"/>'
                '<w:shd w:val="clear" w:color="auto" w:fill="auto"/>'
                '<w:tcMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
                '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tcMar>'
                '</w:tcPr><w:p/></w:tc>'
            )

    grid_xml = "".join(f'<w:gridCol w:w="{w}"/>' for w in grid_cols_dxa)
    row_xml = f'<w:tr><w:trPr/>{"".join(cells_xml)}</w:tr>'

    tbl_xml = (
        '<w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:tblPr>'
        f'<w:tblpPr w:leftFromText="0" w:rightFromText="0" w:topFromText="0" w:bottomFromText="0" '
        f'w:vertAnchor="page" w:horzAnchor="page" w:tblpX="{left_dxa}" w:tblpY="{top_dxa}"/>'
        f'<w:tblW w:w="{width_dxa}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        '<w:tblOverlap w:val="overlap"/>'
        '<w:tblBorders>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>'
        '<w:shd w:val="clear" w:color="auto" w:fill="auto"/>'
        '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
        '</w:tblPr>'
        f'<w:tblGrid>{grid_xml}</w:tblGrid>'
        f'{row_xml}'
        '</w:tbl>'
    )

    tbl_elem = parse_xml(tbl_xml)
    target = ref_elem._p if hasattr(ref_elem, "_p") else ref_elem
    target.addnext(tbl_elem)
    return tbl_elem


def _normalize_image_bytes(image_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        if img.mode in ("P", "LA"):
            img = img.convert("RGBA")
        elif img.mode == "1":
            img = img.convert("L")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


def add_absolute_image_bytes(paragraph, image_bytes, left_pt, top_pt, width_pt, height_pt):
    if width_pt <= 0 or height_pt <= 0:
        return
    image_bytes = _normalize_image_bytes(image_bytes)
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(image_bytes), width=Pt(width_pt), height=Pt(height_pt))
    drawing = run._r.find(qn("w:drawing"))
    if drawing is None:
        return
    blip = drawing.find(".//" + qn("a:blip"))
    r_id = blip.get(qn("r:embed")) if blip is not None else None
    run._r.remove(drawing)
    if not r_id:
        return
    vml_xml = (
        '<w:pict xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:o="urn:schemas-microsoft-com:office:office">'
        f'<v:shape style="position:absolute;left:{left_pt}pt;top:{top_pt}pt;'
        f'width:{width_pt}pt;height:{height_pt}pt;'
        'mso-position-horizontal:absolute;mso-position-horizontal-relative:page;'
        'mso-position-vertical:absolute;mso-position-vertical-relative:page" '
        f'o:allowincell="f" stroked="f" filled="f">'
        f'<v:imagedata r:id="{r_id}" o:title=""/></v:shape></w:pict>'
    )
    run._r.append(parse_xml(vml_xml))


# --------------------------------------------------------------------------- #
# Trace-format download mode
#
# "Trace" output is a normal editable DOCX where every page also carries a
# faded, washed-out image of its own original PDF page as a true background
# watermark -- reflecting the source right through the SAME page (like
# holding paper up to the light), not as a separate following page. It's
# built the same way Word's own "Picture Watermark" feature works: a
# page-relative floating image with a very negative z-index, so it always
# renders behind every other element on the page regardless of insertion
# order.
# --------------------------------------------------------------------------- #

def render_pdf_page_trace_image(page: "fitz.Page", dpi: int = 150, fade: float = 0.28,
                                 label: str = None) -> bytes:
    """Rasterizes a PDF page and blends it toward white (the same "washout"
    look Word's own picture-watermark recolor produces). Returns PNG bytes.

    fade: 0.0 = fully white/blank, 1.0 = full-strength original image.
    Lower values are a lighter/fainter watermark.
    label: optional short diagonal stamp (e.g. "DRAFT") baked faintly into
    the image on top of the washout. Omitted by default -- the washed-out
    original content alone is the watermark.
    """
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    white = Image.new("RGB", img.size, (255, 255, 255))
    washed = Image.blend(white, img, max(0.05, min(1.0, fade))).convert("RGBA")

    if label:
        try:
            overlay = Image.new("RGBA", washed.size, (0, 0, 0, 0))
            font_size = max(24, washed.size[1] // 16)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            probe = ImageDraw.Draw(overlay)
            bbox = probe.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_layer = Image.new("RGBA", (tw + 40, th + 40), (0, 0, 0, 0))
            ImageDraw.Draw(text_layer).text((20, 20), label, font=font, fill=(140, 140, 140, 120))
            text_layer = text_layer.rotate(35, expand=True)
            ox = (washed.size[0] - text_layer.size[0]) // 2
            oy = (washed.size[1] - text_layer.size[1]) // 2
            overlay.paste(text_layer, (ox, oy), text_layer)
            washed = Image.alpha_composite(washed, overlay)
        except Exception:
            pass  # the label stamp is purely cosmetic -- never fail the page over it

    buf = io.BytesIO()
    washed.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def add_page_watermark_image(paragraph, image_bytes, page_width_pt: float, page_height_pt: float):
    """Places a full-page image as a true background watermark on the
    CURRENT page -- floats behind all text/content via a page-relative,
    very-negative z-index VML shape (the same mechanism Word's own
    "Picture Watermark" feature uses), rather than appearing as a separate
    page. Works no matter where in the document this is called from."""
    if page_width_pt <= 0 or page_height_pt <= 0:
        return
    image_bytes = _normalize_image_bytes(image_bytes)
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(image_bytes), width=Pt(page_width_pt), height=Pt(page_height_pt))
    drawing = run._r.find(qn("w:drawing"))
    if drawing is None:
        return
    blip = drawing.find(".//" + qn("a:blip"))
    r_id = blip.get(qn("r:embed")) if blip is not None else None
    run._r.remove(drawing)
    if not r_id:
        return
    vml_xml = (
        '<w:pict xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word">'
        f'<v:shape style="position:absolute;left:0;top:0;'
        f'width:{page_width_pt}pt;height:{page_height_pt}pt;'
        'z-index:-251658240;'
        'mso-position-horizontal:absolute;mso-position-horizontal-relative:page;'
        'mso-position-vertical:absolute;mso-position-vertical-relative:page" '
        'o:allowincell="f" stroked="f" filled="f">'
        f'<v:imagedata r:id="{r_id}" o:title=""/>'
        '<w10:wrap type="none"/>'
        '</v:shape></w:pict>'
    )
    run._r.append(parse_xml(vml_xml))


def _normalize_hf_text(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())



def detect_rotated_pdf_table(page, lines, drawings):
    """Detect a tall table whose text is rotated 90 degrees in PDF space.

    Uses PDF geometry only: long vertical rules + a dense set of vertical
    text spans. No page number, filename or fixed coordinates are used.
    """
    try:
        # If caller passed the filtered drawing list, fall back to raw page
        # drawings because thin table rules may intentionally be filtered from
        # normal visual drawing output.
        drawing_source = drawings or page.get_drawings()
        vertical_rules=[]
        for d in drawing_source:
            r=d.get("rect")
            if not r: continue
            x0,y0,x1,y1=map(float,r)
            if abs(x1-x0) <= 1.5 and (y1-y0) > page.rect.height*0.55:
                vertical_rules.append((x0,y0,x1,y1))
        if len(vertical_rules) < 2:
            return None

        vlines=sorted(set(round(r[0],1) for r in vertical_rules))
        # Candidate area is between the leftmost/rightmost meaningful rules.
        x0=min(vlines); x1=max(vlines)
        if x1-x0 < 30 or x1-x0 > page.rect.width*0.35:
            return None

        vlines_data=[]
        spans=[]
        for ln in lines or []:
            b=ln.get("bbox")
            if not b or b[2] < x0-2 or b[0] > x1+2:
                continue
            # Once the narrow vertical-table corridor has been identified,
            # retain every span inside it. Single-letter labels such as N
            # are too short to pass a height/width vertical-text heuristic.
            for sp in ln.get("spans",[]):
                sb=sp.get("bbox",b)
                if sb[0] >= x0-4 and sb[2] <= x1+4:
                    spans.append(sp)
        if len(spans) < 15:
            return None

        # Cluster x-centres into physical/logical rows.
        # Use the long PDF rules as broad row-band separators, then cluster
        # wrapped text pieces inside each band. This handles cells such as
        # "23 geometric" + "mean (95% CI)" without merging neighboring rows.
        xs=sorted([(float((s["bbox"][0]+s["bbox"][2])/2),s) for s in spans], key=lambda z:z[0])
        major=sorted(set(vlines))
        broad=[]
        edges=[major[0]]+major[1:]+[major[-1]] if len(major)>=2 else [x0,x1]
        # The interval list between consecutive long rules.
        for bi in range(len(major)-1):
            lo,hi=major[bi],major[bi+1]
            members=[(xc,sp) for xc,sp in xs if lo-1.5<=xc<=hi+1.5]
            if not members: continue
            groups=[]
            for xc,sp in members:
                if not groups or abs(xc-groups[-1]["x"])>10.5:
                    groups.append({"x":xc,"spans":[sp]})
                else:
                    g=groups[-1]; g["spans"].append(sp); g["x"]=sum((q["bbox"][0]+q["bbox"][2])/2 for q in g["spans"])/len(g["spans"])
            broad.extend(groups)
        # Any text outside the long-rule corridor is ignored for table rows.
        row_groups=broad
        if not (3 <= len(row_groups) <= 8):
            return None
        for g in row_groups:
            g["bbox"]=(min(sp["bbox"][0] for sp in g["spans"]),min(sp["bbox"][1] for sp in g["spans"]),max(sp["bbox"][2] for sp in g["spans"]),max(sp["bbox"][3] for sp in g["spans"]))
        row_groups.sort(key=lambda g:g["x"])
        # Header/first row gives the logical column locations. Cluster y-centres
        # using a 12pt minimum gap, which also combines split glyph spans.
        header_spans=row_groups[0]["spans"]
        ys=sorted([(float((s["bbox"][1]+s["bbox"][3])/2),s) for s in header_spans], key=lambda z:z[0])
        col_groups=[]
        for yc,sp in ys:
            if not col_groups or abs(yc-col_groups[-1]["y"])>30.0:
                col_groups.append({"y":yc,"spans":[sp]})
            else:
                g=col_groups[-1]; g["spans"].append(sp); g["y"]=(g["y"]*(len(g["spans"])-1)+yc)/len(g["spans"])
        if not (8 <= len(col_groups) <= 14):
            return None
        col_groups.sort(key=lambda g:g["y"])

        # Build column intervals from midpoint gaps between observed centres.
        centers=[g["y"] for g in col_groups]
        y_min=min(s["bbox"][1] for s in spans); y_max=max(s["bbox"][3] for s in spans)
        bounds=[]
        edges=[y_min]+[(centers[i]+centers[i+1])/2 for i in range(len(centers)-1)]+[y_max]
        for i in range(len(centers)):
            bounds.append((edges[i],edges[i+1]))

        # Row intervals are similarly based on x-centres, but constrained to the
        # actual text extent; this avoids the missing middle rule problem.
        rcenters=[g["x"] for g in row_groups]
        rx0=min(s["bbox"][0] for s in spans); rx1=max(s["bbox"][2] for s in spans)
        redges=[rx0]+[(rcenters[i]+rcenters[i+1])/2 for i in range(len(rcenters)-1)]+[rx1]
        rbounds=[(redges[i],redges[i+1]) for i in range(len(rcenters))]

        rows=[]
        for rg,rb in zip(row_groups, rbounds):
            row=[]
            for cb in bounds:
                cell_sp=[s for s in spans if rb[0]-1<=((s["bbox"][0]+s["bbox"][2])/2)<=rb[1]+1 and cb[0]-2<=((s["bbox"][1]+s["bbox"][3])/2)<=cb[1]+2]
                cell_sp.sort(key=lambda s:s["bbox"][1], reverse=True)  # bottom-to-top for btLr
                if cell_sp:
                    text="".join(s.get("text","") for s in cell_sp).strip()
                    s0=cell_sp[0]
                    row.append({"text":text,"font":s0.get("font","Arial"),"size":float(s0.get("size",8)),"bold":bool(s0.get("flags",0)&16),"italic":bool(s0.get("flags",0)&2),"color":(0,0,0)})
                else:
                    row.append({"text":"","font":"Arial","size":8})
            rows.append(row)

        # Include a nearby vertical caption (for example "TABLE 1 | ...")
        # in the same visual fallback image so it cannot be emitted as
        # horizontal text by the normal paragraph path.
        cap_spans=[]
        for ln in lines or []:
            b=ln.get("bbox")
            if not b or b[2] > x0 + 2 or b[2] < x0 - 22:
                continue
            if b[3] >= y_min-8 and b[1] <= y_max+8:
                cap_spans.extend(ln.get("spans",[]))
        # Also absorb vertical footnote text immediately to the right of the
        # table. The journal sample uses this space for the table definition.
        foot_spans=[]
        for ln in lines or []:
            b=ln.get("bbox")
            if not b: continue
            if b[0] >= x1-1 and b[0] <= x1+30 and (b[3]-b[1]) > max(100.0,(b[2]-b[0])*3):
                foot_spans.extend(ln.get("spans",[]))
        side_spans=cap_spans+foot_spans
        if side_spans:
            sx0=min(s["bbox"][0] for s in side_spans); sx1=max(s["bbox"][2] for s in side_spans)
            sy0=min(s["bbox"][1] for s in side_spans); sy1=max(s["bbox"][3] for s in side_spans)
            bbox=(min(rx0,sx0), min(y_min,sy0), max(rx1,sx1), max(y_max,sy1))
        else:
            bbox=(rx0,y_min,rx1,y_max)
        # Require at least half the row/column cells to contain text.
        nonempty=sum(bool(c.get("text")) for r in rows for c in r)
        if nonempty < len(rows)*len(bounds)*0.45:
            return None
        return {"bbox":bbox,"rows":rows,"row_bounds":rbounds,"col_bounds":bounds,"rotation":90}
    except Exception:
        return None

def build_docx_exact_layout(pdf_bytes: bytes, dpi=200, use_ocr_for_scanned=True,
                             ocr_lang="en", target_lang=None, translator_fn=None,
                             include_figure_crops=True, progress_cb=None, report=None,
                             font_override=None, table_layout=False, include_drawings=True,
                             custom_page_size_pt=None, trace_mode=False):
    """Builds a pixel-faithful DOCX: every page is sized to match the PDF page.

    custom_page_size_pt: optional (width_pt, height_pt). When given, this
    size is applied uniformly to every generated Word page instead of each
    page's own detected PDF dimensions (content stays positioned at its
    original absolute coordinates, so the page canvas around it may be
    larger/smaller than the source -- a note is added to the report).

    trace_mode: when True, every generated page carries a faded, washed-out
    image of its own original PDF page as a background watermark, floating
    behind the editable content on that SAME page (not a separate page) --
    like the source showing through from the back of the sheet.

    When table_layout is True:
    Groups adjacent lines into unified paragraph blocks and creates a SINGLE floating
    borderless Word table (<w:tbl> with w:tblpPr) for each paragraph at its exact top-left
    pixel position (para_x0, para_y0). Table background is 100% clear (<w:shd w:fill="auto"/>),
    making all paragraph text natively continuous and easy to click and edit."""
    report = report if report is not None else []
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    doc = Document()

    first_page = pdf_doc[0]
    section = doc.sections[0]
    if custom_page_size_pt:
        section.page_width = Pt(custom_page_size_pt[0])
        section.page_height = Pt(custom_page_size_pt[1])
        section.orientation = WD_ORIENT.LANDSCAPE if custom_page_size_pt[0] > custom_page_size_pt[1] else WD_ORIENT.PORTRAIT
    else:
        section.page_width = Pt(first_page.rect.width)
        section.page_height = Pt(first_page.rect.height)
        section.orientation = WD_ORIENT.LANDSCAPE if first_page.rect.width > first_page.rect.height else WD_ORIENT.PORTRAIT
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin", "header_distance", "footer_distance"):
        setattr(section, m, Pt(0))

    ocr_engine = None
    if use_ocr_for_scanned and PADDLEOCR_AVAILABLE:
        ocr_engine = get_ocr_engine(ocr_lang)

    n_pages = len(pdf_doc)
    per_page = []
    hf_candidates = {"header": {}, "footer": {}}

    for i, page in enumerate(pdf_doc):
        scanned = is_scanned_page(page)
        if scanned:
            lines = ocr_lines_multirun(render_page_image(page, dpi=dpi), dpi, ocr_engine) if ocr_engine else []
            images, drawings, pdf_tables, hint_boxes = [], [], [], []
        else:
            lines, images = extract_lines_multirun(page)
            try:
                pdf_tables = list(page.find_tables().tables)
            except Exception:
                pdf_tables = []
            # PyMuPDF find_tables() does not recognize some rotated tables.
            # Detect them from long vector rules + vertical text geometry.
            rotated_table = detect_rotated_pdf_table(page, lines, [])
            table_bboxes = [tuple(t.bbox) for t in pdf_tables]
            if rotated_table:
                table_bboxes.append(tuple(rotated_table["bbox"]))
            raw_drawings, raw_hint_boxes = extract_drawings(page, include_drawings=include_drawings)
            if rotated_table is None:
                rotated_table = detect_rotated_pdf_table(page, lines, raw_drawings)
                if rotated_table:
                    table_bboxes.append(tuple(rotated_table["bbox"]))
            drawings = [
                d for d in raw_drawings
                if not _is_bbox_inside_any_table(d.get("bbox"), table_bboxes, threshold=0.35)
            ]
            # Boxing-hint rectangles are never rendered and never treated as
            # real PDF tables -- they are pure grouping metadata for the
            # free-flow paragraph reconstruction below.
            hint_boxes = [
                h for h in raw_hint_boxes
                if not _is_bbox_inside_any_table(h.get("bbox"), table_bboxes, threshold=0.35)
            ]
        hyperlinks = extract_hyperlinks(page)
        ph = page.rect.height

        for ln in lines:
            y0, y1 = ln["bbox"][1], ln["bbox"][3]
            text = " ".join(s["text"] for s in ln["spans"]).strip()
            if not text:
                continue
            zone = "header" if y1 < ph * 0.10 else ("footer" if y0 > ph * 0.90 else None)
            if zone:
                norm = _normalize_hf_text(text)
                entry = hf_candidates[zone].setdefault(norm, [0, text])
                entry[0] += 1

        per_page.append({
            "page": page, "scanned": scanned, "lines": lines, "images": images,
            "drawings": drawings, "hyperlinks": hyperlinks,
            "tables": pdf_tables if not scanned else [],
            "rotated_table": rotated_table if not scanned else None,
            "table_bboxes": table_bboxes if not scanned else [],
            "hint_boxes": hint_boxes,
        })

    promoted_header = max(hf_candidates["header"].items(), key=lambda kv: kv[1][0], default=None)
    promoted_footer = max(hf_candidates["footer"].items(), key=lambda kv: kv[1][0], default=None)

    if n_pages < 2:
        header_norm = footer_norm = None
    else:
        min_repeats = max(2, (n_pages + 1) // 2)
        header_norm = promoted_header[0] if promoted_header and promoted_header[1][0] >= min_repeats else None
        footer_norm = promoted_footer[0] if promoted_footer and promoted_footer[1][0] >= min_repeats else None

    if header_norm:
        doc.sections[0].header.paragraphs[0].text = promoted_header[1][1]
    if footer_norm:
        doc.sections[0].footer.paragraphs[0].text = promoted_footer[1][1]

    page_meta = []
    current_size = (round(first_page.rect.width), round(first_page.rect.height))
    custom_size_note_added = False

    for i, pdata in enumerate(per_page):
        page = pdata["page"]
        if progress_cb:
            progress_cb(i + 1, n_pages)

        page_size = (round(page.rect.width), round(page.rect.height))
        if custom_page_size_pt:
            # Uniform custom size for every page -- never switch sections,
            # even if the source PDF itself has mixed page sizes.
            if i == 0:
                section = doc.sections[0]
            else:
                doc.add_page_break()
            custom_size_r = (round(custom_page_size_pt[0]), round(custom_page_size_pt[1]))
            if page_size != custom_size_r and not custom_size_note_added:
                custom_size_note_added = True
                report.append(
                    f"Custom page size ({custom_page_size_pt[0]:.1f}x{custom_page_size_pt[1]:.1f}pt) differs from "
                    f"one or more original PDF page sizes -- the page canvas was resized, but absolutely-"
                    f"positioned content keeps its original coordinates and may not perfectly fill/align to "
                    f"the new page."
                )
        elif i == 0:
            section = doc.sections[0]
        elif page_size != current_size:
            section = doc.add_section(WD_SECTION.NEW_PAGE)
            section.page_width = Pt(page.rect.width)
            section.page_height = Pt(page.rect.height)
            section.orientation = WD_ORIENT.LANDSCAPE if page.rect.width > page.rect.height else WD_ORIENT.PORTRAIT
            for m in ("top_margin", "bottom_margin", "left_margin", "right_margin", "header_distance", "footer_distance"):
                setattr(section, m, Pt(0))
            current_size = page_size
            report.append(f"Page {i + 1} has a different page size/orientation from the previous page -- "
                           f"started a new Word section so it keeps its own true dimensions.")
        else:
            doc.add_page_break()

        anchor_p = doc.add_paragraph()
        p_pr = anchor_p._p.get_or_add_pPr()
        p_pr.append(parse_xml(
            '<w:spacing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
        ))
        pw, ph = page.rect.width, page.rect.height

        if trace_mode:
            wm_w, wm_h = custom_page_size_pt if custom_page_size_pt else (pw, ph)
            try:
                trace_png = render_pdf_page_trace_image(page)
                add_page_watermark_image(anchor_p, trace_png, wm_w, wm_h)
            except Exception as e:
                report.append(f"Page {i + 1}: watermark could not be generated: {e}")

        for r in pdata["drawings"]:
            x0, y0, x1, y1 = clamp_bbox_to_page(r["bbox"], pw, ph, min_size=0.75)
            add_absolute_rect(anchor_p, x0, y0, x1 - x0, y1 - y0, r["color_hex"])

        if pdata.get("rotated_table"):
            rt = pdata["rotated_table"]
            try:
                crop_rect = fitz.Rect(*rt["bbox"])
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=crop_rect, alpha=False)
                crop_png = pix.tobytes("png")
                add_absolute_image_bytes(anchor_p, crop_png, crop_rect.x0, crop_rect.y0, crop_rect.width, crop_rect.height)
                report.append(f"Page {i + 1}: detected a 90-degree rotated table; preserved its exact visual layout as an embedded editable-document image fallback because Word does not reliably rotate full native tables across renderers.")
            except Exception as e:
                report.append(f"Page {i + 1}: rotated table image fallback failed: {e}")

        for img in pdata["images"]:
            x0, y0, x1, y1 = clamp_bbox_to_page(img["bbox"], pw, ph)
            try:
                add_absolute_image_bytes(anchor_p, img["bytes"], x0, y0, x1 - x0, y1 - y0)
            except Exception as e:
                report.append(f"Page {i + 1}: an image could not be embedded (unsupported codec) and was skipped: {e}")
                continue

        n_lines_rendered = 0

        # IMPORTANT: PDF tables are reconstructed as real native Word tables.
        # The previous implementation placed every table line into an independent
        # VML textbox. That can look correct in LibreOffice but Word's layout engine
        # may collapse/reposition those anchored textboxes, producing the overlap seen
        # in the user's test document. Native fixed-layout tables are much more stable
        # and are also the structure used by iLovePDF-style DOCX conversions.

        if pdata["tables"]:
            for pdf_table in pdata["tables"]:
                try:
                    add_absolute_native_pdf_table(
                        anchor_p, page, pdf_table, font_override=font_override,
                        target_lang=target_lang, translator_fn=translator_fn
                    )
                except Exception as e:
                    report.append(f"Page {i + 1}: native table reconstruction failed; table text was kept as positioned text. {e}")

        bands = []
        if table_layout:
            # ALWAYS group free-flow text into paragraph-blocks and render them
            # with add_absolute_paragraph_table -- regardless of whether this
            # page also contains a detected table. Previously this branch only
            # ran when pdata["tables"] was empty, so any page WITH a table fell
            # through to the per-line VML textbox path below, which has no
            # page-width clamp and produced the overflow/overlap you saw.
            #
            # Lines that belong to a detected table are excluded here (they
            # are already rendered above via add_absolute_native_pdf_table),
            # exactly as the old textbox branch used to exclude them.
            non_table_lines = [
                ln for ln in pdata["lines"]
                if not (pdata["table_bboxes"] and _is_bbox_inside_any_table(
                    ln["bbox"], pdata["table_bboxes"], threshold=0.35))
            ]

            def _filter_hf(block):
                out = []
                for ln in block:
                    text = " ".join(s["text"] for s in ln["spans"]).strip()
                    y0, y1 = ln["bbox"][1], ln["bbox"][3]
                    zone = "header" if y1 < ph * 0.10 else ("footer" if y0 > ph * 0.90 else None)
                    if zone and text and _normalize_hf_text(text) == (header_norm if zone == "header" else footer_norm):
                        continue
                    out.append(ln)
                return out

            # Boxing-hint rectangles (blue annotation boxes from the source
            # PDF -- never rendered themselves) force ALL lines they enclose
            # into a SINGLE table matching the box's exact width and left
            # indentation, overriding the normal gap-based auto-split. Lines
            # outside every hint box fall through to the existing behaviour.
            boxed_groups, leftover_lines = partition_lines_by_hint_boxes(
                non_table_lines, pdata.get("hint_boxes") or []
            )

            all_page_blocks = []
            for hint_box, box_lines in boxed_groups:
                filtered = _filter_hf(box_lines)
                if filtered:
                    all_page_blocks.append((filtered, hint_box["bbox"]))

            # IMPORTANT FIX: do NOT reconstruct a two-column journal page as
            # many horizontal one-row tables. That makes Word lay the page out
            # row-by-row: the taller left cell pushes the next right-column
            # block downward. Detect the page-level columns once and create ONE
            # side-by-side Word table covering the continuous column flow.
            full_cols = detect_full_page_columns_for_reconstruction(leftover_lines, pw, ph)

            if full_cols:
                left_lines = _filter_hf(full_cols["left"])
                right_lines = _filter_hf(full_cols["right"])
                spanning_lines = _filter_hf(full_cols["spanning"])

                column_regions = []
                if left_lines:
                    column_regions.append({
                        "bbox": [min(l["bbox"][0] for l in left_lines),
                                 min(l["bbox"][1] for l in left_lines),
                                 max(l["bbox"][2] for l in left_lines),
                                 max(l["bbox"][3] for l in left_lines)],
                        "lines": left_lines,
                    })
                if right_lines:
                    column_regions.append({
                        "bbox": [min(l["bbox"][0] for l in right_lines),
                                 min(l["bbox"][1] for l in right_lines),
                                 max(l["bbox"][2] for l in right_lines),
                                 max(l["bbox"][3] for l in right_lines)],
                        "lines": right_lines,
                    })

                if len(column_regions) >= 2:
                    # IMPORTANT: keep the independent-column fix, but represent
                    # the columns as a REAL native Word table instead of VML
                    # textboxes.  The table has ONE row and one cell per
                    # detected column, so Word keeps the columns side-by-side
                    # for the full vertical flow without turning the content
                    # into editable textboxes.  This also preserves dynamic
                    # 2/3/... column layouts because add_absolute_multi_column_table
                    # builds one cell per detected region.
                    add_absolute_multi_column_table(
                        anchor_p, anchor_p, column_regions, page_width_pt=pw,
                        page_height_pt=ph, hyperlinks=pdata["hyperlinks"],
                        font_override=font_override
                    )
                    n_lines_rendered += sum(len(r["lines"]) for r in column_regions)

                # Page-wide captions/figure labels that cross the gutter remain
                # independently positioned at their original PDF coordinates.
                if spanning_lines:
                    span_blocks = group_lines_into_paragraph_blocks(spanning_lines)
                    span_entries = [(b, None) for b in span_blocks if b]
                    if span_entries:
                        add_absolute_page_paragraph_table(
                            anchor_p, anchor_p, span_entries, page_width_pt=pw, page_height_pt=ph,
                            hyperlinks=pdata["hyperlinks"], font_override=font_override
                        )
                        n_lines_rendered += sum(len(b[0]) for b in span_entries)

                # Keep debug metadata compatible with the rest of the app.
                bands = [{
                    "bbox": [min(r["bbox"][0] for r in column_regions),
                             min(r["bbox"][1] for r in column_regions),
                             max(r["bbox"][2] for r in column_regions),
                             max(r["bbox"][3] for r in column_regions)],
                    "column_count": 2,
                    "columns": column_regions,
                }]
            else:
                # No reliable continuous page-level split: retain the existing
                # region-aware fallback for irregular/mixed layouts.
                bands = detect_page_regions_recursive(leftover_lines, pw, ph)
                for band in bands:
                    for col in band["columns"]:
                        col["lines"] = _filter_hf(col["lines"])
                bands = [b for b in bands if any(c["lines"] for c in b["columns"])]

                for band in bands:
                    if band["column_count"] == 1:
                        band_lines = band["columns"][0]["lines"]
                        if not band_lines:
                            continue
                        band_blocks = group_lines_into_paragraph_blocks(band_lines)
                        band_entries = [(b, None) for b in band_blocks if b]
                        if band_entries:
                            add_absolute_page_paragraph_table(
                                anchor_p, anchor_p, band_entries, page_width_pt=pw, page_height_pt=ph,
                                hyperlinks=pdata["hyperlinks"], font_override=font_override
                            )
                            n_lines_rendered += sum(len(b[0]) for b in band_entries)
                    else:
                        column_regions = [c for c in band["columns"] if c["lines"]]
                        if column_regions:
                            add_absolute_multi_column_table(
                                anchor_p, anchor_p, column_regions, page_width_pt=pw, page_height_pt=ph,
                                hyperlinks=pdata["hyperlinks"], font_override=font_override
                            )
                            n_lines_rendered += sum(len(r["lines"]) for r in column_regions)

                if all_page_blocks:
                    add_absolute_page_paragraph_table(
                        anchor_p, anchor_p, all_page_blocks, page_width_pt=pw, page_height_pt=ph,
                        hyperlinks=pdata["hyperlinks"], font_override=font_override
                    )
                    n_lines_rendered += sum(len(b[0]) for b in all_page_blocks)
        else:
            # table_layout disabled entirely: fall back to independent
            # per-line floating textboxes (legacy exact-pixel mode).
            for ln in pdata["lines"]:
                text = " ".join(s["text"] for s in ln["spans"]).strip()
                y0, y1 = ln["bbox"][1], ln["bbox"][3]
                zone = "header" if y1 < ph * 0.10 else ("footer" if y0 > ph * 0.90 else None)
                if zone and text and _normalize_hf_text(text) == (header_norm if zone == "header" else footer_norm):
                    continue
                if pdata["table_bboxes"] and _is_bbox_inside_any_table(ln["bbox"], pdata["table_bboxes"], threshold=0.35):
                    continue

                x0, y0, x1, y1 = clamp_bbox_to_page(ln["bbox"], pw, ph)
                spans = ln["spans"]
                if target_lang and target_lang != "None (keep original)" and translator_fn:
                    merged_text = translate_text(" ".join(s["text"] for s in spans), target_lang, translator_fn)
                    spans = [{**spans[0], "text": merged_text}]
                uri = find_hyperlink_for_bbox(ln["bbox"], pdata["hyperlinks"])
                add_absolute_textbox_multirun(
                    anchor_p, spans, x0, y0, max(x1 - x0, 8), max(y1 - y0, 8) + 2,
                    hyperlink_uri=uri, font_override=font_override
                )
        # ------------------------------------------------------------- #
        # 33. Overlap / overflow detection -- checked against the actual
        # bboxes about to be placed on this page (tables, images, and the
        # free-flow text bands), BEFORE the DOCX is finalized.
        # ------------------------------------------------------------- #
        check_elements = []
        for t in pdata.get("tables") or []:
            try:
                check_elements.append({"type": "table", "bbox": list(t.bbox)})
            except Exception:
                pass
        for img in pdata.get("images") or []:
            if img.get("bbox"):
                check_elements.append({"type": "image", "bbox": list(img["bbox"])})
        band_list = bands
        for band in band_list:
            for col in band["columns"]:
                if col["lines"]:
                    check_elements.append({"type": "text", "bbox": list(col["bbox"])})

        overlap_findings = detect_element_overlaps(check_elements, pw, ph)
        if overlap_findings:
            shown = overlap_findings[:5]
            for f in shown:
                report.append(f"Page {i + 1}: {f['kind']} detected -- {f['detail']}")
            if len(overlap_findings) > 5:
                report.append(f"Page {i + 1}: {len(overlap_findings) - 5} additional "
                               f"overlap/overflow finding(s) omitted from the report for brevity.")

        if page.rotation:
            report.append(f"Page {i + 1}: source PDF page has a {page.rotation}-degree rotation "
                           f"applied; content coordinates were read in PyMuPDF's already-rotated "
                           f"page space, but per-element counter-rotation in Word is not applied -- "
                           f"verify this page visually.")

        debug_entry = page_debug_entry(
            page_num=i + 1, width_pt=pw, height_pt=ph,
            bands=band_list,
            tables=[{"bbox": list(t.bbox)} for t in (pdata.get("tables") or []) if hasattr(t, "bbox")],
            images=[{"bbox": list(img["bbox"])} for img in (pdata.get("images") or []) if img.get("bbox")],
            text_blocks=[{"bbox": list(ln["bbox"])} for ln in pdata.get("lines", [])],
            overlaps=overlap_findings,
        )

        page_meta.append({
            "page_idx": i,
            "page_num": i + 1,
            "scanned": pdata["scanned"],
            "n_lines": n_lines_rendered,
            "n_images": len(pdata["images"]),
            "hint_boxes": pdata.get("hint_boxes", []),
            "boxed_groups": boxed_groups if table_layout else [],
            "leftover_lines": leftover_lines if table_layout else pdata.get("lines", []),
            "non_table_lines": non_table_lines if table_layout else pdata.get("lines", []),
            "rotation": page.rotation,
            "debug_entry": debug_entry,
            "overlaps": overlap_findings,
        })

    pdf_doc.close()
    # Keep processing diagnostics in the Streamlit report instead of adding
    # an extra physical page to the reconstructed document.
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read(), page_meta, report


def render_boxing_debug_page(pdf_bytes: bytes, page_idx: int, hint_boxes: list, dpi: int = 150):
    """Renders a PDF page with detected hint_boxes drawn over it for visual debugging.
    Does NOT modify the PDF and does NOT alter DOCX generation."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(img)

    scale_x = img.width / page.rect.width
    scale_y = img.height / page.rect.height

    for bi, hb in enumerate(hint_boxes):
        x0, y0, x1, y1 = hb["bbox"]
        bx0, by0, bx1, by1 = x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y
        color_hex = hb.get("color_hex", "0066FF")
        try:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            outline_rgb = (r, g, b)
        except Exception:
            outline_rgb = (0, 102, 255)

        draw.rectangle([bx0, by0, bx1, by1], outline=outline_rgb, width=3)
        draw.text((bx0 + 4, max(0, by0 - 14)), f"Box #{bi+1}", fill=outline_rgb)

    doc.close()
    return img


# --------------------------------------------------------------------------- #
# Pre-conversion detection preview
#
# Lightweight, per-page detection used purely for the UI preview shown right
# after upload (before the user clicks "Convert"). Reuses the exact same
# detection primitives as the real pipeline (is_scanned_page,
# detect_layout_regions, extract_native_lines) but skips OCR and
# PP-StructureV3 table extraction so the preview renders quickly even for
# large documents.
# --------------------------------------------------------------------------- #

def _detect_native_tables(page: "fitz.Page"):
    """Detect real tables directly from the PDF's vector/text structure using
    PyMuPDF's built-in table finder -- works even without DocLayout-YOLO
    installed, and is what feeds the "Table" boxes/count in the preview."""
    try:
        finder = page.find_tables()
        return [tuple(t.bbox) for t in finder.tables]
    except Exception:
        return []


def _classify_preview_line(bbox, page_height, regions, seen_any_region, table_bboxes=None):
    """Return a human-readable element label + a counts-dict key for one
    detected text line, using DocLayout-YOLO regions when available and
    falling back to a position heuristic otherwise."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    if table_bboxes:
        for tb in table_bboxes:
            if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                return "Table Text", "table"

    if seen_any_region:
        for r in regions:
            rb = r["bbox"]
            if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                role = r["role"]
                if role == "heading":
                    return "Title", "title"
                if role == "caption":
                    return "Caption", "text"
                if role == "table":
                    return "Table Text", "table"
                if role in ("figure", "formula"):
                    return None, None  # handled as an image/formula region, not a text line
                break  # "skip"/"body"/unmatched roles fall through to the position heuristic

    # Position heuristic (also used as the fallback when no layout model
    # is installed, or a line fell outside every detected region).
    y0, y1 = bbox[1], bbox[3]
    if y1 < page_height * 0.10:
        return "Header", "header"
    if y0 > page_height * 0.90:
        return "Footer", "footer"

    return "Text", "text"


def detect_page_preview(pdf_bytes: bytes, layout_conf: float = 0.25, device: str = "cpu",
                         dpi: int = 150) -> list:
    """Per-page detection summary for the UI: which of
    header / footer / title / text / images / tables were found on each
    page, plus a font name / size / style table for every detected text line.

    Returns a list (one entry per page) of:
        {
          "page_num": int, "scanned": bool, "layout_model_used": bool,
          "counts": {"header": n, "footer": n, "title": n, "text": n, "images": n, "tables": n},
          "font_rows": [{"Element", "Font Name", "Size (pt)", "Style", "Text Preview"}, ...],
        }
    """
    layout_model = get_layout_model() if DOCLAYOUT_AVAILABLE and HF_HUB_AVAILABLE else None
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    previews = []
    try:
        for i, page in enumerate(pdf_doc):
            scanned = is_scanned_page(page)
            page_height = page.rect.height

            regions = []
            if not scanned and layout_model is not None:
                page_img = render_page_image(page, dpi=dpi)
                regions = detect_layout_regions(page_img, dpi, layout_model, conf=layout_conf, device=device)

            counts = {"header": 0, "footer": 0, "title": 0, "text": 0, "images": 0, "tables": 0}
            font_rows = []

            if scanned:
                counts["images"] = 1  # the whole page is effectively one scanned image
            else:
                lines, images = extract_native_lines(page)
                counts["images"] = len(images)

                table_bboxes = _detect_native_tables(page)
                counts["tables"] = len(table_bboxes)

                sizes = [ln.get("size", 11) for ln in lines if ln.get("text", "").strip()]
                median_size = sorted(sizes)[len(sizes) // 2] if sizes else 11.0

                for ln in lines:
                    label, count_key = _classify_preview_line(
                        ln["bbox"], page_height, regions, bool(regions), table_bboxes
                    )
                    if label is None:
                        continue  # figure/formula region -- not a text line

                    # Font-size based title heuristic, only used when no
                    # layout model is installed to classify headings.
                    if not regions and count_key == "text":
                        is_bigger = ln.get("size", 11) >= median_size * 1.15
                        is_bold_and_bigger = ln.get("bold") and ln.get("size", 11) >= median_size * 1.05
                        if is_bigger or is_bold_and_bigger:
                            label, count_key = "Title", "title"

                    if count_key != "table":
                        counts[count_key] = counts.get(count_key, 0) + 1

                    style_bits = []
                    if ln.get("bold"):
                        style_bits.append("Bold")
                    if ln.get("italic"):
                        style_bits.append("Italic")
                    style = " + ".join(style_bits) if style_bits else "Regular"

                    text_preview = (ln.get("text") or "").strip()
                    if len(text_preview) > 70:
                        text_preview = text_preview[:70] + "…"

                    font_rows.append({
                        "Element": label,
                        "Font Name": ln.get("font", "Calibri"),
                        "Size (pt)": round(float(ln.get("size", 11)), 1),
                        "Style": style,
                        "Text Preview": text_preview,
                    })

            previews.append({
                "page_num": i + 1,
                "scanned": scanned,
                "layout_model_used": layout_model is not None,
                "counts": counts,
                "font_rows": font_rows,
            })
    finally:
        pdf_doc.close()
    return previews


# --------------------------------------------------------------------------- #
# Visual (bounding-box) detection preview -- draws colored, labeled boxes
# over the rendered page image (header/footer/title/text/image), the same
# style as an object-detection preview.
# --------------------------------------------------------------------------- #

_PREVIEW_BOX_COLORS = {
    "header": (255, 140, 0),    # orange
    "footer": (147, 51, 234),   # purple
    "title": (236, 26, 122),    # magenta/pink
    "text": (16, 185, 129),     # mint green
    "image": (37, 99, 235),     # blue
    "table": (20, 184, 166),    # teal
}


def _get_preview_font(size: int = 18):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_labeled_box(draw: "ImageDraw.ImageDraw", bbox_px, label: str, color, font):
    x0, y0, x1, y1 = bbox_px
    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

    try:
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = len(label) * 8, 14

    pad = 4
    pill_h = th + 2 * pad
    pill_top = y0 - pill_h if y0 - pill_h >= 0 else y0
    draw.rectangle([x0, pill_top, x0 + tw + 2 * pad, pill_top + pill_h], fill=color)
    draw.text((x0 + pad, pill_top + pad - 1), label, fill=(255, 255, 255), font=font)


def render_page_detection_boxes(pdf_bytes: bytes, page_idx: int, layout_conf: float = 0.25,
                                 device: str = "cpu", dpi: int = 150) -> "Image.Image":
    """Renders one PDF page as an image with colored, labeled bounding boxes
    over every detected Header / Footer / Title / Text / Image region --
    an object-detection-style visual preview of what the converter sees."""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = pdf_doc[page_idx]
        img = render_page_image(page, dpi=dpi).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = _get_preview_font(max(14, dpi // 10))
        px_per_pt = dpi / 72.0

        if is_scanned_page(page):
            draw.rectangle([4, 4, img.width - 4, img.height - 4],
                            outline=_PREVIEW_BOX_COLORS["image"], width=6)
            _draw_labeled_box(draw, [4, 4, img.width - 4, min(40, img.height - 4)],
                               "Scanned Page (OCR needed)", _PREVIEW_BOX_COLORS["image"], font)
            return img

        layout_model = get_layout_model() if DOCLAYOUT_AVAILABLE and HF_HUB_AVAILABLE else None
        regions = []
        if layout_model is not None:
            page_img_for_model = render_page_image(page, dpi=dpi)
            regions = detect_layout_regions(page_img_for_model, dpi, layout_model, conf=layout_conf, device=device)

        lines, images = extract_native_lines(page)
        sizes = [ln.get("size", 11) for ln in lines if ln.get("text", "").strip()]
        median_size = sorted(sizes)[len(sizes) // 2] if sizes else 11.0

        table_bboxes = _detect_native_tables(page)

        buckets = {"header": [], "footer": [], "title": [], "text": []}
        for ln in lines:
            label, count_key = _classify_preview_line(ln["bbox"], page.rect.height, regions, bool(regions), table_bboxes)
            if label is None:
                continue
            if count_key == "table":
                continue  # drawn as one box per detected table region below, not per line
            if not regions and count_key == "text":
                is_bigger = ln.get("size", 11) >= median_size * 1.15
                is_bold_bigger = ln.get("bold") and ln.get("size", 11) >= median_size * 1.05
                if is_bigger or is_bold_bigger:
                    count_key = "title"
            buckets[count_key].append(ln)

        # Merge nearby same-role lines into paragraph-level blocks so the
        # preview shows a handful of clear boxes rather than one per line.
        for count_key, bucket_lines in buckets.items():
            label = count_key.capitalize()
            for group in group_lines_into_paragraphs(bucket_lines):
                xs0 = min(l["bbox"][0] for l in group)
                ys0 = min(l["bbox"][1] for l in group)
                xs1 = max(l["bbox"][2] for l in group)
                ys1 = max(l["bbox"][3] for l in group)
                bbox_px = [xs0 * px_per_pt, ys0 * px_per_pt, xs1 * px_per_pt, ys1 * px_per_pt]
                _draw_labeled_box(draw, bbox_px, label, _PREVIEW_BOX_COLORS[count_key], font)

        for tb in table_bboxes:
            bbox_px = [v * px_per_pt for v in tb]
            _draw_labeled_box(draw, bbox_px, "Table", _PREVIEW_BOX_COLORS["table"], font)

        for img_info in images:
            bbox_px = [v * px_per_pt for v in img_info["bbox"]]
            _draw_labeled_box(draw, bbox_px, "Image", _PREVIEW_BOX_COLORS["image"], font)

        return img
    finally:
        pdf_doc.close()


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(page_title="PDF → Editable DOCX", page_icon="📄", layout="centered")
    st.title("📄 PDF → Editable Word (DOCX) Converter")
    st.caption(
        "Exact Visual Layout mode preserves font size/style/color/alignment/images "
        "at their original position. Reflow mode reorganizes into headings/paragraphs/tables."
    )

    missing = []
    if not DOCLAYOUT_AVAILABLE:
        missing.append("doclayout-yolo")
    if not HF_HUB_AVAILABLE:
        missing.append("huggingface_hub")
    if not PADDLEOCR_AVAILABLE:
        missing.append("paddleocr (also provides PP-StructureV3)")

    # Uploader is read before the sidebar so the sidebar can show detected
    # PDF page-size/application info and custom-size controls right after
    # upload, without a second uploader widget.
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    pdf_bytes = None
    pdf_page_info = None
    if uploaded is not None:
        pdf_bytes = uploaded.read()
        try:
            pdf_page_info = detect_pdf_page_info(pdf_bytes)
        except Exception as e:
            st.error(f"Could not read PDF page information: {e}")

        st.markdown("---")
        st.subheader("🔍 Page-by-Page Detection Preview")
        st.caption(
            "Detected before conversion, from the actual PDF content -- header, "
            "footer, title, and body text regions, plus every detected line's "
            "font name, size, and style."
        )
        with st.spinner("Detecting page content..."):
            try:
                page_previews = detect_page_preview(pdf_bytes)
            except Exception as e:
                page_previews = None
                st.error(f"Could not run detection preview: {e}")

        if page_previews:
            if not (DOCLAYOUT_AVAILABLE and HF_HUB_AVAILABLE):
                st.info(
                    "DocLayout-YOLO isn't installed, so Header/Footer/Title are "
                    "detected using a position + font-size heuristic instead of "
                    "the layout model. Install `doclayout-yolo` + `huggingface_hub` "
                    "for model-based detection."
                )

            page_labels = [f"Page {p['page_num']}" for p in page_previews]
            selected_label = st.selectbox("Select a page to inspect", page_labels, key="preview_page_select")
            sel = page_previews[page_labels.index(selected_label)]

            sel_idx = page_labels.index(selected_label)

            with st.spinner("Rendering detection boxes..."):
                try:
                    box_img = render_page_detection_boxes(pdf_bytes, sel_idx)
                    st.image(box_img, use_container_width=True,
                              caption=f"Page {sel['page_num']} -- detected regions")
                except Exception as e:
                    st.error(f"Could not render detection boxes: {e}")

            if sel["scanned"]:
                st.warning(
                    f"Page {sel['page_num']} appears to be a **scanned/image page** -- "
                    "no embedded font data is available here. Text and fonts will be "
                    "recovered via OCR during conversion."
                )
            else:
                c = sel["counts"]
                cols = st.columns(6)
                cols[0].metric("Header", c.get("header", 0))
                cols[1].metric("Footer", c.get("footer", 0))
                cols[2].metric("Title", c.get("title", 0))
                cols[3].metric("Text", c.get("text", 0))
                cols[4].metric("Images", c.get("images", 0))
                cols[5].metric("Tables", c.get("tables", 0))

                if sel["font_rows"]:
                    st.markdown("**Detected text -- font name, size & style:**")
                    st.dataframe(sel["font_rows"], use_container_width=True, hide_index=True)
                else:
                    st.caption("No text lines detected on this page.")

            with st.expander("Detection summary for every page"):
                summary_rows = [
                    {
                        "Page": p["page_num"],
                        "Type": "Scanned (image)" if p["scanned"] else "Text",
                        "Header": p["counts"].get("header", 0),
                        "Footer": p["counts"].get("footer", 0),
                        "Title": p["counts"].get("title", 0),
                        "Text": p["counts"].get("text", 0),
                        "Images": p["counts"].get("images", 0),
                        "Tables": p["counts"].get("tables", 0),
                    }
                    for p in page_previews
                ]
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Conversion mode",
            ["High-Fidelity Visual Layout", "Reflowable Structured Document"],
            index=0,
            help=(
                "High-Fidelity Visual Layout: preserves page geometry, text positions, fonts, images, "
                "tables and headers/footers as closely as Word allows. Text remains editable; "
                "rare rotated PDF tables use a visual fallback.\n\n"
                "Reflowable: uses DocLayout-YOLO + PP-StructureV3 to rebuild headings, "
                "flowing paragraphs, and real Word tables -- easier to heavily re-edit, "
                "but layout/position will differ from the source PDF."
            ),
        )
        target_lang = st.selectbox(
            "Translate output to",
            ["None (keep original)", "English", "Hindi", "Tamil", "Spanish",
             "French", "German", "Chinese (Simplified)", "Japanese"],
        )
        dpi = st.slider("Render DPI (higher = more accurate, slower)", 150, 300, 200, step=25)

        show_boxing_debug = st.checkbox(
            "Show boxing debug",
            value=False,
            help="Show visual debugging of detected boxing hint rectangles and text grouping."
        )

        st.markdown("---")
        override_font = st.checkbox(
            "Force a single font for the whole output",
            value=False,
            help=(
                "Off (default): keeps each line's own detected font from the "
                "PDF (mapped to a Word-safe substitute where needed), at its "
                "own detected size. On: ignores every detected font and "
                "writes the entire document in the font chosen below, at "
                "each line's own original size."
            ),
        )
        font_override = None
        if override_font:
            FONT_CHOICES = [
                "Arial", "Calibri", "Times New Roman", "Cambria", "Georgia",
                "Verdana", "Tahoma", "Garamond", "Century Gothic",
                "Franklin Gothic Book", "Trebuchet MS", "Segoe UI",
            ]
            chosen_font = st.selectbox("Font", FONT_CHOICES, index=0)
            font_override = {"font": chosen_font}

        st.markdown("---")
        st.subheader("Download Format")
        download_format = st.radio(
            "Choose what to download",
            ["Normal DOCX", "Trace Format (original page as background watermark)"],
            index=0,
            help=(
                "Normal DOCX: just the converted, editable Word document.\n\n"
                "Trace Format: the same editable DOCX and the same page count, "
                "but every page also carries a faded, washed-out image of its "
                "own original PDF page as a background watermark -- the source "
                "shows through right behind the editable content on that same "
                "page, like paper held up to the light. Useful for verifying "
                "the conversion/translation against the untouched original."
            ),
        )
        also_generate_both = False
        if uploaded is not None:
            also_generate_both = st.checkbox(
                "Also generate the other format",
                value=False,
                help=(
                    "Generates both a Normal DOCX and a Trace Format DOCX from "
                    "this one conversion, so both download buttons appear "
                    "afterwards. For Exact Visual Layout mode this reprocesses "
                    "the PDF a second time and takes roughly twice as long."
                ),
            )
        trace_mode = download_format.startswith("Trace")

        table_layout = False
        include_drawings = True
        if mode == "High-Fidelity Visual Layout":
            table_layout = st.checkbox(
                "Use native Word tables for columns and detected PDF tables",
                value=True,
                help=(
                    "Recommended: reconstructs multi-column regions and normal PDF tables as fixed-layout editable Word tables. "
                    "Each detected column is a separate table cell, so left/right columns stay side-by-side and independently editable. "
                    "Rare 90-degree rotated tables are detected dynamically and preserved visually so they do not break pagination or overlap article text. "
                    "If disabled, the legacy positioned-text fallback may use textboxes."
                ),
            )
            include_drawings = st.checkbox(
                "Include background title banners / section header bars",
                value=True,
                help="When checked, extracts colored title banners and background header bars from the PDF."
            )

        if mode == "Reflowable Structured Document":
            run_tables = st.checkbox("Extract tables with PP-StructureV3", value=True)
            device = st.selectbox("Inference device", ["cpu", "cuda:0"], index=0)
            layout_conf = st.slider("DocLayout-YOLO confidence threshold", 0.1, 0.6, 0.25, step=0.05)
        else:
            run_tables, device, layout_conf = True, "cpu", 0.25

        st.markdown("---")
        if missing:
            st.warning(
                "Some pipeline components aren't installed in this environment:\n\n"
                + "\n".join(f"- `{m}`" for m in missing)
                + "\n\nInstall from `requirements-full.txt`. Exact Visual Layout mode "
                  "only needs PaddleOCR (for scanned pages) -- it doesn't require "
                  "DocLayout-YOLO or PP-StructureV3 at all."
            )
        else:
            st.success("Full pipeline available: DocLayout-YOLO + PaddleOCR + PP-StructureV3 ✅")

        # ------------------------------------------------------------- #
        # PDF Page Settings -- detected page size/application + custom
        # width/height override. Only shown once a PDF is uploaded.
        # ------------------------------------------------------------- #
        custom_page_size_pt = None
        if pdf_page_info is not None:
            st.markdown("---")
            st.header("PDF Information")

            app_info = pdf_page_info["application"]
            st.subheader("PDF Application")
            st.text(f"Creator: {app_info['creator']}")
            st.text(f"Producer: {app_info['producer']}")

            pages = pdf_page_info["pages"]
            st.subheader("Page Size")
            st.text(f"Detected Pages: {len(pages)}")

            if not pdf_page_info["mixed"]:
                p0 = pages[0]
                st.text(f"Width:  {pt_to_unit(p0['width_pt'], 'mm'):.2f} mm")
                st.text(f"Height: {pt_to_unit(p0['height_pt'], 'mm'):.2f} mm")
                st.text(f"Orientation: {p0['orientation']}")
                st.text(f"Detected Size: {p0['standard_size']}")
            else:
                st.warning("Status: Mixed Page Sizes -- see details below.")

            st.subheader("Custom Page Size")
            apply_all = st.checkbox(
                "Apply custom size to all pages",
                value=False,
                key="apply_custom_all",
                help="When checked, the width/height below override every "
                     "page's original PDF dimensions in the generated DOCX. "
                     "When unchecked, each page's true detected size is kept.",
            )
            unit = st.selectbox("Unit", ["mm", "cm", "inch", "pt"], index=0, key="page_size_unit")

            default_w = round(pt_to_unit(pages[0]["width_pt"], unit), 2)
            default_h = round(pt_to_unit(pages[0]["height_pt"], unit), 2)
            custom_width = st.number_input("Width", min_value=0.0, value=default_w,
                                            step=0.1, key="custom_width_input")
            custom_height = st.number_input("Height", min_value=0.0, value=default_h,
                                             step=0.1, key="custom_height_input")

            if st.button("Apply Size"):
                if custom_width <= 0 or custom_height <= 0:
                    st.error("Width and Height must be greater than 0.")
                    st.session_state["custom_page_size_pt"] = None
                else:
                    st.session_state["custom_page_size_pt"] = (
                        unit_to_pt(custom_width, unit),
                        unit_to_pt(custom_height, unit),
                    )
                    st.success(f"Custom page size set: {custom_width} × {custom_height} {unit}")

            if apply_all:
                custom_page_size_pt = st.session_state.get("custom_page_size_pt")
                if custom_page_size_pt is None:
                    st.info("Enter a width/height above and click \"Apply Size\" to use a custom size.")

            with st.expander("Detected Page Details", expanded=pdf_page_info["mixed"]):
                for p in pages:
                    w_mm = pt_to_unit(p["width_pt"], "mm")
                    h_mm = pt_to_unit(p["height_pt"], "mm")
                    st.text(f"Page {p['page_num']}: {w_mm:.2f} x {h_mm:.2f} mm "
                            f"({p['orientation']}, {p['standard_size']})")

    if uploaded is not None:
        if st.button("Convert to editable DOCX", type="primary"):
            progress_bar = st.progress(0.0, text="Starting...")

            def progress_cb(cur, total):
                progress_bar.progress(cur / total, text=f"Processing page {cur}/{total}...")

            page_dims_pt = (
                [(p["width_pt"], p["height_pt"]) for p in pdf_page_info["pages"]]
                if pdf_page_info else None
            )

            try:
                docx_bytes_alt = None  # the "other" format, only built if also_generate_both

                if mode == "High-Fidelity Visual Layout":
                    docx_bytes, page_meta, report = build_docx_exact_layout(
                        pdf_bytes, dpi=dpi, use_ocr_for_scanned=True,
                        target_lang=target_lang, translator_fn=None, progress_cb=progress_cb,
                        font_override=font_override, table_layout=table_layout,
                        include_drawings=include_drawings,
                        custom_page_size_pt=custom_page_size_pt,
                        trace_mode=trace_mode,
                    )
                    progress_bar.progress(1.0, text="Done")
                    scanned_pages = sum(1 for m in page_meta if m["scanned"])
                    st.success(
                        f"Done! {len(page_meta)} page(s) processed "
                        f"({scanned_pages} via OCR, {len(page_meta) - scanned_pages} native text), "
                        f"exact original layout preserved."
                    )
                    if also_generate_both:
                        with st.spinner("Also generating the other download format..."):
                            docx_bytes_alt, _, _ = build_docx_exact_layout(
                                pdf_bytes, dpi=dpi, use_ocr_for_scanned=True,
                                target_lang=target_lang, translator_fn=None, progress_cb=None,
                                font_override=font_override, table_layout=table_layout,
                                include_drawings=include_drawings,
                                custom_page_size_pt=custom_page_size_pt,
                                trace_mode=not trace_mode,
                            )
                else:
                    all_elements, page_meta, report = process_pdf_full(
                        pdf_bytes, dpi=dpi, layout_conf=layout_conf, device=device,
                        run_tables=run_tables, progress_cb=progress_cb,
                    )
                    docx_bytes = build_docx(all_elements, target_lang=target_lang, translator_fn=None,
                                             report=report, font_override=font_override,
                                             page_dims_pt=page_dims_pt,
                                             custom_page_size_pt=custom_page_size_pt,
                                             pdf_bytes=pdf_bytes, trace_mode=trace_mode)
                    progress_bar.progress(1.0, text="Done")
                    scanned_pages = sum(1 for m in page_meta if m["scanned"])
                    total_tables = sum(m["n_tables"] for m in page_meta)
                    st.success(
                        f"Done! {len(page_meta)} page(s) processed "
                        f"({scanned_pages} via OCR, {len(page_meta) - scanned_pages} native text, "
                        f"{total_tables} table(s) extracted)."
                    )
                    if also_generate_both:
                        # Reuses the already-extracted all_elements -- cheap,
                        # no re-run of layout detection / OCR / table extraction.
                        docx_bytes_alt = build_docx(all_elements, target_lang=target_lang, translator_fn=None,
                                                     report=list(report), font_override=font_override,
                                                     page_dims_pt=page_dims_pt,
                                                     custom_page_size_pt=custom_page_size_pt,
                                                     pdf_bytes=pdf_bytes, trace_mode=not trace_mode)

                base_name = os.path.splitext(uploaded.name)[0]
                primary_suffix = "_trace" if trace_mode else "_editable"
                dl_col1, dl_col2 = st.columns(2) if docx_bytes_alt is not None else (st, None)

                dl_col1.download_button(
                    label="⬇️ Download " + ("Trace Format DOCX" if trace_mode else "Normal DOCX"),
                    data=docx_bytes,
                    file_name=f"{base_name}{primary_suffix}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                if docx_bytes_alt is not None:
                    alt_suffix = "_editable" if trace_mode else "_trace"
                    dl_col2.download_button(
                        label="⬇️ Download " + ("Normal DOCX" if trace_mode else "Trace Format DOCX"),
                        data=docx_bytes_alt,
                        file_name=f"{base_name}{alt_suffix}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

                if report:
                    unique_report = list(dict.fromkeys(report))
                    with st.expander(f"⚠️ Processing report -- {len(unique_report)} item(s) degraded gracefully"):
                        st.caption(
                            "Nothing was silently dropped. Anything below couldn't be converted with full "
                            "fidelity and was handled with a documented fallback instead (this same list is "
                            "also appended as the last page of the DOCX)."
                        )
                        for line in unique_report:
                            st.markdown(f"- {line}")
                else:
                    st.caption("✅ No degraded elements -- every element converted with full fidelity.")

                # ------------------------------------------------------- #
                # 32. Debug JSON export -- inspectable without opening Word.
                # ------------------------------------------------------- #
                if page_meta:
                    debug_entries = [m["debug_entry"] for m in page_meta if m.get("debug_entry")]
                    if debug_entries:
                        debug_json_str = build_debug_json(debug_entries)
                        st.download_button(
                            label="🐞 Download debug JSON (regions / columns / tables / images / overlaps)",
                            data=debug_json_str,
                            file_name=f"{base_name}_debug.json",
                            mime="application/json",
                        )
                        total_overlaps = sum(len(m.get("overlaps", [])) for m in page_meta)
                        if total_overlaps:
                            st.warning(f"⚠️ {total_overlaps} overlap/overflow finding(s) detected across "
                                       f"the document -- see the processing report above and the debug JSON "
                                       f"for exact locations.")

                # ------------------------------------------------------- #
                # 25/26. Render -> compare -> report validation loop.
                # This is opt-in (it shells out to LibreOffice and can be
                # slow) and NEVER claims pixel-perfect fidelity -- it only
                # reports measured deltas between the original PDF and the
                # DOCX rendered back to PDF.
                # ------------------------------------------------------- #
                if st.checkbox("Run visual validation (render DOCX -> PDF and compare, requires LibreOffice)",
                               value=False, key="run_validation"):
                    with st.spinner("Rendering the generated DOCX back to PDF and comparing with the source..."):
                        val = validate_docx_against_pdf(pdf_bytes, docx_bytes)
                    if not val["performed"]:
                        st.info(f"Validation skipped: {val['reason']}")
                    else:
                        if not val["page_count_match"]:
                            st.error("Page count mismatch between original PDF and generated DOCX.")
                        if val["warnings"]:
                            with st.expander(f"⚠️ Validation findings -- {len(val['warnings'])} item(s)", expanded=True):
                                for w in val["warnings"]:
                                    st.markdown(f"- {w}")
                        else:
                            st.success("✅ Validation found no page-size or major content-coverage discrepancies.")
                        if val.get("visual_similarity_avg") is not None:
                            st.metric("Average visual similarity", f"{val.get('visual_similarity_avg', 0):.1f}/100")
                        st.caption(
                            "Validation now compares page size, text coverage and a rendered grayscale visual-similarity "
                            "score. It is a practical QA signal, not a pixel-perfect guarantee because Word/LibreOffice "
                            "font rasterization can differ from the source PDF."
                        )

                if show_boxing_debug and page_meta:
                    st.markdown("---")
                    st.subheader("🔍 Boxing Hint Debugger")
                    has_any_boxes = any(len(m.get("hint_boxes", [])) > 0 for m in page_meta)
                    if not has_any_boxes:
                        st.info("No boxing hint rectangles detected on any page of this document.")

                    for pinfo in page_meta:
                        p_num = pinfo.get("page_num", 1)
                        p_idx = pinfo.get("page_idx", p_num - 1)
                        hint_boxes = pinfo.get("hint_boxes", [])
                        boxed_groups = pinfo.get("boxed_groups", [])
                        leftover = pinfo.get("leftover_lines", [])

                        with st.expander(f"Page {p_num} (Detected {len(hint_boxes)} boxing hint(s))", expanded=(len(hint_boxes) > 0)):
                            if not hint_boxes:
                                st.info("No boxing hint rectangles detected on this page.")
                            else:
                                st.markdown(f"**Detected {len(hint_boxes)} Boxing Hint(s):**")
                                for bi, hb in enumerate(hint_boxes):
                                    st.markdown(
                                        f"- **Box #{bi+1}** | `bbox`: `({hb['bbox'][0]:.1f}, {hb['bbox'][1]:.1f}, {hb['bbox'][2]:.1f}, {hb['bbox'][3]:.1f})` "
                                        f"| `color`: `#{hb.get('color_hex', 'N/A')}`"
                                    )

                                debug_img = render_boxing_debug_page(pdf_bytes, p_idx, hint_boxes, dpi=150)
                                st.image(debug_img, caption=f"Page {p_num} - Detected Boxing Hints", use_container_width=True)

                                st.markdown("#### Text inside detected boxes:")
                                if boxed_groups:
                                    for bi, (hb, blines) in enumerate(boxed_groups):
                                        st.markdown(f"**BOX #{bi+1}**")
                                        st.caption(f"bbox: `({hb['bbox'][0]:.1f}, {hb['bbox'][1]:.1f}, {hb['bbox'][2]:.1f}, {hb['bbox'][3]:.1f})` | color: `#{hb.get('color_hex', 'N/A')}`")
                                        st.markdown("**Text inside:**")
                                        for line in blines:
                                            l_txt = " ".join(s["text"] for s in line["spans"]).strip()
                                            st.text(f"    {l_txt}")
                                else:
                                    st.caption("No lines fell inside the detected hint box(es).")

                            st.markdown(f"#### Leftover lines not in any hint box ({len(leftover)}):")
                            if leftover:
                                leftover_texts = [" ".join(s["text"] for s in ln["spans"]).strip() for ln in leftover]
                                st.text_area(f"Page {p_num} Leftover Text Lines", value="\n".join(leftover_texts), height=150, key=f"leftover_{p_idx}")
                            else:
                                st.text("    None")
            except Exception as e:
                msg = str(e)
                if "dependency error occurred during pipeline creation" in msg:
                    st.error(
                        "PaddleOCR's pipeline is missing a required extra. Run this in "
                        "your terminal, then restart Streamlit:\n\n"
                        '```\npip install "paddlex[ocr]"\n```'
                    )
                else:
                    st.error(f"Conversion failed: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()