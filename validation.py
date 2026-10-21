"""
validation.py -- kept deliberately separate from extraction / layout /
docx reconstruction (spec section 24), covering:

  * overlap / overflow detection on the elements about to be written to a
    page, BEFORE the DOCX is saved (spec section 33)
  * a structured per-page debug JSON export (spec section 32)
  * an actual render -> compare -> report validation loop: convert the
    generated DOCX back to PDF (via LibreOffice headless) and compare its
    page geometry / text coverage against the ORIGINAL source PDF (spec
    sections 25 and 26)

None of this file imports Streamlit or touches app6.py's DOCX-writing
internals -- it only consumes plain bbox/text data structures, so it can be
unit tested and reused headlessly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - fitz is a hard dependency elsewhere
    fitz = None

if TYPE_CHECKING:
    import fitz as _fitz_types  # noqa: F401 - only used for the type hint below


# --------------------------------------------------------------------------- #
# 33. Overlap / overflow detection
# --------------------------------------------------------------------------- #

def _bbox_overlap_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return iw * ih


def detect_element_overlaps(elements: List[Dict[str, Any]], page_width_pt: float,
                             page_height_pt: float, overlap_ratio_threshold: float = 0.15
                             ) -> List[Dict[str, Any]]:
    """Pairwise-checks a page's elements (each needs "type" and "bbox") for:
      - text/text, text/table, text/image, table/image overlap beyond a
        small tolerance (elements naturally touch at edges; only report
        overlaps that cover a meaningful fraction of the smaller element)
      - elements that fall (partially or fully) outside the page bounds

    Returns a list of finding dicts:
        {"kind": "overlap"|"overflow", "detail": str,
         "bbox_a": [...], "bbox_b": [...] (overlap only)}
    This never raises and never mutates input -- it only reports, matching
    spec section 33 ("If detected: report the problem.").
    """
    findings: List[Dict[str, Any]] = []

    for el in elements:
        bbox = el.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        if x0 < -1.0 or y0 < -1.0 or x1 > page_width_pt + 1.0 or y1 > page_height_pt + 1.0:
            findings.append({
                "kind": "overflow",
                "detail": f"{el.get('type', 'element')} at ({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) "
                          f"extends outside the page ({page_width_pt:.1f}x{page_height_pt:.1f}pt).",
                "bbox_a": list(bbox),
            })

    n = len(elements)
    for i in range(n):
        a = elements[i]
        a_bbox = a.get("bbox")
        if not a_bbox:
            continue
        a_area = max(1.0, (a_bbox[2] - a_bbox[0]) * (a_bbox[3] - a_bbox[1]))
        for j in range(i + 1, n):
            b = elements[j]
            b_bbox = b.get("bbox")
            if not b_bbox:
                continue
            b_area = max(1.0, (b_bbox[2] - b_bbox[0]) * (b_bbox[3] - b_bbox[1]))
            inter = _bbox_overlap_area(a_bbox, b_bbox)
            if inter <= 0:
                continue
            ratio = inter / min(a_area, b_area)
            if ratio >= overlap_ratio_threshold:
                findings.append({
                    "kind": "overlap",
                    "detail": f"{a.get('type', 'element')} overlaps {b.get('type', 'element')} "
                              f"by {ratio * 100:.0f}% of the smaller element's area.",
                    "bbox_a": list(a_bbox),
                    "bbox_b": list(b_bbox),
                })
    return findings


# --------------------------------------------------------------------------- #
# 32. Debug JSON export
# --------------------------------------------------------------------------- #

def build_debug_json(pages: List[Dict[str, Any]], pretty: bool = True) -> str:
    """Builds the debug JSON representation described in spec section 32.

    `pages` is a list of per-page dicts the caller assembles, one per PDF
    page, each shaped like:
        {
          "page": 1, "width": ..., "height": ...,
          "regions": [{"bbox":[...], "type": "...", "columns":[{"bbox":[...]}]}],
          "tables": [...], "images": [...], "text_blocks": [...],
          "overlaps": [...],  # from detect_element_overlaps, optional
        }
    This function only serializes -- it does not compute layout itself, so
    it stays decoupled from the extraction/layout code.
    """
    return json.dumps({"pages": pages}, indent=2 if pretty else None, default=str)


def page_debug_entry(page_num: int, width_pt: float, height_pt: float,
                      bands: Optional[List[Dict[str, Any]]] = None,
                      tables: Optional[List[Dict[str, Any]]] = None,
                      images: Optional[List[Dict[str, Any]]] = None,
                      text_blocks: Optional[List[Dict[str, Any]]] = None,
                      overlaps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Convenience builder that turns detect_page_regions_recursive()'s band
    output directly into the debug-JSON "regions" shape."""
    regions = []
    for band in (bands or []):
        regions.append({
            "bbox": list(band["bbox"]),
            "type": "single_column" if band.get("column_count", 1) == 1 else f"{band['column_count']}_column",
            "columns": [{"bbox": list(c["bbox"])} for c in band.get("columns", [])],
        })
    return {
        "page": page_num,
        "width": width_pt,
        "height": height_pt,
        "regions": regions,
        "tables": tables or [],
        "images": images or [],
        "text_blocks": text_blocks or [],
        "overlaps": overlaps or [],
    }


# --------------------------------------------------------------------------- #
# 25 / 26. Render -> compare -> report validation loop
# --------------------------------------------------------------------------- #

def _find_soffice() -> Optional[str]:
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def docx_bytes_to_pdf_bytes(docx_bytes: bytes, timeout_sec: int = 120) -> Optional[bytes]:
    """Converts DOCX -> PDF via headless LibreOffice, the same rendering
    path Word documents most reliably reproduce fixed-layout tables and
    absolute positioning through. Returns None (never raises) if
    LibreOffice isn't installed -- callers must treat that as
    "validation skipped", not a pipeline failure."""
    soffice = _find_soffice()
    if not soffice:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.docx")
        with open(in_path, "wb") as f:
            f.write(docx_bytes)
        try:
            subprocess.run(
                [soffice, "--headless", "--norestore", "--convert-to", "pdf", "--outdir", tmp, in_path],
                check=True, timeout=timeout_sec, capture_output=True,
            )
        except Exception:
            return None
        out_path = os.path.join(tmp, "input.pdf")
        if not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            return f.read()


def _page_text_bbox_coverage(page: "_fitz_types.Page") -> float:
    """Fraction of the page area covered by the union of text-line bboxes
    (a cheap, font-independent proxy for 'how much content ended up where'
    -- used for comparison, not pixel-perfect diffing)."""
    rect = page.rect
    page_area = max(1.0, rect.width * rect.height)
    covered = 0.0
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            b = line.get("bbox")
            if not b:
                continue
            covered += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return min(1.0, covered / page_area)



def _page_visual_similarity(a_page, b_page, max_dim=900) -> float:
    """Return a conservative 0..100 visual similarity score.

    The pages are rendered at the same normalized size and compared using
    grayscale mean absolute pixel error. This is intentionally a practical
    QA score, not a claim of pixel identity across Word/LibreOffice fonts.
    """
    try:
        aw, ah = a_page.rect.width, a_page.rect.height
        bw, bh = b_page.rect.width, b_page.rect.height
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0
        scale = min(1.0, max_dim / max(aw, ah))
        mat_a = fitz.Matrix(scale, scale)
        mat_b = fitz.Matrix(scale * aw / bw, scale * ah / bh)
        pa = a_page.get_pixmap(matrix=mat_a, colorspace=fitz.csGRAY, alpha=False)
        pb = b_page.get_pixmap(matrix=mat_b, colorspace=fitz.csGRAY, alpha=False)
        import numpy as np
        aa = np.frombuffer(pa.samples, dtype=np.uint8).reshape(pa.height, pa.width)
        bb = np.frombuffer(pb.samples, dtype=np.uint8).reshape(pb.height, pb.width)
        h=min(aa.shape[0],bb.shape[0]); w=min(aa.shape[1],bb.shape[1])
        aa=aa[:h,:w].astype(np.float32); bb=bb[:h,:w].astype(np.float32)
        mae=float(np.mean(np.abs(aa-bb))) / 255.0
        # Nonlinear mapping: 0 error = 100, 0.25 average normalized error ~= 50.
        score=max(0.0, min(100.0, 100.0 * (1.0 - mae) ** 2))
        return score
    except Exception:
        return 0.0

def validate_docx_against_pdf(original_pdf_bytes: bytes, generated_docx_bytes: bytes,
                               dims_tolerance_pt: float = 3.0) -> Dict[str, Any]:
    """Implements the mandatory validation loop (spec section 25):

        DOCX -> render to PDF -> compare with original PDF pages

    Compares, page by page: page dimensions, page count, and a rough
    text-coverage ratio (as a stand-in for 'how much of the page's content
    survived reconstruction' without requiring pixel-identical fonts).

    Returns:
        {
          "performed": bool,       # False if LibreOffice wasn't available
          "reason": str,           # explanation when not performed
          "page_count_match": bool,
          "pages": [ {page, orig_size, rendered_size, size_match,
                      orig_text_coverage, rendered_text_coverage,
                      coverage_delta, visual_similarity}, ... ],
          "warnings": [str, ...],
        }
    This NEVER claims perfect/pixel-perfect fidelity (spec section 50) --
    it only reports measured deltas.
    """
    if fitz is None:
        return {"performed": False, "reason": "PyMuPDF not available", "pages": []}

    rendered_pdf_bytes = docx_bytes_to_pdf_bytes(generated_docx_bytes)
    if rendered_pdf_bytes is None:
        return {
            "performed": False,
            "reason": "LibreOffice ('soffice') is not installed in this environment, so the "
                      "generated DOCX could not be rendered back to PDF for comparison. Install "
                      "LibreOffice to enable automatic visual validation.",
            "pages": [],
        }

    orig = fitz.open(stream=original_pdf_bytes, filetype="pdf")
    rendered = fitz.open(stream=rendered_pdf_bytes, filetype="pdf")

    warnings: List[str] = []
    page_count_match = len(orig) == len(rendered)
    if not page_count_match:
        warnings.append(f"Page count differs: original has {len(orig)} pages, "
                         f"generated DOCX renders to {len(rendered)} pages.")

    pages_report = []
    n = min(len(orig), len(rendered))
    for i in range(n):
        o_page, r_page = orig[i], rendered[i]
        o_w, o_h = o_page.rect.width, o_page.rect.height
        r_w, r_h = r_page.rect.width, r_page.rect.height
        size_match = abs(o_w - r_w) <= dims_tolerance_pt and abs(o_h - r_h) <= dims_tolerance_pt

        o_cov = _page_text_bbox_coverage(o_page)
        r_cov = _page_text_bbox_coverage(r_page)
        delta = abs(o_cov - r_cov)
        visual_similarity = _page_visual_similarity(o_page, r_page)

        entry = {
            "page": i + 1,
            "orig_size_pt": [round(o_w, 1), round(o_h, 1)],
            "rendered_size_pt": [round(r_w, 1), round(r_h, 1)],
            "size_match": size_match,
            "orig_text_coverage": round(o_cov, 3),
            "rendered_text_coverage": round(r_cov, 3),
            "coverage_delta": round(delta, 3),
            "visual_similarity": round(visual_similarity, 1),
        }
        pages_report.append(entry)

        if not size_match:
            warnings.append(f"Page {i + 1}: size mismatch (orig {o_w:.1f}x{o_h:.1f}pt vs "
                             f"rendered {r_w:.1f}x{r_h:.1f}pt).")
        if visual_similarity < 70.0:
            warnings.append(f"Page {i + 1}: visual similarity score is {visual_similarity:.1f}/100 -- "
                             "inspect this page for position, font, table or spacing differences.")
        if delta > 0.25:
            warnings.append(f"Page {i + 1}: text coverage differs substantially "
                             f"(orig {o_cov:.0%} vs rendered {r_cov:.0%} of page area) -- "
                             f"likely missing or badly repositioned content.")

    orig.close()
    rendered.close()

    return {
        "performed": True,
        "reason": "",
        "page_count_match": page_count_match,
        "pages": pages_report,
        "visual_similarity_avg": round(sum(p["visual_similarity"] for p in pages_report) / len(pages_report), 1) if pages_report else 0.0,
        "warnings": warnings,
    }
