"""Inspect exported vector bytes; source declarations are not output evidence."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import numpy as np

from tools.figure_visual_qa import QAResult, SEVERITY_ERROR, SEVERITY_INFO
from tools.origin_figures import STYLE


def _font_allowed(name):
    normalized = re.sub(r"^[A-Z]{6}\+", "", name).replace(" ", "").strip("'\"")
    return any(normalized.startswith(p) for p in STYLE["allowed_font_prefixes"])


def _size(result, actual, expected):
    result.add("physical_size", SEVERITY_INFO, f"Actual size: {actual[0]:.3f} x {actual[1]:.3f} mm.")
    if expected is None:
        result.add("physical_size", SEVERITY_ERROR,
                   "Bind the intended landing size with width_mm and height_mm; actual size alone is insufficient.")
    elif len(expected) != 2 or min(expected) <= 0 or any(
            abs(a - b) > STYLE["size_tolerance_mm"] for a, b in zip(actual, expected)):
        result.add("physical_size", SEVERITY_ERROR, f"Actual size {actual} mm differs from locked size {expected} mm.")


def _pdf(path, result, size_mm):
    try:
        import fitz
    except ImportError:
        result.add("pdf_reader_missing", SEVERITY_ERROR, "Install PyMuPDF to inspect actual PDF output.")
        return
    with fitz.open(path) as doc:
        if len(doc) != 1:
            result.add("page_count", SEVERITY_ERROR, "One figure must have exactly one PDF page.")
        for page in doc:
            _size(result, (page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72), size_mm)
            if page.cropbox != page.mediabox:
                result.add("crop_box", SEVERITY_ERROR, "PDF CropBox differs from MediaBox; export the fixed canvas without cropping.")
            fonts = page.get_fonts(full=True)
            for font in fonts:
                xref, _, kind, name = font[:4]
                if not _font_allowed(name):
                    result.add("font_family", SEVERITY_ERROR, f"PDF uses unexpected font {name}.")
                if kind == "Type3" or not xref or not doc.extract_font(xref)[3]:
                    result.add("font_embedding", SEVERITY_ERROR, f"PDF font {name} is unembedded or Type3.")
            # Infinite clipping includes text outside the page; default extraction
            # would hide precisely the strings this check is meant to detect.
            spans = [s for b in page.get_text("dict", clip=fitz.INFINITE_RECT())["blocks"]
                     if "lines" in b for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            if not spans:
                result.add("editable_text", SEVERITY_ERROR, "PDF has no editable text; outlined or flattened labels are not accepted.")
            for s in spans:
                floor = STYLE["math_glyph_floor_pt"] if s["font"].startswith("STIX") else STYLE["font_floor_pt"]
                if s["size"] < floor - .05:
                    result.add("font_size", SEVERITY_ERROR,
                               f"PDF text {s['text'][:35]!r} is {s['size']:.2f} pt, below {floor} pt.")
                bb = fitz.Rect(s["bbox"])
                bounds = page.rect + (-.7, -.7, .7, .7)
                if not bounds.contains(bb):
                    result.add("text_clipping", SEVERITY_ERROR, f"PDF text extends outside the canvas: {s['text'][:40]!r}.")
            if not page.get_drawings():
                result.add("vector_geometry", SEVERITY_ERROR,
                           "PDF has no vector geometry; keep plots and annotations vector over genuine raster photos/fields.")
            result.add("pdf_fonts", SEVERITY_INFO, "Embedded fonts: " + ", ".join(sorted({f[3] for f in fonts})))


def _length_mm(value):
    m = re.fullmatch(r"\s*([\d.+eE-]+)(mm|cm|in|pt|px)?\s*", value or "")
    if not m:
        raise ValueError(f"Unsupported SVG physical length: {value!r}")
    return float(m[1]) * {"mm": 1, "cm": 10, "in": 25.4, "pt": 25.4 / 72,
                          "px": 25.4 / 96, None: 25.4 / 96}[m[2]]


def _svg(path, result, size_mm):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    root = ET.parse(path).getroot()
    _size(result, (_length_mm(root.get("width")), _length_mm(root.get("height"))), size_mm)
    view = [float(v) for v in root.get("viewBox", "").replace(",", " ").split()]
    if len(view) != 4 or min(view[2:]) <= 0:
        result.add("svg_viewbox", SEVERITY_ERROR, "SVG needs a valid fixed viewBox.")
        return
    # Matplotlib exports SVG units as points (width/height in pt). Other tools
    # may use px; derive the scale from the actual viewport, never assume it.
    viewport_pt = (_length_mm(root.get("width")) * 72 / 25.4,
                   _length_mm(root.get("height")) * 72 / 25.4)
    sx, sy = viewport_pt[0] / view[2], viewport_pt[1] / view[3]
    aspect = root.get("preserveAspectRatio", "xMidYMid meet")
    if "none" in aspect:
        root_matrix = np.array([[sx, 0, -view[0]*sx], [0, sy, -view[1]*sy], [0, 0, 1]])
    else:
        scale = max(sx, sy) if "slice" in aspect else min(sx, sy)
        dx = (viewport_pt[0] - view[2]*scale) * (0 if "xMin" in aspect else 1 if "xMax" in aspect else .5)
        dy = (viewport_pt[1] - view[3]*scale) * (0 if "YMin" in aspect else 1 if "YMax" in aspect else .5)
        root_matrix = np.array([[scale, 0, dx-view[0]*scale], [0, scale, dy-view[1]*scale], [0, 0, 1]])
    text_count = 0
    families = set()
    def transform_matrix(value):
        matrix = np.eye(3)
        for name, raw in re.findall(r"(matrix|translate|scale|rotate)\s*\(([^)]+)\)", value or ""):
            v = [float(x) for x in re.split(r"[ ,]+", raw.strip())]
            local = np.eye(3)
            if name == "matrix" and len(v) == 6:
                local = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
            elif name == "translate":
                local[:2, 2] = [v[0], v[1] if len(v) > 1 else 0]
            elif name == "scale":
                local[0, 0], local[1, 1] = v[0], v[1] if len(v) > 1 else v[0]
            elif name == "rotate":
                angle = np.deg2rad(v[0])
                local[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                if len(v) == 3:
                    center = np.array(v[1:])
                    local[:2, 2] = center - local[:2, :2] @ center
            matrix = matrix @ local
        return matrix

    def walk(node, inherited, parent_matrix, parent_pos=(0, 0)):
        nonlocal text_count
        matrix = parent_matrix @ transform_matrix(node.get("transform"))
        scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).min())
        # Matplotlib positions each math run explicitly; propagated positions
        # also cover single text nodes used by simple SVG annotation tools.
        def first_number(value, fallback):
            return float(re.split(r"[ ,]+", value.strip())[0]) if value else fallback
        pos = (first_number(node.get("x"), parent_pos[0]), first_number(node.get("y"), parent_pos[1]))
        props = dict(inherited)
        props.update(dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", node.get("style", ""))))
        props.update({k: node.get(k) for k in ("font-family", "font-size") if node.get(k)})
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"text", "tspan"} and node.text and node.text.strip():
            text_count += 1
            shorthand = props.get("font", "")
            family = props.get("font-family", "")
            if not family:
                match = re.search(r"[\d.]+(?:px|pt)\s+(.+)", shorthand)
                family = match[1] if match else ""
            first = family.split(",")[0].strip(" '\"")
            families.add(first)
            if not _font_allowed(first):
                result.add("font_family", SEVERITY_ERROR, f"SVG text uses unexpected or implicit font {family!r}.")
            size = props.get("font-size", "")
            if not size:
                match = re.search(r"([\d.]+(?:px|pt))", shorthand)
                size = match[1] if match else ""
            match = re.fullmatch(r"([\d.]+)(px|pt)?", size)
            if not match:
                result.add("font_size", SEVERITY_ERROR, "SVG text has no verifiable font size.")
            else:
                units = float(match[1]) * (96 / 72 if match[2] == "pt" else 1)
                printed = units * scale
                floor = STYLE["math_glyph_floor_pt"] if first.startswith("STIX") else STYLE["font_floor_pt"]
                if printed < floor - .05:
                    result.add("font_size", SEVERITY_ERROR, f"SVG text is {printed:.2f} pt, below {floor} pt.")
                if _font_allowed(first):
                    prop = FontProperties(family=first, size=units,
                                          style=props.get("font-style", "normal"),
                                          weight=props.get("font-weight", "normal"))
                    width, height, descent = TextToPath().get_text_width_height_descent(node.text, prop, False)
                    anchor = props.get("text-anchor", node.get("text-anchor", "start"))
                    x, y = pos
                    x -= width * {"start": 0, "middle": .5, "end": 1}.get(anchor, 0)
                    box = np.array([[x, y-height+descent, 1], [x+width, y-height+descent, 1],
                                    [x, y+descent, 1], [x+width, y+descent, 1]]) @ matrix.T
                    if (box[:, 0].min() < -.7 or box[:, 1].min() < -.7
                            or box[:, 0].max() > viewport_pt[0] + .7
                            or box[:, 1].max() > viewport_pt[1] + .7):
                        result.add("text_clipping", SEVERITY_ERROR,
                                   f"SVG text extends outside its viewBox: {node.text[:40]!r}.")
        if tag == "image":
            href = node.get("href") or node.get("{http://www.w3.org/1999/xlink}href", "")
            if not href.startswith("data:"):
                result.add("external_image", SEVERITY_ERROR, "SVG image bytes must be embedded, not linked to a local/remote file.")
        for child in node:
            walk(child, props, matrix, pos)
    walk(root, {}, root_matrix)
    if not text_count:
        result.add("editable_text", SEVERITY_ERROR, "SVG has no text nodes; keep labels editable, not outlined.")
    result.add("svg_fonts", SEVERITY_INFO, "Editable SVG text families: " + ", ".join(sorted(families)))
    result.add("svg_review", SEVERITY_INFO,
               "Canvas text bounds and affine scale checked; inspect artwork for local clip paths and data/legend occlusion.")


def audit_vector(path, *, size_mm=None):
    result = QAResult()
    try:
        (_pdf if path.suffix.lower() == ".pdf" else _svg)(path, result, size_mm)
    except Exception as exc:
        result.add("vector_unreadable", SEVERITY_ERROR, f"Cannot inspect {path.name}: {exc}")
    return result
