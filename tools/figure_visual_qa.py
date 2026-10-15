"""Render and output verification for the shared GeoSkills Origin figure style.

Live artists are checked for typography, axes, clipping, overlap, glyphs and
colorbars. PDF/SVG outputs are reopened to check physical size, actual fonts
and editable text. Chart family, scientific encodings and composition still
need figure-only visual inspection against the complete bound data.

Two modes:

* ``audit_figure(fig, ...)`` — the authoritative mode.  Introspects a live
  matplotlib ``Figure`` (call it from inside the render code, just before
  ``savefig``).  Detects a draw that fails outright, missing glyphs / tofu, text
  clipped past the figure edge, overlapping tick / axis / legend text, a
  color-mapped field with no colorbar (the reader cannot decode the colors at
  all), and a panel count that disagrees with the declared ``figure_spec``.

* ``audit_output(path, size_mm=(width, height))`` — actual PDF/SVG verification,
  or print resolution for raster outputs. A raster alone proves neither font
  identity nor editable text and cannot satisfy the vector delivery contract.

Dependencies: matplotlib, Pillow, PyMuPDF; vector inspection fails closed when
the required PDF reader is unavailable.
The module never selects an interactive backend at import time, so importing it
from render code is side-effect free; the CLI forces the Agg backend so a
standalone run never opens a window.

CLI::

    python tools/figure_visual_qa.py <pdf|svg|png> --width-mm 90 --height-mm 70 [--json]

Exit 0 when there are no error-severity findings (with ``--strict``, also no
warnings); exit 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.cm as mcm
import matplotlib.text as mtext

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JOURNAL_SPECS_PATH = _REPO_ROOT / "workflows" / "figures" / "journal_specs.json"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Min intersection / smaller-box area before two text boxes count as overlapping.
OVERLAP_AREA_FRACTION = 0.10

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass
class Finding:
    """One visual-QA observation."""

    check: str
    severity: str
    message: str
    location: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QAResult:
    """Structured result of a visual-QA pass."""

    findings: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [f for f in self.findings if f.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    @property
    def ok(self) -> bool:
        """True when no error-severity finding fired (warnings/info do not fail)."""
        return not self.errors

    def add(self, check: str, severity: str, message: str, location: str = "") -> None:
        self.findings.append(Finding(check, severity, message, location))

    def extend(self, findings: list) -> None:
        self.findings.extend(findings)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_error": len(self.errors),
            "n_warning": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load_journal_specs() -> dict:
    try:
        return json.loads(_JOURNAL_SPECS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - defensive
            pass


class _DrawCapture:
    """Capture matplotlib's missing-glyph diagnostics during a draw.

    matplotlib emits the missing-glyph notice through the ``matplotlib`` logger
    on some versions and through ``warnings`` on others, so capture both.
    """

    def __enter__(self) -> "_DrawCapture":
        self._handler = _ListHandler()
        self._logger = logging.getLogger("matplotlib")
        self._prev_level = self._logger.level
        if self._prev_level == logging.NOTSET or self._prev_level > logging.WARNING:
            self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(self._handler)
        self._wcatch = warnings.catch_warnings(record=True)
        self._wlist = self._wcatch.__enter__()
        warnings.simplefilter("always")
        self.messages: list = []
        return self

    def __exit__(self, *exc) -> bool:
        self._wcatch.__exit__(*exc)
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)
        self.messages = list(self._handler.messages) + [str(w.message) for w in self._wlist]
        return False


def _get_renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        return FigureCanvasAgg(fig).get_renderer()


def _bbox(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer=renderer)
    except (RuntimeError, ValueError, TypeError):
        return None
    if bb is None or bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def _overlap_area(b1, b2) -> float:
    x0 = max(b1.x0, b2.x0)
    x1 = min(b1.x1, b2.x1)
    y0 = max(b1.y0, b2.y0)
    y1 = min(b1.y1, b2.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _mappables(fig) -> list:
    """Color-mapped artists that actually map data to color (have an array).

    ``ScalarMappable`` is a mixin, not an ``Artist`` subclass, so ``findobj``
    cannot match it by type — use an isinstance predicate.
    """
    out = []
    for m in fig.findobj(lambda a: isinstance(a, mcm.ScalarMappable)):
        host = getattr(m, "axes", None)
        # Skip a colorbar's own internal gradient mesh — it lives on the
        # '<colorbar>' axes and would otherwise read as an uncolorbarred field.
        if host is not None and (host.get_label() == "<colorbar>" or getattr(host, "_colorbar", None) is not None):
            continue
        try:
            arr = m.get_array()
        except Exception:  # pragma: no cover - defensive
            arr = None
        if arr is None:
            continue
        # A true-color RGB(A) image (photograph, pre-rendered panel) carries
        # no colormap; there is no scale a colorbar could decode.
        if getattr(arr, "ndim", 0) == 3:
            continue
        out.append(m)
    return out


# --------------------------------------------------------------------------- #
# Individual checks (mode a, live Figure)
# --------------------------------------------------------------------------- #
def _check_missing_glyph(messages: list) -> list:
    out, seen = [], set()
    for m in messages:
        ml = m.lower()
        if "missing from" in ml and ("font" in ml or "glyph" in ml):
            if m in seen:
                continue
            seen.add(m)
            out.append(Finding("missing_glyph", SEVERITY_ERROR,
                                f"A glyph the active font cannot render (tofu): {m}", "figure"))
    return out


def _visible_text(fig):
    """Exclude locator ticks outside view limits; they exist but are not drawn."""
    hidden = set()
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo - 1e-10 <= tick.get_loc() <= hi + 1e-10:
                    hidden.update((id(tick.label1), id(tick.label2)))
    return [t for t in fig.findobj(mtext.Text)
            if id(t) not in hidden and t.get_visible() and t.get_text().strip()]


def _check_clipping(fig, renderer) -> list:
    out = []
    fig_bb = fig.bbox
    tol = 0.5
    for t in _visible_text(fig):
        s = t.get_text()
        if not t.get_visible() or not s or not s.strip():
            continue
        bb = _bbox(t, renderer)
        if bb is None:
            continue
        if (bb.x0 < fig_bb.x0 - tol or bb.x1 > fig_bb.x1 + tol
                or bb.y0 < fig_bb.y0 - tol or bb.y1 > fig_bb.y1 + tol):
            out.append(Finding("text_clipping", SEVERITY_ERROR,
                                f"Text extends past the figure edge (cut off): {s[:40]!r}", "figure"))
    return out


def _check_origin_style(fig) -> list:
    from matplotlib import colors, font_manager
    from tools.origin_figures import STYLE
    out = []
    for t in _visible_text(fig):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < STYLE["font_floor_pt"] - 1e-6:
            out.append(Finding("font_size", SEVERITY_ERROR,
                               f"Text below {STYLE['font_floor_pt']} pt at final size: {t.get_text()[:40]!r}"))
        try:
            font = font_manager.get_font(font_manager.findfont(t.get_fontproperties(), fallback_to_default=False))
            name = font.family_name.replace(" ", "")
            if not any(name.startswith(p) for p in STYLE["allowed_font_prefixes"]):
                out.append(Finding("font_family", SEVERITY_ERROR,
                                   f"Actual font {font.family_name!r} is outside the Times New Roman/STIX style."))
        except (ValueError, RuntimeError) as exc:
            out.append(Finding("font_family", SEVERITY_ERROR, f"Cannot resolve the selected font: {exc}"))
        if t.get_usetex():
            out.append(Finding("editable_text", SEVERITY_ERROR,
                               "Use STIX mathtext for editable SVG text; usetex may outline labels."))
        elif t.get_parse_math() and "$" in t.get_text() and t.get_math_fontfamily() != STYLE["math_fontset"]:
            out.append(Finding("math_font", SEVERITY_ERROR, "Math labels must use the coordinated STIX fontset."))
    if colors.to_rgba(fig.get_facecolor()) != colors.to_rgba("white"):
        out.append(Finding("background", SEVERITY_ERROR, "The figure canvas must be opaque white."))
    for i, ax in enumerate(fig.get_axes()):
        # Colorbars, 3D fields, photographs and diagram canvases have their own
        # geometry; a four-sided Cartesian frame would misrepresent them.
        if (not ax.axison or ax.get_label() == "<colorbar>"
                or getattr(ax, "_colorbar", None) is not None or ax.name != "rectilinear"):
            continue
        if any(not s.get_visible() or colors.to_rgba(s.get_edgecolor()) != colors.to_rgba("black")
               for s in ax.spines.values()):
            out.append(Finding("axis_frame", SEVERITY_ERROR, "Numeric axes require four black sides.", f"axes:{i}"))
        allowed = {colors.to_rgb(c) for c in STYLE["colors"]}
        if colors.to_rgba(ax.get_facecolor()) != colors.to_rgba("white"):
            out.append(Finding("background", SEVERITY_ERROR, "Numeric axes require a white background.", f"axes:{i}"))
        if any(line.get_visible() for line in [*ax.get_xgridlines(), *ax.get_ygridlines()]):
            out.append(Finding("axis_grid", SEVERITY_ERROR, "Remove inherited plot-theme grids.", f"axes:{i}"))
        def allowed_ink(value):
            rgb = colors.to_rgb(value)
            return rgb in allowed or max(rgb) - min(rgb) < 1e-6
        for line in ax.lines:
            if line.get_visible() and not allowed_ink(line.get_color()):
                out.append(Finding("series_color", SEVERITY_ERROR,
                                   "Numeric plot lines use the shared black/red/blue cycle or neutral reference ink.", f"axes:{i}"))
        for artist in [*ax.collections, *ax.patches]:
            if not artist.get_visible():
                continue
            if isinstance(artist, mcm.ScalarMappable) and artist.get_array() is not None:
                continue  # scientific continuous color encoding, checked by colorbar QA
            inks = []
            for getter in ("get_facecolor", "get_edgecolor"):
                value = getattr(artist, getter, lambda: [])()
                if len(value):
                    inks.extend([value] if isinstance(value[0], (float, int)) else value)
            if any(ink[3] > 0 and not allowed_ink(ink) for ink in inks):
                out.append(Finding("series_color", SEVERITY_ERROR,
                                   "Categorical markers/fills use Origin series colors or neutral ink.", f"axes:{i}"))
        for axis in (ax.xaxis, ax.yaxis):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if tick.get_visible() and tick.get_tickdir() != "in":
                    out.append(Finding("tick_direction", SEVERITY_ERROR,
                                       "Major and minor ticks must point inward.", f"axes:{i}"))
                    break
    return out


def _check_overlaps(fig, renderer) -> list:
    out = []
    for ax in fig.get_axes():
        for getter, axis_name in ((ax.get_xticklabels, "x"), (ax.get_yticklabels, "y")):
            boxes = []
            for t in getter():
                if not t.get_text().strip():
                    continue
                bb = _bbox(t, renderer)
                if bb is not None:
                    boxes.append((bb, t.get_text()))
            hit = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    b1, s1 = boxes[i]
                    b2, s2 = boxes[j]
                    smaller = min(b1.width * b1.height, b2.width * b2.height)
                    if smaller > 0 and _overlap_area(b1, b2) > OVERLAP_AREA_FRACTION * smaller:
                        out.append(Finding("tick_overlap", SEVERITY_WARNING,
                                           f"{axis_name}-axis tick labels overlap ({s1!r}/{s2!r}); "
                                           "reduce ticks, rotate, or widen the axis.", f"axes:{axis_name}"))
                        hit = True
                        break
                if hit:
                    break
    # legend occluding axis text
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = _bbox(leg, renderer)
        if lb is None:
            continue
        targets = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label, ax.title]
        for t in targets:
            if not t.get_text().strip():
                continue
            tb = _bbox(t, renderer)
            if tb is None:
                continue
            area = tb.width * tb.height
            if area > 0 and _overlap_area(lb, tb) > OVERLAP_AREA_FRACTION * area:
                out.append(Finding("legend_occlusion", SEVERITY_WARNING,
                                   f"Legend overlaps axis text {t.get_text()[:30]!r}; "
                                   "move it outside the data area (bbox_to_anchor).", "legend"))
                break
    # Check actual data paths/positions, including figure-level legends placed
    # in a margin. A line can cross a legend even when both endpoints are clear.
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.get_axes() if ax.get_legend())]
    for legend in legends:
        lb = _bbox(legend, renderer)
        if lb is None or not legend.get_visible():
            continue
        overlaps = False
        for ax in fig.get_axes():
            if not lb.overlaps(ax.bbox):
                continue
            for line in ax.lines:
                if line.get_visible() and line.get_transform().transform_path(line.get_path()).intersects_bbox(lb, filled=False):
                    overlaps = True
            for collection in ax.collections:
                # PathCollection offsets describe scatter centers. Other
                # collections (field meshes, error bars) need visual review.
                if type(collection).__name__ != "PathCollection":
                    continue
                pts = collection.get_offset_transform().transform(collection.get_offsets())
                if any(lb.contains(float(x), float(y)) for x, y in pts):
                    overlaps = True
        if overlaps:
            out.append(Finding("legend_data_overlap", SEVERITY_WARNING,
                               "Legend covers data paths/markers; move it to empty space or reserved margins.", "legend"))
    return out


def _check_colorbar(fig) -> list:
    out = []
    for m in _mappables(fig):
        if getattr(m, "colorbar", None) is None:
            out.append(Finding("missing_colorbar", SEVERITY_ERROR,
                               "A color-mapped field/scatter has no colorbar; the colors carry values "
                               "the reader has no way to decode. Add a colorbar with a units label.",
                               type(m).__name__))
    return out


def _check_spec_consistency(fig, spec) -> list:
    out = []
    if not isinstance(spec, dict):
        return out
    fspec = spec.get("figure_spec", spec)
    panels = fspec.get("panels") if isinstance(fspec, dict) else None
    if not isinstance(panels, list) or not panels:
        return out
    data_axes = [a for a in fig.get_axes() if a.get_label() != "<colorbar>"
                 and getattr(a, "_colorbar", None) is None]
    if len(data_axes) < len(panels):
        out.append(Finding("panel_count_mismatch", SEVERITY_WARNING,
                           f"figure_spec declares {len(panels)} panel(s) but the figure has "
                           f"{len(data_axes)} data axes.", "figure"))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def audit_figure(fig, *, spec=None) -> QAResult:
    """Audit a live matplotlib ``Figure`` (authoritative mode).

    Call just before ``savefig``.  ``spec`` is the optional ``figure_spec`` dict
    used for a panel-count cross-check.
    """
    result = QAResult()
    with _DrawCapture() as cap:
        try:
            fig.canvas.draw()
        except Exception as exc:
            result.add("draw_failed", SEVERITY_ERROR, f"Figure failed to render: {exc}", "figure")
            return result
    renderer = _get_renderer(fig)
    result.extend(_check_missing_glyph(cap.messages))
    result.extend(_check_clipping(fig, renderer))
    result.extend(_check_overlaps(fig, renderer))
    result.extend(_check_origin_style(fig))
    result.extend(_check_colorbar(fig))
    result.extend(_check_spec_consistency(fig, spec))
    return result


def audit_output(png_path, spec_path=None, *, journal: str | None = None, size_mm=None) -> QAResult:
    """Reopen actual outputs; a raster audit covers resolution only."""
    result = QAResult()
    png = Path(png_path)
    if not png.is_file():
        result.add("file_missing", SEVERITY_ERROR, f"Output image not found: {png}", str(png))
        return result

    if png.suffix.lower() in {".pdf", ".svg"}:
        from tools.figure_artifact_qa import audit_vector
        return audit_vector(png, size_mm=size_mm)

    if spec_path:
        try:
            json.loads(Path(spec_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.add("spec_unreadable", SEVERITY_WARNING, f"Could not read spec {spec_path}: {exc}", str(spec_path))

    specs = _load_journal_specs()
    jspec = specs.get((journal or "").lower(), {}) if journal else {}
    min_dpi = float(jspec.get("min_dpi", specs.get("_default", {}).get("min_dpi", 300)))

    try:
        from PIL import Image

        with Image.open(png) as im:
            w_px = im.size[0]
            dpi = im.info.get("dpi")
        if dpi:
            xdpi = float(dpi[0]) if isinstance(dpi, (tuple, list)) else float(dpi)
            # matplotlib stores DPI as integer pixels-per-metre, so a 300-dpi
            # save reads back as ~299.9994 — round before comparing.
            if xdpi and round(xdpi) < min_dpi:
                result.add("low_dpi", SEVERITY_ERROR,
                           f"Image DPI {xdpi:.0f} is below the {min_dpi:.0f} floor; the raster resolves "
                           f"below print legibility. Re-export at >= {min_dpi:.0f} DPI, or use a vector "
                           "format.", str(png))
            if xdpi:
                width_mm = w_px / xdpi * 25.4
                if width_mm < 40:
                    result.add("tiny_width", SEVERITY_WARNING,
                               f"Rendered width {width_mm:.0f}mm is unusually small for a publication figure.",
                               str(png))
        else:
            result.add("dpi_unknown", SEVERITY_WARNING,
                       "PNG carries no DPI metadata; print resolution cannot be verified. Re-export with an "
                       "explicit dpi, or use a vector format.", str(png))
    except ImportError:
        result.add("pil_missing", SEVERITY_INFO,
                   "Pillow not installed; size/DPI check skipped (audit_figure does not need Pillow).", str(png))
    except Exception as exc:  # pragma: no cover - defensive
        result.add("image_unreadable", SEVERITY_WARNING, f"Could not read image metadata: {exc}", str(png))

    result.add("glyph_check_unavailable", SEVERITY_INFO,
               "Glyph/tofu detection is unreliable on a flat PNG; run audit_figure(fig) during render for the "
               "authoritative glyph check.", str(png))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Origin figure PDF/SVG outputs or raster print resolution.")
    parser.add_argument("image", help="Path to the rendered figure (PDF, SVG, PNG or TIFF).")
    parser.add_argument("--width-mm", type=float)
    parser.add_argument("--height-mm", type=float)
    parser.add_argument("--spec", default=None, help="Path to the figure_spec JSON (optional).")
    parser.add_argument("--journal", default=None, help="Journal key for the DPI floor (e.g. nature, ieee).")
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings as well as errors.")
    args = parser.parse_args(argv)

    matplotlib.use("Agg")  # standalone runs never open a window
    if (args.width_mm is None) != (args.height_mm is None):
        parser.error("Pass both --width-mm and --height-mm.")
    size = (args.width_mm, args.height_mm) if args.width_mm is not None else None
    result = audit_output(args.image, args.spec, journal=args.journal, size_mm=size)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        if not result.findings:
            print("PASS: no visual-QA findings.")
        for f in result.findings:
            loc = f" ({f.location})" if f.location else ""
            print(f"[{f.severity.upper()}] {f.check}: {f.message}{loc}")
        print(f"\n{'PASS' if result.ok else 'FAIL'}: "
              f"{len(result.errors)} error(s), {len(result.warnings)} warning(s).")

    failed = (not result.ok) or (args.strict and bool(result.warnings))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
