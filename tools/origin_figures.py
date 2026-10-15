"""Origin-style figures at their final physical size, with verified PDF/SVG export.

The numerical style owner is workflows/figures/origin-style.json. Adapted from
the approved C-G figure implementation at 91f2f7c; no project data is imported.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib as mpl
from matplotlib import font_manager
from cycler import cycler

STYLE_PATH = Path(__file__).resolve().parents[1] / "workflows/figures/origin-style.json"
STYLE = json.loads(STYLE_PATH.read_text(encoding="utf-8"))
BLACK, RED, BLUE = STYLE["colors"]


def apply_style() -> None:
    """Install the sole default; a missing Times font is an actionable error."""
    try:
        font_manager.findfont(STYLE["font_family"], fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError("Times New Roman is required; install the font before rendering.") from exc
    mpl.rcParams.update(STYLE["rc"])
    mpl.rcParams["axes.prop_cycle"] = (
        cycler(color=STYLE["colors"])
        + cycler(linestyle=STYLE["linestyles"][:3])
    )


def figure(width_mm: float, height_mm: float, **kwargs):
    """Create a white canvas in millimetres; add content-specific axes yourself."""
    from matplotlib import pyplot as plt
    if not all(math.isfinite(v) and v > 0 for v in (width_mm, height_mm)):
        raise ValueError("Figure dimensions must be positive.")
    apply_style()
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), **kwargs)


def style_axes(ax, *, minor: bool = True, categorical: str = ""):
    """Box numeric axes; preserve log locators and omit categorical minor ticks.

    Photographs and schematics normally use ax.set_axis_off() instead. Fields
    keep physical coordinates, aspect, scientific colormap and a labelled bar.
    """
    from matplotlib.ticker import AutoMinorLocator, NullLocator
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=STYLE["rc"]["axes.linewidth"], edgecolor=BLACK)
    ax.set_facecolor("white")
    ax.grid(False, which="both")
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=STYLE["rc"]["xtick.major.size"],
                   width=STYLE["rc"]["xtick.major.width"], colors=BLACK, pad=3)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=STYLE["rc"]["xtick.minor.size"],
                   width=STYLE["rc"]["xtick.minor.width"], colors=BLACK)
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        if not minor or name in categorical:
            axis.set_minor_locator(NullLocator())
        elif axis.get_scale() == "linear":
            axis.set_minor_locator(AutoMinorLocator(2))
    return ax


def series(index: int, *, role: str, replicate: int = 0) -> dict:
    """Return explicit experiment/model encoding, independent of plot order.

    This only chooses artist properties. It never filters, samples or changes
    data. Use markevery solely to space symbols on an otherwise complete line.
    """
    if role not in {"measured", "model"}:
        raise ValueError("role must be measured or model")
    color = STYLE["colors"][index % len(STYLE["colors"])]
    marker = STYLE["markers"][index % len(STYLE["markers"])]
    return {
        "color": color,
        "linestyle": "none" if role == "measured" else STYLE["linestyles"][(index + replicate) % 4],
        "marker": marker if role == "measured" else "none",
        "markerfacecolor": color if replicate % 2 == 0 else "white",
        "markeredgecolor": color,
        "markeredgewidth": .7,
        "markersize": STYLE["rc"]["lines.markersize"],
        "linewidth": STYLE["rc"]["lines.linewidth"],
    }


def save(fig, pdf_path, *, size_mm, spec=None, journal=None, preview_path=None) -> dict:
    """Validate the live artists and actual exports before replacing any output.

    Fixed media size is deliberate: bbox_inches='tight' changes the landing
    width and hides layout errors. Resolve crowding through layout, then export.
    The QA receipt binds both vector outputs; it is evidence, not visual review.
    """
    from tools.figure_visual_qa import audit_figure, audit_output
    pdf = Path(pdf_path).resolve()
    if pdf.suffix.lower() != ".pdf":
        raise ValueError("The publication output must have a .pdf suffix.")
    expected = tuple(float(v) for v in size_mm)
    if len(expected) != 2 or not all(math.isfinite(v) and v > 0 for v in expected):
        raise ValueError("size_mm must contain positive width and height.")
    preview = Path(preview_path).resolve() if preview_path is not None else None
    if preview is not None and not preview.is_relative_to(Path("H:/view-cache").resolve()):
        raise ValueError("Temporary figure previews belong under H:/view-cache/.")
    actual = tuple(float(v) * 25.4 for v in fig.get_size_inches())
    if any(abs(a - b) > STYLE["size_tolerance_mm"] for a, b in zip(actual, expected)):
        raise ValueError(f"Canvas {actual} mm differs from the locked size {expected} mm.")
    live = audit_figure(fig, spec=spec)
    if not live.ok:
        raise RuntimeError(json.dumps(live.to_dict(), ensure_ascii=False))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".origin-export-", dir=pdf.parent) as scratch:
        staged = Path(scratch) / pdf.name
        # Pin export semantics even if another library modified global rcParams.
        with mpl.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none",
                             "savefig.bbox": None, "savefig.transparent": False}):
            fig.savefig(staged, bbox_inches=None, facecolor="white", transparent=False,
                        metadata={"Creator": None, "Producer": None, "CreationDate": None})
            fig.savefig(staged.with_suffix(".svg"), bbox_inches=None, facecolor="white",
                        transparent=False, metadata={"Date": None, "Creator": None})
        outputs = {}
        for ext in ("pdf", "svg"):
            path = staged.with_suffix("." + ext)
            result = audit_output(path, size_mm=expected, journal=journal)
            if not result.ok:
                raise RuntimeError(json.dumps(result.to_dict(), ensure_ascii=False))
            outputs[ext] = {"file": pdf.with_suffix("." + ext).name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "audit": result.to_dict()}
        report = {"ok": True, "style_id": STYLE["style_id"], "size_mm": expected,
                  "live": live.to_dict(), "outputs": outputs}
        staged.with_suffix(".qa.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for ext in ("pdf", "svg", "qa.json"):
            os.replace(staged.with_suffix("." + ext), pdf.with_suffix("." + ext))
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=180, bbox_inches=None, facecolor="white")
    return report
