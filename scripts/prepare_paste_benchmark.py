"""Extract Dunant et al. (2020) Figures 5 and 10 from their PDF polylines.

Requires PyMuPDF for source extraction. The runtime material model only needs
the resulting CSV files and NumPy. No instrument precision is implied by the
coordinate precision retained here.
"""

from pathlib import Path
import argparse
import csv
import json
from urllib.request import urlopen

import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references/papers/R033_dunant_2020/source.pdf"
OUTPUT = ROOT / "data/public_paste_benchmark"


def polyline(draw: dict) -> np.ndarray:
    points = [tuple(draw["items"][0][1])]
    for item in draw["items"]:
        if item[0] == "l":
            points.append(tuple(item[2]))
        elif item[0] == "c":
            controls = np.array([tuple(p) for p in item[1:]])
            u = np.linspace(0, 1, 17)[1:, None]
            curve = ((1-u)**3 * controls[0] + 3*(1-u)**2*u * controls[1]
                     + 3*(1-u)*u**2 * controls[2] + u**3 * controls[3])
            points.extend(curve)
        else:
            raise ValueError(f"Unexpected source path item: {item[0]}")
    points = np.array(points)
    if points[0, 0] > points[-1, 0]:
        points = points[::-1]
    backward = np.flatnonzero(np.diff(points[:, 0]) < 0)
    if backward.size:
        # The nearly vertical initial calorimetry spikes in W32/W40
        # double back horizontally in the published artwork. Retain the
        # subsequent single-valued path, rather than reorder those points
        # into a fictitious time history. No missing initial heat is imputed.
        print(f"Omit unresolved initial path through PDF x={points[backward[-1], 0]:.4f}")
        points = points[backward[-1] + 1:]
    # A vertical plotted segment has no resolvable duration. Collapse its
    # endpoints to their mean ordinate, keeping the original x coordinate.
    x, inverse = np.unique(points[:, 0], return_inverse=True)
    y = np.bincount(inverse, weights=points[:, 1]) / np.bincount(inverse)
    return np.column_stack((x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="Fetch the accepted manuscript from Cambridge before extraction")
    args = parser.parse_args()
    if args.download:
        metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
        with urlopen(metadata["source_pdf_url"], timeout=60) as response:
            content = response.read()
        if not content.startswith(b"%PDF-"):
            raise ValueError("The source download is not a PDF")
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        SOURCE.write_bytes(content)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with fitz.open(SOURCE) as pdf:
        q_drawings = pdf[17].get_drawings()
        e_drawings = pdf[21].get_drawings()
        # Tick endpoints in PDF points, read from the source drawing paths.
        # Using the full ordinate/abscissa span avoids fitting the small
        # placement irregularities of intermediate printed tick marks.
        figures = [
            ("heat_flow.csv", "heat_rate_w_kg_cement", q_drawings,
             {"W32": 1, "W40": 2, "W48": 3},
             (194.4, 382.9, 60.0), (235.45074462890625, 86.00079345703125, 6.0)),
            ("modulus.csv", "modulus_pa", e_drawings,
             {"W32": 9, "W40": 10, "W48": 11},
             (195.0, 372.5, 48.0), (496.2507629394531, 339.5007629394531, 25e9)),
        ]
        for filename, quantity, drawings, curves, x_axis, y_axis in figures:
            with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["mix_id", "age_h", quantity])
                for mix_id, index in curves.items():
                    points = polyline(drawings[index])
                    x0, x1, xmax = x_axis
                    y0, y1, ymax = y_axis
                    age = (points[:, 0] - x0) * xmax / (x1 - x0)
                    value = (y0 - points[:, 1]) * ymax / (y0 - y1)
                    # Clip a segment crossing the zero tick to the plot's
                    # physical time origin, retaining its interpolated value.
                    # Do not shift the remaining curve or force E(0) to zero.
                    if age[0] < 0 < age[-1]:
                        zero_value = np.interp(0, age, value)
                        keep = age > 0
                        age = np.r_[0.0, age[keep]]
                        value = np.r_[zero_value, value[keep]]
                    for t, v in zip(age, value):
                        writer.writerow([mix_id, f"{t:.8f}", f"{v:.10g}"])
                    print(f"{filename} {mix_id}: {len(age)} vertices, "
                          f"{age[0]:.3f} to {age[-1]:.3f} h")


if __name__ == "__main__":
    main()
