"""Update loading times for a fixed material along one construction history.

The common clock starts at the specified state of the placed annulus. Each
checkpoint carries the heat and temperature fields reached on that history;
reference equivalent ages describe those fields, independently of the clock.
"""

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import simulate_paste
from grouting_pino.support_response import SectionResponse, first_attainment, loading_event

OUT = ROOT / "results/support_design"


def main():
    section = SectionResponse.load(OUT / "reference_response.npz")
    refs = load_paste_references(ROOT / "data/public_paste_benchmark")
    with (OUT / "radial_histories.csv").open(newline="", encoding="utf-8") as stream:
        fields = [r for r in csv.DictReader(stream)
                  if r["scenario"] == "nominal" and r["case"] == "reference"]
    rows = []
    for mix, ref in refs.items():
        whole = simulate_paste(ref, section.geometry, np.linspace(0, 6, 601), 20, 20)
        history = sorted((r for r in fields if r["mix_id"] == mix), key=lambda r: float(r["time_h"]))
        times = np.array([float(r["time_h"]) for r in history])
        response = np.array([float(r["response_um_kpa"]) for r in history])*1e-9
        for checkpoint in (0.0, 1.0, 2.0):
            # These are checkpoints on the same computed history. Reuse the
            # subsequent radial field responses, retaining the construction clock.
            after = times > checkpoint
            relative = np.r_[0., times[after]-checkpoint]
            continued = np.r_[np.interp(checkpoint, times, response), response[after]]
            hit = first_attainment(relative, continued, 2.2e-9)
            event = loading_event(hit, relative[-1], 4.0, state_time_h=checkpoint)
            i = np.flatnonzero(whole.construction_time_h == checkpoint).item()
            age = ref.age_at_heat(whole.heat_j_kg_cement[i])
            rows.append(dict(
                case="reference", mix_id=mix, target_um_kpa=2.2,
                state_time_h=checkpoint,
                reference_age_min_h=float(age.min()), reference_age_max_h=float(age.max()),
                latest_loading_time_h=4.0, remaining_time_h=event.remaining_time_h,
                earliest_loading_time_h=event.earliest_time_h,
                status=event.status, supported_end_time_h=times[-1]))
    with (OUT / "state_updates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved fixed-material loading updates at construction times 0, 1 and 2 h.")


if __name__ == "__main__":
    main()
