"""Read material reaction requirements at the saved radial-FE loading events.

Reconstruct each thermal history from radial_runs.csv; no FE solve is needed.
Heat increments are relative to the supplied state and integrated over the
initial annular volume, using the same finite-volume weights as heat transfer.
"""

from collections import defaultdict
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import PasteThermalProperties, simulate_paste
from grouting_pino.support_response import SectionResponse
from run_paste_support import thermal_scenarios, write_csv
from run_support_events import load_events

OUT = ROOT / "results/support_design"


def reaction_at_event(history, time_h):
    """Volume-integrated heat released between a supplied state and attainment."""
    if not 0 <= time_h <= history.time_h[-1]:
        raise ValueError("Reaction interval must lie within the thermal history")
    heat = history.heat_j_kg_cement
    delta = np.array([np.interp(time_h, history.time_h, column) - column[0]
                      for column in heat.T])
    volume = history.mesh["paste_volume"]
    mean = float(delta @ volume / volume.sum())
    return dict(mean_heat_increment_j_kg_cement=mean,
                min_heat_increment_j_kg_cement=float(delta.min()),
                max_heat_increment_j_kg_cement=float(delta.max()),
                released_heat_mj_m3=mean * history.volume_properties["cement_kg_m3"] / 1e6)


def main():
    section = SectionResponse.load(OUT / "reference_response.npz")
    refs = load_paste_references(ROOT / "data/public_paste_benchmark")
    events = load_events()
    targets = sorted({r["target_um_kpa"] for r in events})
    props = PasteThermalProperties()
    uniform = []
    for mix, ref in refs.items():
        cement = props.volume_properties(ref.water_cement_ratio)["cement_kg_m3"]
        for target in targets:
            required = section.required_modulus_pa(target * 1e-9)
            q = None if required is None else ref.first_heat_at_modulus(required)
            uniform.append(dict(mix_id=mix, target_um_kpa=target,
                                required_modulus_gpa=required / 1e9 if required is not None else None,
                                reference_age_h=float(ref.age_at_heat(q)) if q is not None else None,
                                heat_increment_j_kg_cement=q,
                                released_heat_mj_m3=cement*q/1e6 if q is not None else None))
    write_csv("uniform_reaction_requirements.csv", uniform)

    keys = ("scenario", "case", "mix_id")
    grouped = defaultdict(list)
    for event in events:
        grouped[tuple(event[k] for k in keys)].append(event)
    with (OUT / "radial_runs.csv").open(encoding="utf-8", newline="") as stream:
        runs = list(csv.DictReader(stream))
    scenarios = {name: (properties, options) for name, properties, options in thermal_scenarios()}
    results = []
    for i, row in enumerate(runs, 1):
        properties, options = scenarios[row["scenario"]]
        inputs = {k: float(row[k]) for k in ("initial_reference_age_h", "initial_paste_temperature_c",
                                           "initial_ring_temperature_c", "soil_temperature_c")}
        history = simulate_paste(refs[row["mix_id"]], section.geometry, np.linspace(0, 12, 1201),
                                 air_temperature_c=20., properties=properties, **inputs, **options)
        for event in grouped[tuple(row[k] for k in keys)]:
            values = (reaction_at_event(history, event["time_h"]) if event["time_h"] is not None
                      else dict.fromkeys(("mean_heat_increment_j_kg_cement", "min_heat_increment_j_kg_cement",
                                          "max_heat_increment_j_kg_cement", "released_heat_mj_m3")))
            results.append(dict(**{k: event[k] for k in (*keys, "target_um_kpa", "time_h", "status")}, **values))
        if i % 33 == 0:
            print(f"Read attained heat states from {i}/{len(runs)} thermal histories.", flush=True)
    write_csv("attainment_reaction.csv", results)
    print(f"Saved {len(uniform)} uniform requirements and {len(results)} radial-event heat records.")


if __name__ == "__main__":
    main()
