"""Compare the radial paste modulus field with its uniform-annulus reduction.

Run in the dolfinx container after run_paste_support.py. The section, mesh,
driver and all other materials are identical to the common-demand FE table.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from support_section import RadialSection
from grouting_pino.support_response import SectionResponse, first_attainment
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import simulate_paste
from run_paste_support import thermal_scenarios

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/support_design"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state-queries", action="store_true",
                        help="Check opposing initial temperatures at the 2.2 um/kPa crossing")
    parser.add_argument("--design-decisions", action="store_true",
                        help="Check illustrative selections and decisions with small response margins")
    parser.add_argument("--critical-refinement", action="store_true",
                        help="Check the cold-paste W48 4 h decision with half the annulus mesh size")
    args = parser.parse_args()
    radial = RadialSection(mesh_scale=0.5 if args.critical_refinement else 1.0)
    table = SectionResponse.load(OUT/"reference_response.npz")
    results = []
    source = "query_profiles.npz" if args.initial_state_queries else "paste_profiles.npz"
    output = "query_profile_fe_checks.csv" if args.initial_state_queries else "profile_fe_checks.csv"
    if args.design_decisions or args.critical_refinement:
        with (OUT/"query_deadline_responses.csv").open(newline="", encoding="utf-8") as f:
            responses = list(csv.DictReader(f))
        with (OUT/"query_runs.csv").open(newline="", encoding="utf-8") as f:
            runs = list(csv.DictReader(f))
        refs = load_paste_references(ROOT/"data/public_paste_benchmark")
        scenarios = {key: (props, options) for key, props, options in thermal_scenarios()}
        selections, data = [], {}
        cases = ([("cold_paste_warm_surroundings", "W48", 4.0)] if args.critical_refinement else
                 [("reference", "W32", 2.0), ("reference", "W40", 3.0),
                  ("reference", "W48", 4.0), ("paste_16", "W40", 3.0),
                  ("cold_paste_warm_surroundings", "W48", 4.0)])
        for case, mix, deadline in cases:
            candidates = [r for r in responses if r["case"] == case and r["mix_id"] == mix
                          and float(r["deadline_h"]) == deadline]
            if not all(np.isfinite(float(r["response_um_kpa"])) for r in candidates):
                raise ValueError("This decision check needs all scenarios inside the model domain")
            worst = (next(r for r in candidates if r["scenario"] == "low_heat_removal")
                     if args.critical_refinement else
                     max(candidates, key=lambda r: float(r["response_um_kpa"])))
            input_row = next(r for r in runs if r["scenario"] == worst["scenario"]
                             and r["case"] == case and r["mix_id"] == mix)
            props, options = scenarios[worst["scenario"]]
            inputs = {k: float(input_row[k]) for k in ("initial_reference_age_h", "initial_paste_temperature_c",
                                               "initial_ring_temperature_c", "soil_temperature_c")}
            run = simulate_paste(refs[mix], table.geometry, np.linspace(0, 12, 241),
                                 air_temperature_c=20.0, properties=props, **inputs, **options)
            key = case+"_"+mix+"_"+worst["scenario"]
            data[key+"_time_h"] = run.time_h
            data[key+"_radius_m"] = run.mesh["radius_m"][run.mesh["interface"]:]
            data[key+"_modulus_pa"] = run.modulus_pa
            selections.append((key, deadline))
        output = "critical_profile_resolution.csv" if args.critical_refinement else "design_profile_fe_checks.csv"
    else:
        with np.load(OUT/source) as saved:
            data = dict(saved)
        if args.initial_state_queries:
            selections = []
            for case in ("cold_paste_warm_surroundings", "warm_paste_cold_surroundings"):
                for mix in ("W32", "W40", "W48"):
                    key = case + "_" + mix
                    radius = data[key+"_radius_m"]
                    edges = np.r_[radius[0], (radius[1:]+radius[:-1])/2, radius[-1]]
                    volumes = np.diff(edges**2)/2
                    mean = data[key+"_modulus_pa"] @ volumes / volumes.sum()
                    hit = first_attainment(data[key+"_time_h"], table.compliance(mean), 2.2e-9)
                    if hit.time_h is None:
                        raise ValueError(f"No supported target crossing for {key}")
                    selections.append((key, hit.time_h))
        else:
            selections = [("reference_W40", 2.4), ("reference_W40", 4.5),
                          ("warm_W48", 3.25), ("warm_W48", 5.), ("reference_W32", 5.)]
    for key, elapsed in selections:
        times, radius = data[key+"_time_h"], data[key+"_radius_m"]
        e = np.array([np.interp(elapsed, times, col) for col in data[key+"_modulus_pa"].T])
        edges = np.r_[radius[0], (radius[1:]+radius[:-1])/2, radius[-1]]
        volumes = np.diff(edges**2)/2
        mean = float(e@volumes/volumes.sum())
        values = {}
        for model in ("uniform", "radial"):
            modulus = e if model == "radial" else np.full_like(e, mean)
            values[model+"_response_um_kpa"] = radial.response(radius, modulus)
        results.append({"profile": key, "time_h": elapsed, "mean_modulus_gpa": mean/1e9,
                        "min_modulus_gpa": e.min()/1e9, "max_modulus_gpa": e.max()/1e9,
                        **values, "table_response_um_kpa": float(table.compliance(mean))*1e9,
                        "relative_uniform_reduction_error": values["uniform_response_um_kpa"]/values["radial_response_um_kpa"]-1})
        if args.critical_refinement:
            results[-1]["mesh_size_near_cavity_m"] = radial.geometry["near_annulus_size_m"]
        print(results[-1], flush=True)
    with (OUT/output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
