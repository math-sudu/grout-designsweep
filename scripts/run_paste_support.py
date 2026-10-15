"""Representative common-demand calculations for the public paste benchmark.

Continuous demand is sampled directly in section compliance; no site safety
limit or candidate-relative endpoint is imposed. Domain termination produces
an unknown later decision, never an artificial infeasible candidate.
"""

import csv
from collections import defaultdict
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import PasteThermalProperties, simulate_paste
from grouting_pino.support_response import SectionResponse, first_attainment, meets_deadline

OUT = ROOT / "results/support_design"


def write_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_decisions(decisions):
    """Keep unknown candidates distinct when comparing a common fill volume."""
    grouped = defaultdict(list)
    for row in decisions:
        grouped[(row["scenario"], row["case"], row["deadline_h"], row["target_um_kpa"])].append(row)
    rows = []
    for (scenario, case, deadline, target), candidates in grouped.items():
        feasible = sorted((r for r in candidates if r["status"] == "feasible"),
                          key=lambda r: float(r["cement_kg_m3"]))
        unknown = [r for r in candidates if r["status"] == "outside_model_domain"]
        best = feasible[0] if feasible else None
        resolved = best is not None and all(float(r["cement_kg_m3"]) >= float(best["cement_kg_m3"])
                                            for r in unknown)
        rows.append({"scenario": scenario, "case": case, "deadline_h": deadline,
                     "target_um_kpa": target,
                     "feasible_mix_ids": ";".join(r["mix_id"] for r in feasible),
                     "unknown_mix_ids": ";".join(r["mix_id"] for r in unknown),
                     "least_cement_supported_mix": best["mix_id"] if best else "",
                     "cement_kg_m3": best["cement_kg_m3"] if best else "",
                     "least_cement_choice_resolved": resolved})
    return rows


def thermal_scenarios():
    """Finite transfer assumptions; these are not confidence limits."""
    base = PasteThermalProperties()
    scenarios = [("nominal", base, {})]
    for ea, cp, k in itertools.product((30000., 50000.), (0.74, 1.), (0.77, 1.16)):
        scenarios.append((f"ea{ea/1000:g}_cp{cp:g}_k{k:g}", replace(base,
                          activation_energy_j_mol=ea, paste_cp_factor=cp,
                          conductivity_w_mk=k), {}))
    scenarios += [
        ("low_heat_removal", base, {"air_h_w_m2k": 4., "soil_h_w_m2k": 8.,
                                   "ring_conductivity": 1.5, "ring_capacity": 1.8e6}),
        ("high_heat_removal", base, {"air_h_w_m2k": 16., "soil_h_w_m2k": 32.,
                                    "ring_conductivity": 2.2, "ring_capacity": 2.6e6})]
    return scenarios


def main():
    refs = load_paste_references(ROOT/"data/public_paste_benchmark")
    section = SectionResponse.load(OUT/"reference_response.npz")
    t = np.linspace(0, 12, 241)
    cases = {"cool": (16., 16., 16.), "reference": (20., 20., 20.),
             "warm": (30., 30., 30.),
             "air_cycle": (20+5*np.sin(2*np.pi*t/24), 20., 20.)}
    scenarios = thermal_scenarios()
    demands = np.linspace(1.5, 3.5, 41)*1e-9  # m/Pa = micrometre/kPa / 1e9
    deadlines = np.arange(1., 6.1, 0.5)
    summaries, crossings, decisions, histories, profiles = [], [], [], [], {}
    convergence = []
    for scenario, props, options in scenarios:
        for case, (air, soil, ring) in cases.items():
            # The periodic case tests the explicit lining/air position once;
            # parameter transfer is checked across the three thermal levels.
            if case == "air_cycle" and scenario != "nominal":
                continue
            for mix, reference in refs.items():
                run = simulate_paste(reference, section.geometry, t, air, soil, props,
                                     initial_ring_temperature_c=ring, **options)
                c = section.compliance(run.effective_modulus_pa)
                ch = section.compliance(run.harmonic_modulus_pa)
                stored = (run.temperature_c-run.temperature_c[0]) @ run.mesh["capacity"]
                released = ((run.heat_j_kg_cement-run.heat_j_kg_cement[0])
                            @ run.mesh["paste_volume"] * run.volume_properties["cement_kg_m3"])
                residual = stored-released-run.boundary_energy_j
                summaries.append({"scenario": scenario, "case": case, "mix_id": mix,
                    **run.volume_properties, "supported_end_h": run.time_h[-1],
                    "end_reason": run.status, "max_temperature_c": run.temperature_c.max(),
                    "end_modulus_gpa": run.effective_modulus_pa[-1]/1e9,
                    "max_energy_residual_j_per_m_per_rad": abs(residual).max(),
                    "max_reduction_spread_um_kpa": np.max(ch-c)*1e9})
                for demand in demands:
                    attainment = first_attainment(run.time_h, c, demand)
                    crossings.append({"scenario": scenario, "case": case, "mix_id": mix,
                        "target_um_kpa": demand*1e9,
                        "required_modulus_gpa": section.required_modulus_pa(demand)/1e9,
                        "first_time_h": attainment.time_h, "attainment_status": attainment.status,
                        "supported_end_h": run.time_h[-1], "end_reason": run.status})
                    for deadline in deadlines:
                        supported = deadline <= run.time_h[-1]
                        feasible = meets_deadline(run.time_h, c, demand, deadline) if supported else None
                        decisions.append({"scenario": scenario, "case": case, "mix_id": mix,
                            "deadline_h": deadline, "target_um_kpa": demand*1e9,
                            "response_um_kpa": float(np.interp(deadline, run.time_h, c))*1e9 if supported else None,
                            "status": ("feasible" if feasible else "infeasible") if supported else "outside_model_domain",
                            "cement_kg_m3": run.volume_properties["cement_kg_m3"]})
                if scenario == "nominal":
                    for j, elapsed in enumerate(run.time_h):
                        histories.append({"case": case, "mix_id": mix, "time_h": elapsed,
                            "modulus_gpa": run.effective_modulus_pa[j]/1e9,
                            "harmonic_modulus_gpa": run.harmonic_modulus_pa[j]/1e9,
                            "response_um_kpa": c[j]*1e9,
                            "paste_min_temperature_c": run.temperature_c[j, run.mesh["interface"]:].min(),
                            "paste_max_temperature_c": run.temperature_c[j, run.mesh["interface"]:].max()})
                    if case in ("reference", "warm"):
                        key = case+"_"+mix
                        profiles[key+"_time_h"] = run.time_h
                        profiles[key+"_radius_m"] = run.mesh["radius_m"][run.mesh["interface"]:]
                        profiles[key+"_modulus_pa"] = run.modulus_pa
                        profiles[key+"_temperature_c"] = run.temperature_c
                        profiles[key+"_heat_j_kg_cement"] = run.heat_j_kg_cement
                    if case in ("reference", "warm") and mix == "W40":
                        fine = simulate_paste(reference, section.geometry, t, air, soil, props,
                                              initial_ring_temperature_c=ring, n_paste=49, n_ring=97,
                                              rtol=5e-6, max_step_h=0.05)
                        cf = section.compliance(fine.effective_modulus_pa)
                        common = run.time_h[run.time_h <= fine.time_h[-1]]
                        target = 2.2e-9
                        a = first_attainment(run.time_h, c, target)
                        b = first_attainment(fine.time_h, cf, target)
                        convergence.append({"case": case,
                            "max_response_change_um_kpa": float(np.max(abs(np.interp(common, run.time_h, c)-np.interp(common, fine.time_h, cf)))*1e9),
                            "temperature_scope_end_change_h": fine.time_h[-1]-run.time_h[-1],
                            "first_time_change_h_at_2_2_um_kpa": None if a.time_h is None or b.time_h is None else b.time_h-a.time_h})
            print(f"Completed {scenario} / {case}", flush=True)
    write_csv("paste_runs.csv", summaries)
    write_csv("support_attainment.csv", crossings)
    write_csv("support_equal_designs.csv", decisions)
    write_csv("feasible_recipe_map.csv", summarize_decisions(decisions))
    write_csv("paste_histories.csv", histories)
    np.savez_compressed(OUT/"paste_profiles.npz", **profiles)
    (OUT/"paste_resolution.json").write_text(json.dumps(convergence, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(convergence, indent=2))


if __name__ == "__main__":
    main()
