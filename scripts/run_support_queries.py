"""Run finite-scenario support queries on the public paste benchmark.

Each physical trajectory is integrated once and then reused for all demands
and deadlines. Temperatures and initial reference ages are declared numerical
scenarios on the public paste benchmark, not field measurements.
"""

import argparse
import csv
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import simulate_paste
from grouting_pino.support_response import (
    SectionResponse, classify_scenarios, deadline_responses, first_attainment,
)
from run_paste_support import thermal_scenarios

OUT = ROOT / "results/support_design"
COST_TASK = ("One coupled trajectory per material/initial-state/environment/thermal assumption; "
             "reuse for 41 demands and 61 deadlines. Report integration, vector deadline readout, "
             "and design grouping/export separately. The component sum excludes source loading, "
             "raw-trajectory export and offline section-table/field-FE work.")


def write_rows(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_designs(responses, demands, cement_mass, response_probe_um_kpa=0.0):
    """Separate agreement, model disagreement and censored candidates."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in responses:
        grouped[(row["case"], row["deadline_h"])][row["mix_id"]].append(
            row["response_um_kpa"] * 1e-9)
    if response_probe_um_kpa:
        delta = response_probe_um_kpa * 1e-9
        for candidates in grouped.values():
            for mix, values in candidates.items():
                values = np.asarray(values)
                candidates[mix] = np.r_[np.maximum(0, values-delta), values, values+delta]
    envelopes, designs = [], []
    for (case, deadline), candidates in grouped.items():
        for mix, values in candidates.items():
            values = np.asarray(values)
            known = values[np.isfinite(values)]
            envelopes.append(dict(case=case, mix_id=mix, deadline_h=deadline,
                                  minimum_known_response_um_kpa=float(known.min()*1e9) if known.size else None,
                                  maximum_known_response_um_kpa=float(known.max()*1e9) if known.size else None,
                                  known_count=len(known), outside_domain_count=len(values)-len(known)))
        for target in demands:
            states = {mix: classify_scenarios(values, target)["status"]
                      for mix, values in candidates.items()}
            feasible = sorted((mix for mix, state in states.items()
                               if state == "feasible_all_evaluated"), key=cement_mass.get)
            infeasible = [mix for mix, state in states.items() if state == "infeasible_all_evaluated"]
            sensitive = [mix for mix, state in states.items() if state == "assumption_sensitive"]
            limited = [mix for mix, state in states.items() if state == "domain_limited"]
            best = feasible[0] if feasible else None
            unresolved = sensitive + limited
            resolved = (all(cement_mass[m] >= cement_mass[best] for m in unresolved)
                        if best else not unresolved)
            margin = target - max(candidates[best]) if best else None
            designs.append(dict(assumption_set="thermal_and_spatial_probe" if response_probe_um_kpa else "thermal",
                                response_probe_um_kpa=response_probe_um_kpa,
                                case=case, deadline_h=deadline, target_um_kpa=target*1e9,
                                feasible_all_evaluated=";".join(feasible),
                                infeasible_all_evaluated=";".join(infeasible),
                                assumption_sensitive=";".join(sensitive),
                                domain_limited=";".join(limited),
                                least_cement_all_evaluated_mix=best or "",
                                cement_kg_m3=cement_mass[best] if best else None,
                                selected_margin_um_kpa=margin*1e9 if margin is not None else None,
                                least_cement_choice_resolved=resolved))
    return envelopes, designs


def write_designs(responses, demands, cement_mass, *, with_spatial_probe=False):
    start = perf_counter()
    envelopes, designs = summarize_designs(responses, demands, cement_mass)
    write_rows("query_envelopes.csv", envelopes)
    if not with_spatial_probe:
        write_rows("query_designs.csv", designs)
        return perf_counter() - start
    # An empirical perturbation scale tests whether the finite thermal-scenario
    # decision survives spatial-reduction differences of the size observed in
    # the actual FE checks. It is not asserted to bound untested fields.
    checked = []
    for name in ("profile_fe_checks.csv", "query_profile_fe_checks.csv", "design_profile_fe_checks.csv"):
        with (OUT / name).open(newline="", encoding="utf-8") as stream:
            checked.extend(csv.DictReader(stream))
    probe = max(abs(float(row["radial_response_um_kpa"])-float(row["table_response_um_kpa"]))
                for row in checked)
    _, perturbed = summarize_designs(responses, demands, cement_mass, probe)
    write_rows("query_designs.csv", designs + perturbed)
    return perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true",
                        help="Reuse saved trajectories and updated spatial FE checks")
    args = parser.parse_args()
    refs = load_paste_references(ROOT / "data/public_paste_benchmark")
    section = SectionResponse.load(OUT / "reference_response.npz")
    if args.summarize:
        with (OUT / "query_deadline_responses.csv").open(newline="", encoding="utf-8") as stream:
            responses = list(csv.DictReader(stream))
        for row in responses:
            row["deadline_h"] = float(row["deadline_h"])
            row["response_um_kpa"] = float(row["response_um_kpa"])
        props = thermal_scenarios()[0][1]
        masses = {mix: props.volume_properties(ref.water_cement_ratio)["cement_kg_m3"]
                  for mix, ref in refs.items()}
        elapsed = write_designs(responses, np.linspace(1.5, 3.5, 41)*1e-9, masses,
                               with_spatial_probe=True)
        timing = json.loads((OUT / "query_cost.json").read_text(encoding="utf-8"))
        timing["task"] = COST_TASK
        timing["design_grouping_and_export_s"] = elapsed
        timing["compute_and_design_export_component_sum_s"] = (
            timing["trajectory_total_s"] + timing["readout_total_s"] + elapsed)
        (OUT / "query_cost.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
        return
    cases = {
        "reference": {},
        "paste_16": {"initial_paste_temperature_c": 16.0},
        "paste_30": {"initial_paste_temperature_c": 30.0},
        "ring_16": {"initial_ring_temperature_c": 16.0},
        "ring_30": {"initial_ring_temperature_c": 30.0},
        "soil_16": {"soil_temperature_c": 16.0},
        "soil_30": {"soil_temperature_c": 30.0},
        "age_3": {"initial_reference_age_h": 3.0},
        "age_6": {"initial_reference_age_h": 6.0},
        "all_16": {"initial_paste_temperature_c": 16.0,
                   "initial_ring_temperature_c": 16.0, "soil_temperature_c": 16.0},
        "all_30": {"initial_paste_temperature_c": 30.0,
                   "initial_ring_temperature_c": 30.0, "soil_temperature_c": 30.0},
        "cold_paste_warm_surroundings": {"initial_paste_temperature_c": 16.0,
                                        "initial_ring_temperature_c": 30.0,
                                        "soil_temperature_c": 30.0},
        "warm_paste_cold_surroundings": {"initial_paste_temperature_c": 30.0,
                                        "initial_ring_temperature_c": 16.0,
                                        "soil_temperature_c": 16.0},
    }
    time = np.linspace(0, 12, 241)
    demands = np.linspace(1.5, 3.5, 41) * 1e-9
    deadlines = np.linspace(0, 6, 61)
    runs, crossings, histories, fields, responses = [], [], [], {}, []
    cement_mass = {}
    for (scenario, properties, options), (case, override) in product(thermal_scenarios(), cases.items()):
        for mix, ref in refs.items():
            inputs = dict(initial_reference_age_h=ref.start_age_h,
                          initial_paste_temperature_c=20.0,
                          initial_ring_temperature_c=20.0, soil_temperature_c=20.0)
            inputs.update(override)
            start = perf_counter()
            run = simulate_paste(ref, section.geometry, time, air_temperature_c=20.0,
                                 properties=properties, **inputs, **options)
            response = section.compliance(run.effective_modulus_pa)
            solve_s = perf_counter() - start
            cement_mass[mix] = run.volume_properties["cement_kg_m3"]
            start = perf_counter()
            # All deadlines and demand levels consume the same trajectory.
            values = deadline_responses(run.time_h, response, deadlines)
            supported = np.isfinite(values)
            decisions = values[supported, None] <= demands[None, :]
            query_s = perf_counter() - start
            runs.append(dict(scenario=scenario, case=case, mix_id=mix, **inputs,
                             activation_energy_j_mol=properties.activation_energy_j_mol,
                             paste_cp_factor=properties.paste_cp_factor,
                             conductivity_w_mk=properties.conductivity_w_mk,
                             supported_end_h=run.time_h[-1], end_reason=run.status,
                             end_response_um_kpa=response[-1] * 1e9,
                             initial_response_um_kpa=response[0] * 1e9,
                             trajectory_wall_s=solve_s, query_wall_s=query_s,
                             queried_cells=decisions.size,
                             outside_domain_cells=int((~supported).sum()) * len(demands)))
            for deadline, value in zip(deadlines, values):
                responses.append(dict(scenario=scenario, case=case, mix_id=mix,
                                      deadline_h=deadline, response_um_kpa=value*1e9))
            for target in demands:
                hit = first_attainment(run.time_h, response, target)
                crossings.append(dict(scenario=scenario, case=case, mix_id=mix, target_um_kpa=target * 1e9,
                                      time_h=hit.time_h, status=hit.status,
                                      supported_end_h=run.time_h[-1]))
            if scenario == "nominal":
                for j, t in enumerate(run.time_h):
                    histories.append(dict(case=case, mix_id=mix, time_h=t,
                                          response_um_kpa=response[j] * 1e9,
                                          modulus_gpa=run.effective_modulus_pa[j] / 1e9,
                                          paste_max_temperature_c=run.temperature_c[j, run.mesh["interface"]:].max()))
            # The opposing initial-temperature cases probe the spatial
            # reduction where a uniform effective modulus is most doubtful.
            if scenario == "nominal" and case in ("cold_paste_warm_surroundings", "warm_paste_cold_surroundings"):
                key = case + "_" + mix
                fields[key + "_time_h"] = run.time_h
                fields[key + "_radius_m"] = run.mesh["radius_m"][run.mesh["interface"]:]
                fields[key + "_modulus_pa"] = run.modulus_pa
        print(f"{scenario} / {case}", flush=True)
    write_rows("query_runs.csv", runs)
    write_rows("query_attainment.csv", crossings)
    write_rows("query_histories.csv", histories)
    write_rows("query_deadline_responses.csv", responses)
    grouping_s = write_designs(responses, demands, cement_mass)
    np.savez_compressed(OUT / "query_profiles.npz", **fields)
    timing = dict(platform=platform.platform(), processor=platform.processor(),
                  python=sys.version.split()[0], trajectory_count=len(runs),
                  trajectory_total_s=sum(r["trajectory_wall_s"] for r in runs),
                  trajectory_median_s=float(np.median([r["trajectory_wall_s"] for r in runs])),
                  readout_total_s=sum(r["query_wall_s"] for r in runs),
                  queried_cells=sum(r["queried_cells"] for r in runs),
                  outside_domain_cells=sum(r["outside_domain_cells"] for r in runs),
                  design_grouping_and_export_s=grouping_s,
                  compute_and_design_export_component_sum_s=(sum(r["trajectory_wall_s"]+r["query_wall_s"] for r in runs)+grouping_s),
                  task=COST_TASK)
    (OUT / "query_cost.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
