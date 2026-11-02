"""Generate and replay complete prescribed handover and continuation tasks."""

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_design import (displacement_margins, feasible_intervals,
    replay_prescribed_plan, solve_prescribed_plan, surface_extrema, surface_response)
from grouting_pino.pinn_history import PINNHistory
from grouting_pino.pinn_mechanics import PRESSURE, annulus_points
from grouting_pino.pinn_pressure import PressureEventPINN, pressure_coefficients
from grouting_pino.pinn_thermal import load_thermal


class CachedPressureFields:
    """Reuse event readouts and record the actual design workload."""

    def __init__(self, family):
        self.family = family
        self.thermal = family.thermal
        self.events = {}
        self.requested_modes = set()
        self.calls = 0

    def freeze(self):
        self.family.freeze()
        return self

    def arrays(self):
        """Expose underlying field state for history compatibility checks."""
        return self.family.arrays()

    def mean_pressure_fraction(self, coefficients):
        return self.family.mean_pressure_fraction(coefficients)

    def event(self, time_h, coefficients):
        self.calls += 1
        values = pressure_coefficients(coefficients)
        key = (float(time_h), values["uniform"], values["sector"])
        if key not in self.events:
            self.events[key] = self.family.event(time_h, values)
        if values["uniform"]:
            self.requested_modes.add((float(time_h), "uniform"))
        if values["sector"]:
            self.requested_modes.update((float(time_h), name) for name in ("sector_even", "sector_odd"))
        return self.events[key]

    def workload(self):
        return {"event_requests": self.calls, "unique_pressure_events": len(self.events),
                "unique_time_mode_pairs": len(self.requested_modes),
                "time_mode_pairs": [[t, mode] for t, mode in sorted(self.requested_modes)]}


def retain_replay(plan, family, initial, arrays, x):
    """Read the output event list through the history API and retain its fields."""
    if not plan["events"]:
        return
    prefix = plan["id"]
    surface = np.c_[x, np.zeros_like(x)]
    if initial is None:
        stage_fields = [np.zeros(len(x))]
    else:
        stage_fields = [-initial.query("soil", surface)["uy"] * 1e9 / PRESSURE]
    lower, upper = plan["displacement_bounds_um_kpa"]
    largest = 0.
    for stage, history in zip(plan["stages"][1:], replay_prescribed_plan(family, plan, initial), strict=True):
        response = lambda at: -history.query("soil", np.c_[at, np.zeros_like(at)])["uy"] * 1e9 / PRESSURE
        field = response(x)
        stage_fields.append(field)
        peaks = surface_extrema(response, x)
        largest = max(largest, *(abs(peaks[key] - stage[key]) for key in
                      ("peak_um_kpa", "uplift_um_kpa")))
        replay_margin = displacement_margins(peaks, lower, upper)["minimum_margin_um_kpa"]
        if abs(replay_margin - stage["minimum_margin_um_kpa"]) > 1e-7:
            raise RuntimeError(f"History replay changed the displacement constraint for {prefix}")
    if max(abs(history.load_coefficients[key] - plan["target_pressure"][key])
           for key in plan["target_pressure"]) > 1e-9:
        raise RuntimeError(f"History replay did not complete the pressure task for {prefix}")
    arrays[prefix + "__surface_um_kpa"] = np.array(stage_fields)
    probes = {"soil": annulus_points(np.array([6.05, 8., 15.])[:, None],
                                     np.linspace(-np.pi, np.pi, 19)[None, :]),
              "paste": annulus_points(5.91, np.linspace(-np.pi, np.pi, 37)),
              "lining": annulus_points(5.525, np.linspace(-np.pi, np.pi, 37))}
    for domain, xy in probes.items():
        arrays[domain + "__xy_m"] = xy
        for key, values in history.query(domain, xy).items():
            arrays[prefix + "__" + domain + "__" + key] = values
    plan["replay_maximum_extremum_difference_um_kpa"] = largest


def run(output, family_factory=None, mixes=("W32", "W40", "W48"), initial=None, field_source=None):
    run_started = perf_counter()
    config = yaml.safe_load((ROOT / "configs/support_handover.yaml").read_text())
    tasks = config["complete_plans"]
    lower, upper = tasks["lower_displacement_um_kpa"], tasks["upper_displacement_um_kpa"]
    x = np.linspace(-30., 30., 1201)
    arrays = {"surface_x_m": x}
    families, series, materials, loading = {}, {}, {}, {}
    for mix in mixes:
        tick = perf_counter()
        thermal = load_thermal(ROOT / f"results/pinn_support/{mix}/thermal_age/model.pt",
                               ROOT / "data/public_paste_benchmark")
        thermal_s = perf_counter() - tick
        tick = perf_counter()
        family = (PressureEventPINN.load(ROOT / f"results/pinn_support/{mix}/pressure/model.npz", thermal)
                  if family_factory is None else family_factory(mix, thermal))
        loading[mix] = {"thermal_load_s": thermal_s, "pressure_family_load_s": perf_counter() - tick}
        families[mix], series[mix] = family, CachedPressureFields(family)
        materials[mix] = thermal.paste_properties["cement_kg_m3"]
    plans, windows, batches = [], [], []
    solve_total = 0.
    replay_total = 0.

    def solve(identifier, mix, times, minima, ratio, initial=None, provider=None):
        nonlocal solve_total, replay_total
        field_provider = provider if provider is not None else series[mix]
        result = solve_prescribed_plan(field_provider, times, minima,
                    {"uniform": 1., "sector": ratio - 1.}, lower, upper, initial, x)
        result.update(id=identifier, mix_id=mix, cement_kg_m3=materials[mix],
                      pressure_ratio=ratio, task_source=tasks["source"],
                      family=f"results/pinn_support/{mix}/pressure/model.npz")
        if field_source is not None:
            result.pop("family")
            result["field_source"] = field_source
        if initial is not None:
            result["initial_checkpoint"] = tasks["continuation"]["checkpoint"]
        solve_total += result["timing_s"]["total"]
        tick = perf_counter()
        replay_family = initial.family if initial is not None and initial.family is not None else families[mix]
        retain_replay(result, replay_family, initial, arrays, x)
        replay_total += perf_counter() - tick
        plans.append(result)
        print(json.dumps({"id": identifier, "status": result["status"],
                          "margin": result.get("minimum_margin_um_kpa"),
                          "fractions": [row["remaining_fraction_increment"] for row in result["events"]]}), flush=True)
        return result

    def select(identifier, candidates):
        feasible = [row for row in candidates if row["status"] == "feasible"]
        failed = any(row["status"] == "solver_failure" for row in candidates)
        chosen = min(feasible, key=lambda row: row["cement_kg_m3"]) if feasible and not failed else None
        batches.append({"id": identifier, "candidate_plan_ids": [row["id"] for row in candidates],
            "status": "solver_failure" if failed else "feasible" if chosen else "no_feasible_plan",
            "selected_mix": chosen["mix_id"] if chosen else None,
            "selected_plan_id": chosen["id"] if chosen else None})

    deadline = config["task"]["latest_handover_h"]
    scan = np.unique(np.r_[np.arange(0., deadline, .5), deadline])
    for ratio in tasks["handover_pressure_ratios"]:
        for mix in families:
            tick = perf_counter()
            rows = {}

            def excess(time_h):
                event = series[mix].event(time_h, {"uniform": 1., "sector": ratio - 1.})
                peaks = surface_extrema(lambda at: surface_response([event], [1.], at), x)
                margins = displacement_margins(peaks, lower, upper)
                rows[time_h] = {"time_h": time_h, **peaks, **margins}
                return -margins["minimum_margin_um_kpa"]

            window = feasible_intervals(excess, scan, .01)
            window.update(mix_id=mix, pressure_ratio=ratio, scan_step_h=.5, crossing_resolution_h=.01,
                          query_time_s=perf_counter() - tick, responses=list(rows.values()))
            windows.append(window)
            for time_h in config["task"]["planned_handover_h"]:
                solve(f"handover_k{ratio:g}_{mix}_t{time_h:g}", mix, [time_h], [1.], ratio)
            for index, (start, _) in enumerate(window["intervals_h"]):
                solve(f"earliest_k{ratio:g}_{mix}_{index}", mix, [start], [1.], ratio)
        for time_h in config["task"]["planned_handover_h"]:
            candidates = [row for row in plans if row["id"].startswith(f"handover_k{ratio:g}_")
                          and row["id"].endswith(f"_t{time_h:g}")]
            select(f"handover_k{ratio:g}_t{time_h:g}", candidates)
    for task in tasks["staged"]:
        candidates = [solve(f"{task['id']}_{mix}", mix, task["times_h"],
                            task["minimum_cumulative_fractions"], task["pressure_ratio"]) for mix in families]
        select(task["id"], candidates)
    for mix in families:
        source = ROOT / tasks["envelope_source"] / f"{mix}.json"
        envelope = json.loads(source.read_text())["two_stage_fractions"]
        endpoint = next(row for row in envelope if row["early_h"] == 0. and row["late_h"] == deadline)
        result = solve(f"initial_envelope_{mix}", mix, [0., deadline],
                       [endpoint["fraction_interval"][1], 1.], 1.)
        result.update(task_source="Retained upper endpoint of the continuous two-stage fraction interval",
                      source_result=source.relative_to(ROOT).as_posix())
    task = tasks["continuation"]
    tick = perf_counter()
    if initial is None and "W40" in families:
        initial = PINNHistory.load(ROOT / task["checkpoint"], ROOT / "data/public_paste_benchmark")
    checkpoint_load_s = perf_counter() - tick
    continued = None
    if "W40" in families:
        continued = CachedPressureFields(initial.family if initial.family is not None else families["W40"])
        for ratio in task["pressure_ratios"]:
            solve(f"continuation_k{ratio:g}_W40", "W40", task["times_h"],
                  task["minimum_cumulative_fractions"], ratio, initial, continued)
    output.mkdir(parents=True, exist_ok=True)
    mode_pairs = {(mix, t, mode) for mix, provider in series.items() for t, mode in provider.requested_modes}
    if continued is not None:
        mode_pairs.update((initial.thermal.config["mix_id"], t, mode) for t, mode in continued.requested_modes)
    result = {"config": config, "materials_cement_kg_m3": materials,
        "method": "Maximize the minimum two-sided displacement margin at prescribed times",
        "fraction_definition": "Nonnegative shares of target pressure minus saved pressure",
        "selection": batches, "handover_windows": windows, "plans": plans,
        "workload": {"environment": {"python": platform.python_version(), "platform": platform.platform(),
                                     "torch_threads": torch.get_num_threads(), "blas_threads": 4},
            "retained_model_loading": loading, "checkpoint_load_s": checkpoint_load_s,
            "elapsed_before_serialization_s": perf_counter() - run_started,
            "plan_solve_s": solve_total, "replay_and_field_export_s": replay_total,
            "handover_scan_s": sum(row["query_time_s"] for row in windows),
            "unique_material_time_mode_pairs": len(mode_pairs),
            "materials": {mix: provider.workload() for mix, provider in series.items()},
            "continuation": continued.workload() if continued is not None else None,
            "cost_scope": "Online task execution with retained models; offline training is not included"}}
    (output / "plans.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "fields.npz", **arrays)
    print(json.dumps({"selection": batches, "plans": len(plans), "output": str(output)}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/pinn_support/design/complete")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=4):
        run(args.output)
