"""Train event-state PINNs and resolve nominal construction decisions."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_design import (compare_plans, feasible_intervals,
                                      surface_maximum, surface_response)
from grouting_pino.pinn_mechanics import SectionPINN
from grouting_pino.pinn_event import EventPINN, SpatialEventOperator, clamped_knots
from grouting_pino.pinn_thermal import load_thermal
from grouting_pino.support_history import two_stage_fraction_interval

OUT = ROOT / "results/pinn_support/design"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_time_domain(thermal, requested_end):
    """Locate the first material/scope boundary using the actual neural fields."""
    end = min(requested_end, thermal.time_scale / 3600)
    device = next(thermal.parameters()).device

    def state(t):
        result = {}
        with torch.no_grad():
            z = torch.tensor(np.column_stack([np.linspace(0, 1, 257),
                              np.full(257, t * 3600 / thermal.time_scale)]),
                             dtype=torch.float64, device=device)
            for domain in thermal.radii:
                temperature, _, heat = thermal.fields(domain, z)
                result[domain + "_max_temperature_k"] = float(temperature.max())
                if heat is not None:
                    result["max_heat_j_kg"] = float(heat.max())
                    result["min_heat_j_kg"] = float(heat.min())
        if not np.isfinite(list(result.values())).all() or result["min_heat_j_kg"] < 0:
            raise ValueError("Invalid thermal field")
        result["valid"] = (max(result["paste_max_temperature_k"], result["lining_max_temperature_k"]) < 313.15
                           and result["max_heat_j_kg"] <= float(thermal.material.max_heat))
        return result

    scan = np.linspace(0, end, max(2, int(np.ceil(end / .05)) + 1))
    states = [state(float(t)) for t in scan]
    invalid = [i for i, row in enumerate(states) if not row["valid"]]
    bracket = None
    reason = "trained_time_end" if end < requested_end else "design_deadline"
    if invalid:
        i = invalid[0]
        if i == 0:
            raise ValueError("The supplied initial state is outside the material domain")
        a, b = float(scan[i - 1]), float(scan[i])
        reason = "temperature_scope" if max(states[i][key] for key in
                  ["paste_max_temperature_k", "lining_max_temperature_k"]) >= 313.15 else "measured_heat_domain"
        while b - a > .001:
            mid = (a + b) / 2
            if state(mid)["valid"]:
                a = mid
            else:
                b = mid
        end, bracket = a, [a, b]
    return {"start_h": 0., "end_h": end, "end_reason": reason,
            "boundary_bracket_h": bracket, "end_state": state(end),
            "radial_query_points": 257, "time_scan_step_h": .05}


class EventFields:
    def __init__(self, mix, refinement=1, thermal_directory=None):
        self.mix = mix
        self.refinement = refinement
        self.thermal_path = ((thermal_directory or ROOT / f"results/pinn_support/{mix}/thermal_age") / "model.pt").resolve()
        self.thermal = load_thermal(self.thermal_path, ROOT / "data/public_paste_benchmark")
        if self.thermal.config["mix_id"] != mix:
            raise ValueError("Thermal model belongs to a different material")
        cfg = json.loads((ROOT / "configs/pinn_support.json").read_text())["mechanics"]
        self.config = cfg | {"seed": cfg["seed"] + 101 * refinement,
            "radial_neurons": {key: value + 4 * refinement for key, value in cfg["radial_neurons"].items()},
            "angular_modes": cfg["angular_modes"] + 2 * refinement,
            "domain_points": {key: 2 * value for key, value in cfg["domain_points"].items()},
            "boundary_points": 2 * cfg["boundary_points"], "mix_id": mix}
        self.models, self.paths = {}, {}
        self.operator = None
        self.x = np.linspace(-30, 30, 1201)

    def event(self, t):
        t = float(t)
        if t in self.models:
            return self.models[t]
        # Query the full layer before training, including its boundaries.
        self.thermal.query("paste", np.linspace(5.8, 6.02, 257), t)
        modulus = lambda r: self.thermal.query("paste", np.clip(r, 5.8, 6.02), t)["modulus_pa"]
        suffix = self.thermal_path.parent.name.removeprefix("thermal")
        cache = f"mechanics{suffix}" + ("_stability" if self.refinement == 2 else "")
        name = f"t{t:.10g}" + ("_refined" if self.refinement == 1 else "")
        directory = self.thermal_path.parent.parent / cache / name
        config = self.config | {"time_h": t, "thermal_model": self.thermal_path.relative_to(ROOT).as_posix()}
        if (directory / "model.npz").exists():
            model = SectionPINN.load(directory / "model.npz", modulus)
            for key, value in config.items():
                # Existing W40 prototype models used this canonical thermal path.
                previous = model.config.get(key, value if key == "thermal_model" else None)
                if previous != value:
                    raise ValueError(f"Stored event configuration differs at {directory}: {key}")
        else:
            start = time.monotonic()
            if self.operator is None:
                reference_config = config | {"time_h": 2.}
                section = SectionPINN(reference_config, lambda r: self.thermal.query(
                    "paste", np.clip(r, 5.8, 6.02), 2.)["modulus_pa"])
                reference = EventPINN(section, self.thermal, clamped_knots([0., 4.]))
                self.operator = SpatialEventOperator([reference])
            parts, stats = self.operator.independent(t)
            model = parts["uniform"]
            directory.mkdir(parents=True, exist_ok=True)
            model.save(directory / "model.npz")
            stats.update(elapsed_s=time.monotonic() - start, config=config,
                         rank=self.operator.preconditioner.shape[1],
                         spatial_weights=model.nweights, spatial_reference_time_h=2.,
                         surface=surface_maximum(lambda x: surface_response([model], [1.], x)))
            write_json(directory / "training.json", stats)
            print(json.dumps({"mix": self.mix, "time_h": t, "elapsed_s": stats["elapsed_s"],
                              **stats["surface"]}), flush=True)
        self.models[t], self.paths[t] = model, directory
        return model

    def maximum(self, times, fractions):
        models = [self.event(t) for t in times]
        return surface_maximum(lambda x: surface_response(models, fractions, x), self.x)

    def fraction_interval(self, early, late, limit):
        models = [self.event(early), self.event(late)]
        x = self.x.copy()
        while True:
            ge, gd = [surface_response([model], [1.], x) for model in models]
            interval = two_stage_fraction_interval(ge, gd, limit)
            if interval is None:
                return {"early_h": early, "late_h": late, "fraction_interval": None}
            edge_checks = []
            for alpha in interval:
                for weights in ([alpha, 0.], [alpha, 1 - alpha]):
                    edge_checks.append(surface_maximum(lambda at: surface_response(models, weights, at)))
            points = np.array([row["peak_x_m"] for row in edge_checks])
            refined_x = np.r_[x, points]
            new_interval = two_stage_fraction_interval(
                surface_response([models[0]], [1.], refined_x),
                surface_response([models[1]], [1.], refined_x), limit)
            if new_interval is None:
                return {"early_h": early, "late_h": late, "fraction_interval": None}
            # Feature cancellation changes the last readout digits with batch
            # shape. Fraction changes below 1e-9 are numerical roundoff here.
            if max(abs(a - b) for a, b in zip(interval, new_interval)) <= 1e-9:
                return {"early_h": early, "late_h": late, "fraction_interval": list(new_interval),
                        "upper_fraction_stage1": edge_checks[2], "upper_fraction_stage2": edge_checks[3]}
            x = refined_x


class ContinuousEventFields(EventFields):
    """Use a trained event family for the same spatial design queries."""

    def __init__(self, mix):
        from grouting_pino.pinn_event import EventPINN
        super().__init__(mix)
        self.directory = ROOT / f"results/pinn_support/{mix}/event"
        self.family = EventPINN.load(self.directory / "model.npz", self.thermal)

    def event(self, t):
        t = float(t)
        if t not in self.models:
            self.models[t] = self.family.event(t)
            self.paths[t] = self.directory
        return self.models[t]


def material_design(series, task_config, scan_step, bracket_width, output_directory=OUT):
    task = task_config["task"]
    limit, deadline = task["remaining_limit_um_kpa"], task["latest_handover_h"]
    validity = valid_time_domain(series.thermal, deadline)
    end = validity["end_h"]
    scan = np.unique(np.r_[np.arange(0, end, scan_step), end,
                          [t for t in task["planned_handover_h"] if t <= end]])
    handover = feasible_intervals(lambda t: series.maximum([t], [1.])["peak_um_kpa"] - limit,
                                 scan, bracket_width)
    plans = []
    for t in task["planned_handover_h"]:
        row = {"mix_id": series.mix, "planned_handover_h": t}
        if t > end:
            row.update(status="outside_material_domain", peak_um_kpa=None, peak_x_m=None, margin_um_kpa=None)
        else:
            peak = series.maximum([t], [1.])
            margin = limit - peak["peak_um_kpa"]
            row.update(status="feasible" if margin >= 0 else "displacement_infeasible",
                       peak_um_kpa=peak["peak_um_kpa"], peak_x_m=peak["peak_x_m"], margin_um_kpa=margin)
        plans.append(row)
    half = None
    fraction_rows = []
    if end >= deadline:
        def half_excess(t):
            first = series.maximum([t], [.5])["peak_um_kpa"]
            final = series.maximum([t, deadline], [.5, .5])["peak_um_kpa"]
            return max(first, final) - limit

        half = feasible_intervals(half_excess, scan, bracket_width)
        for t in scan:
            fraction_rows.append(series.fraction_interval(float(t), deadline, limit))
        first = half["intervals_h"][0][0] if half["intervals_h"] else None
        if first is not None:
            half["first_loading_stage1"] = series.maximum([first], [.5])
            half["first_loading_stage2"] = series.maximum([first, deadline], [.5, .5])
            half["controlling_stage"] = 1 if half["first_loading_stage1"]["peak_um_kpa"] > half["first_loading_stage2"]["peak_um_kpa"] else 2
        half["initial_half_load_final"] = series.maximum([0., deadline], [.5, .5])
    result = {"mix_id": series.mix, "cement_kg_m3": series.thermal.paste_properties["cement_kg_m3"],
              "thermal_model": series.thermal_path.relative_to(ROOT).as_posix(),
              "validity": validity, "limit_um_kpa": limit, "scan_step_h": scan_step,
              "crossing_resolution_h": bracket_width, "handover": handover,
              "plans": plans, "half_load": half, "two_stage_fractions": fraction_rows}
    result["event_representation"] = ("continuous" if isinstance(series, ContinuousEventFields)
                                      else "independent")
    times = sorted(series.models)
    result["events"] = [{"time_h": t, "directory": series.paths[t].relative_to(ROOT).as_posix(),
                         **series.maximum([t], [1.])} for t in times]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / f"{series.mix}.json", result)
    np.savez_compressed(output_directory / f"{series.mix}_surface.npz", time_h=times, x_m=series.x,
                        surface_um_kpa=np.array([surface_response([series.models[t]], [1.], series.x) for t in times]))
    print(json.dumps({"mix": series.mix, "handover": handover["intervals_h"],
                      "half_load": half["intervals_h"] if half else None, "plans": plans}), flush=True)


def assemble(config, output_directory=OUT):
    materials = [json.loads((output_directory / f"{mix}.json").read_text()) for mix in ["W32", "W40", "W48"]]
    task = config["task"]
    plans = [row for material in materials for row in material["plans"]]
    rows = compare_plans(materials, plans, task["planned_handover_h"],
                         task["latest_handover_h"], task["reference_mix"])
    write_csv(output_directory / "planned_responses.csv", plans)
    write_csv(output_directory / "same_task_comparison.csv", rows)
    windows = []
    for material in materials:
        for start, end in material["handover"]["intervals_h"]:
            windows.append({"mix_id": material["mix_id"], "cement_kg_m3": material["cement_kg_m3"],
                            "start_h": start, "end_h": end,
                            "mean_ring_cycle_min": start * 60 / config["process"]["pressure_supported_rings"]})
    if windows:
        write_csv(output_directory / "material_windows.csv", windows)
    print(json.dumps(rows, indent=2), flush=True)


def qualification_design(config):
    """Resolve material boundaries and joint milestone demands from event fields."""
    from scipy.interpolate import PPoly
    from grouting_pino.pinn_design import surface_extrema
    from grouting_pino.pinn_qualification import (latest_required_increments,
        milestone_demand_polygon, polynomial_range, polynomial_sublevel_intervals,
        spline_polynomial, surface_time_spline)

    output = OUT / "qualification"
    task = config["task"]
    limit, deadline = task["remaining_limit_um_kpa"], task["latest_handover_h"]
    timetables = [row for row in config["complete_plans"]["staged"]
                 if row["pressure_ratio"] == 1.]
    x = np.linspace(-30., 30., 1201)
    times = np.linspace(0., deadline, 401)
    fractions = np.linspace(0., 1., 201)
    arrays = {"surface_x_m": x, "time_h": times, "early_fraction": fractions}
    result = {"limit_um_kpa": limit, "completion_h": deadline,
              "reference_mix": task["reference_mix"], "materials": [],
              "timetables": timetables}

    for mix in ("W32", "W40", "W48"):
        series = ContinuousEventFields(mix)
        family = series.family
        surface = spline_polynomial(surface_time_spline(family, x))
        axis = PPoly(surface.c[:, :, len(x)//2], surface.x, extrapolate=False)
        minimum, maximum = polynomial_range(surface)
        derivative_min, derivative_max = polynomial_range(surface.derivative())
        axis_gap = PPoly(axis.c[:, :, None] - surface.c, surface.x, extrapolate=False)
        dominance_min, _ = polynomial_range(axis_gap)
        if minimum.min() < 0 or dominance_min.min() < -1e-10:
            raise ValueError(f"{mix}: use full spatial constraints for the time inverse")
        intervals = polynomial_sublevel_intervals(axis, limit)
        final_response = float(axis(deadline))
        material = {"mix_id": mix,
            "cement_kg_m3": series.thermal.paste_properties["cement_kg_m3"],
            "event_model": f"results/pinn_support/{mix}/event/model.npz",
            "surface_properties": {"time_intervals": len(surface.x)-1,
                "surface_points": len(x), "minimum_response_um_kpa": float(minimum.min()),
                "maximum_response_um_kpa": float(maximum.max()),
                "minimum_time_slope_um_kpa_h": float(derivative_min.min()),
                "maximum_time_slope_um_kpa_h": float(derivative_max.max()),
                "minimum_axis_dominance_um_kpa": float(dominance_min.min())},
            "handover_intervals_h": intervals,
            "handover_boundary": [surface_extrema(lambda at, t=a: surface_response(
                [series.event(t)], [1.], at), x) for a, _ in intervals],
            "final_response_um_kpa": final_response,
            "history_allowance_um_kpa": limit-final_response,
            "inverse_queries": [], "milestone_demands": []}
        arrays[mix+"__response_um_kpa"] = axis(times)
        first_times = []
        for alpha in fractions:
            coefficients = alpha * axis.c.copy()
            coefficients[-1] += (1-alpha)*final_response
            early = polynomial_sublevel_intervals(PPoly(coefficients, axis.x), limit)
            first_times.append(early[0][0] if early else np.nan)
            if any(np.isclose(alpha, value) for value in (0., .1, .2, .5, .8, 1.)):
                material["inverse_queries"].append({"early_fraction": float(alpha),
                                                     "feasible_times_h": early})
        arrays[mix+"__first_loading_h"] = np.array(first_times)

        for timetable in timetables:
            event_times = np.asarray(timetable["times_h"])
            demand = np.asarray(timetable["minimum_cumulative_fractions"])
            fields = surface(event_times)
            ordering = float(np.min(fields[:-1]-fields[1:]))
            if ordering < 0:
                raise ValueError(f"{mix}/{timetable['id']}: resolve demands with the full plan solver")
            polygon = milestone_demand_polygon(fields, -limit, limit)
            increments = latest_required_increments(demand)
            accumulated = np.cumsum(increments[:, None]*fields, axis=0)
            models = [series.event(t) for t in event_times]
            stages = [surface_extrema(lambda at, n=n: surface_response(
                models[:n], increments[:n], at), x) for n in range(1, len(models)+1)]
            axis_values = fields[:, len(x)//2]
            weights = -np.diff(axis_values)
            reserve = limit-axis_values[-1]
            cost = weights*demand[:-1]
            # Intersect the requested second-stage requirement with the complete region.
            numerator = reserve-demand[1]*weights[1]
            max_first = min(demand[1], numerator/weights[0]) if numerator >= 0 else None
            first_target = (limit-(demand[1]-demand[0])*axis_values[1]
                            -(1-demand[1])*axis_values[2])/demand[0]
            early_intervals = polynomial_sublevel_intervals(axis, first_target)
            early_intervals = [[a, min(b, float(event_times[1]))] for a, b in early_intervals
                               if a <= event_times[1]]
            active_vertices = [point for point in polygon
                               if abs(float(np.dot(weights, point))-reserve) < 1e-10]
            boundary_demand = np.r_[np.mean(active_vertices, axis=0) if active_vertices
                                    else [1., 1.], 1.]
            boundary_increments = latest_required_increments(boundary_demand)
            boundary_fields = np.cumsum(boundary_increments[:, None]*fields, axis=0)
            boundary_stages = [surface_extrema(lambda at, n=n: surface_response(
                models[:n], boundary_increments[:n], at), x) for n in range(1, len(models)+1)]
            row = {"id": timetable["id"], "event_times_h": event_times.tolist(),
                "demand": demand.tolist(), "event_response_um_kpa": axis_values.tolist(),
                "memory_weights_um_kpa": weights.tolist(),
                "history_cost_um_kpa": cost.tolist(), "history_allowance_um_kpa": reserve,
                "stage_field_order_gap_um_kpa": ordering, "polygon": polygon.tolist(),
                "increments": increments.tolist(), "stages": stages,
                "controlling_stage": int(np.argmax([s["absolute_peak_um_kpa"] for s in stages]))+1,
                "qualifies": max(s["absolute_peak_um_kpa"] for s in stages) <= limit,
                "max_first_fraction_at_requested_second": max_first,
                "first_deadline_for_requested_fractions_h": early_intervals,
                "boundary_plan": {"demand": boundary_demand.tolist(),
                    "increments": boundary_increments.tolist(), "stages": boundary_stages,
                    "active_constraint": "displacement" if active_vertices else "demand_domain"}}
            material["milestone_demands"].append(row)
            prefix = mix+"__"+timetable["id"]
            arrays[prefix+"__stage_surface_um_kpa"] = accumulated
            arrays[prefix+"__boundary_surface_um_kpa"] = boundary_fields
        result["materials"].append(material)
        print(json.dumps({"mix": mix, "handover_h": intervals,
            "history_allowance_um_kpa": material["history_allowance_um_kpa"],
            "milestones": [{"id": row["id"], "max_first": row["max_first_fraction_at_requested_second"],
                "first_deadline_h": row["first_deadline_for_requested_fractions_h"],
                "history_cost_um_kpa": row["history_cost_um_kpa"]}
                for row in material["milestone_demands"]]}), flush=True)
    for timetable in timetables:
        admitted = [m for m in result["materials"] if next(r for r in m["milestone_demands"]
                    if r["id"] == timetable["id"])["qualifies"]]
        timetable["selected_mix"] = min(admitted, key=lambda m: m["cement_kg_m3"])["mix_id"] if admitted else None
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "qualification.json", result)
    np.savez_compressed(output / "fields.npz", **arrays)


def stability_design(series, task_config, crossing_search="secant"):
    """Re-evaluate consequential times with independent, larger neural fields."""
    from check_pinn_mechanics import assess

    nominal = json.loads((OUT / f"{series.mix}.json").read_text())
    task = task_config["task"]
    limit, deadline = task["remaining_limit_um_kpa"], task["latest_handover_h"]
    validity = valid_time_domain(series.thermal, deadline)
    if validity["end_h"] < deadline:
        raise ValueError("The stability run changes the valid design interval; re-evaluate the design")
    result = {"mix_id": series.mix, "thermal_model": series.thermal_path.relative_to(ROOT).as_posix(),
              "mechanics_config": series.config, "validity": validity, "plans": []}
    for t in task["planned_handover_h"]:
        peak = series.maximum([t], [1.])
        result["plans"].append({"time_h": t, **peak,
                                "margin_um_kpa": limit - peak["peak_um_kpa"]})
    for kind in ["handover", "half_load"]:
        if not nominal[kind] or not nominal[kind]["intervals_h"]:
            continue
        center = nominal[kind]["intervals_h"][0][0]

        def excess(t):
            if kind == "handover":
                return series.maximum([t], [1.])["peak_um_kpa"] - limit
            return max(series.maximum([t], [.5])["peak_um_kpa"],
                       series.maximum([t, deadline], [.5, .5])["peak_um_kpa"]) - limit

        a, b = max(0., center - .125), min(deadline, center + .125)
        fa, fb = excess(a), excess(b)
        while fa <= 0 and a > 0:
            a = max(0., a - .25)
            fa = excess(a)
        while fb > 0 and b < deadline:
            b = min(deadline, b + .25)
            fb = excess(b)
        # A local secant proposes a narrow empirical bracket; its two signs
        # are checked with fresh event PINNs. Fall back to the wider bracket
        # only when the actual fields reject that proposal.
        if crossing_search == "secant" and fa > 0 and fb < 0:
            estimate = a - fa * (b - a) / (fb - fa)
            left, right = round(max(a, estimate - .015), 6), round(min(b, estimate + .015), 6)
            if excess(left) > 0 and excess(right) <= 0:
                a, b = left, right
        crossing = feasible_intervals(excess, [a, b], .03125)
        result[kind] = crossing
        result[kind]["nominal_crossings"] = nominal[kind]["crossings"]
        if crossing["intervals_h"]:
            first = crossing["intervals_h"][0][0]
            model = series.event(first)
            validation = assess(model, order=64)
            write_json(series.paths[first] / "validation.json", validation)
            result[kind]["checked_event_directory"] = series.paths[first].relative_to(ROOT).as_posix()
    result["initial_half_load_final"] = series.maximum([0., deadline], [.5, .5])
    result["initial_two_stage_fraction"] = series.fraction_interval(0., deadline, limit)
    result["events"] = [{"time_h": t, "directory": series.paths[t].relative_to(ROOT).as_posix()}
                        for t in sorted(series.models)]
    write_json(OUT / f"{series.mix}_stability.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mix", choices=["W32", "W40", "W48"])
    p.add_argument("--refinement", type=int, default=1, choices=[1, 2])
    p.add_argument("--thermal-directory", type=Path)
    p.add_argument("--query-times", type=float, nargs="+")
    p.add_argument("--scan-step", type=float, default=.5)
    p.add_argument("--bracket-width", type=float, default=.01)
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--qualification", action="store_true",
                   help="Compute demand boundaries from the retained event-time polynomials")
    p.add_argument("--continuous", action="store_true", help="Query the retained continuous event family")
    p.add_argument("--stability", action="store_true")
    p.add_argument("--stability-search", choices=["secant", "bisection"], default="secant")
    args = p.parse_args()
    if args.continuous and (args.stability or args.refinement != 1 or args.thermal_directory):
        p.error("The retained continuous family supplies its own spatial and thermal configuration")
    if (args.refinement != 1 or args.thermal_directory) and not (args.query_times or args.stability):
        p.error("Stability variants require --query-times or --stability")
    config = yaml.safe_load((ROOT / "configs/support_handover.yaml").read_text())
    torch.set_num_threads(2)
    output_directory = OUT / "continuous" if args.continuous else OUT
    if args.qualification:
        with threadpool_limits(limits=4):
            qualification_design(config)
    elif args.assemble:
        assemble(config, output_directory)
    elif args.mix:
        with threadpool_limits(limits=4):
            series = (ContinuousEventFields(args.mix) if args.continuous else
                      EventFields(args.mix, 2 if args.stability else args.refinement, args.thermal_directory))
            if args.query_times:
                for t in args.query_times:
                    print(json.dumps({"time_h": t, **series.maximum([t], [1.])}), flush=True)
            elif args.stability:
                stability_design(series, config, args.stability_search)
            else:
                material_design(series, config, args.scan_step, args.bracket_width, output_directory)
    else:
        p.error("Specify --mix or --assemble")


if __name__ == "__main__":
    main()
