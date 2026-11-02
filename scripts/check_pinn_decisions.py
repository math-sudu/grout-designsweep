"""Assess the reported staged decisions with richer independent PINN events."""

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PPoly
from scipy.optimize import brentq
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_design import surface_extrema, surface_response
from grouting_pino.pinn_event import EventPINN, SpatialEventOperator
from grouting_pino.pinn_mechanics import SectionPINN
from grouting_pino.pinn_pressure import PressureEventPINN
from grouting_pino.pinn_qualification import (
    polynomial_range, spline_polynomial, surface_time_spline,
)
from grouting_pino.pinn_thermal import load_thermal
from check_pinn_thermal_events import assess_events


OUTPUT = ROOT / "results/pinn_support/design/assessment"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class IndependentEvents:
    """Reuse the existing richer configuration and its retained exact times."""

    def __init__(self, mix, nominal):
        self.mix = mix
        from run_pinn_design import EventFields
        series = EventFields(mix, refinement=2)
        self.directory = ROOT / f"results/pinn_support/{mix}/mechanics_age_stability"
        self.config = series.config | {"time_h": 4.,
            "thermal_model": series.thermal_path.relative_to(ROOT).as_posix()}
        self.thermal = series.thermal
        self.knots = nominal.knots
        self.degree = nominal.degree
        self.operator = None
        self.models = {}
        self.sources = {}

    def event(self, time):
        time = float(time)
        if time in self.models:
            return self.models[time]
        config = self.config | {"time_h": time}
        modulus = lambda radius: self.thermal.query(
            "paste", np.clip(radius, 5.8, 6.02), time)["modulus_pa"]
        retained = self.directory / f"t{time:g}"
        output = OUTPUT / self.mix / "thermal_age" / f"t{time:.10g}"
        if ((retained / "model.npz").exists()
                and json.loads((retained / "training.json").read_text())["config"] == config):
            model = SectionPINN.load(retained / "model.npz", modulus)
            source = retained / "model.npz"
        elif (output / "model.npz").exists():
            model = SectionPINN.load(output / "model.npz", modulus)
            if model.config != config:
                raise ValueError(f"Independent event settings changed: {output}")
            source = output / "model.npz"
        else:
            if self.operator is None:
                reference_config = self.config | {"time_h": 2.}
                reference_modulus = lambda radius: self.thermal.query(
                    "paste", np.clip(radius, 5.8, 6.02), 2.)["modulus_pa"]
                section = SectionPINN(reference_config, reference_modulus)
                family = EventPINN(section, self.thermal, self.knots, self.degree)
                self.operator = SpatialEventOperator([family])
            parts, stats = self.operator.independent(time)
            model = parts["uniform"]
            output.mkdir(parents=True, exist_ok=True)
            source = output / "model.npz"
            model.save(source)
            write_json(output / "training.json", stats | {"config": model.config,
                "rank": self.operator.preconditioner.shape[1],
                "spatial_weights": model.nweights,
                "spatial_reference_time_h": 2.})
            print(f"{self.mix}: independent event at {time:.8f} h", flush=True)
        self.models[time] = model
        self.sources[str(time)] = source.relative_to(ROOT).as_posix()
        return model


def compare_plan(nominal, independent, times, fractions, x):
    current = [nominal.event(t) for t in times]
    richer = [independent.event(t) for t in times]
    stages = []
    envelope = np.zeros(len(x))
    for index in range(len(times)):
        count = index + 1
        a = lambda at: surface_response(current[:count], fractions[:count], at)
        b = lambda at: surface_response(richer[:count], fractions[:count], at)
        delta = surface_extrema(lambda at: b(at) - a(at), x)
        event_delta = surface_response([richer[index]], [1.], x) - surface_response(
            [current[index]], [1.], x)
        envelope += abs(fractions[index]) * np.abs(event_delta)
        nominal_peak, richer_peak = surface_extrema(a, x), surface_extrema(b, x)
        stages.append({"time_h": float(times[index]),
            "nominal": nominal_peak, "independent": richer_peak,
            "nominal_margin_um_kpa": 2.2 - nominal_peak["absolute_peak_um_kpa"],
            "independent_margin_um_kpa": 2.2 - richer_peak["absolute_peak_um_kpa"],
            "maximum_surface_difference_um_kpa": delta["absolute_peak_um_kpa"],
            "difference_at_nominal_control_um_kpa": float(
                b(np.array([nominal_peak["absolute_peak_x_m"]]))[0]
                - a(np.array([nominal_peak["absolute_peak_x_m"]]))[0]),
            "sampled_sum_absolute_event_differences_um_kpa": float(envelope.max())})
    return {"times_h": list(times), "increments": list(fractions), "stages": stages,
            "nominal_feasible": all(row["nominal_margin_um_kpa"] >= -1e-7 for row in stages),
            "independent_feasible": all(row["independent_margin_um_kpa"] >= -1e-7 for row in stages)}


def ordering_checks(family, timetables, x):
    """Time extrema are polynomial-exact at the stated surface grid points."""
    polynomial = spline_polynomial(surface_time_spline(family, x))
    axis = PPoly(polynomial.c[:, :, len(x)//2], polynomial.x)
    minimum, _ = polynomial_range(polynomial)
    dominance, _ = polynomial_range(PPoly(axis.c[:, :, None] - polynomial.c, polynomial.x))
    derivative = spline_polynomial(surface_time_spline(family, x[len(x)//2+1:],
                                                       spatial_derivative=True))
    _, slope_maximum = polynomial_range(derivative)
    result = {"surface_interval_m": [-30., 30.], "surface_points": len(x),
        "surface_spacing_m": float(x[1] - x[0]), "time_interval_h": [0., 4.],
        "time_pieces": len(polynomial.x) - 1, "spatial_refinement_tolerance_m": 1e-6,
        "minimum_event_um_kpa": float(minimum.min()),
        "minimum_axis_dominance_um_kpa": float(dominance.min()),
        "maximum_right_half_spatial_slope_um_kpa_m": float(slope_maximum.max()),
        "timetables": []}
    for times in timetables:
        models = [family.event(t) for t in times]
        minima = [surface_extrema(lambda at, model=model:
            surface_response([model], [1.], at), x) for model in models]
        gaps = [surface_extrema(lambda at, a=a, b=b: surface_response([a, b], [1., -1.], at), x)
                for a, b in zip(models[:-1], models[1:])]
        result["timetables"].append({"times_h": times,
            "minimum_event_um_kpa": min(-row["uplift_um_kpa"] for row in minima),
            "minimum_adjacent_difference_um_kpa": min(-row["uplift_um_kpa"] for row in gaps),
            "maximum_abs_control_position_m": max(abs(row["peak_x_m"]) for row in minima)})
    return result


def assess_mix(mix):
    thermal = load_thermal(ROOT / f"results/pinn_support/{mix}/thermal_age/model.pt",
                           ROOT / "data/public_paste_benchmark")
    nominal = EventPINN.load(ROOT / f"results/pinn_support/{mix}/event/model.npz", thermal)
    independent = IndependentEvents(mix, nominal)
    x = np.linspace(-30., 30., 2401)
    result = {"mix_id": mix,
        "comparison": "Richer independent PINN events; empirical numerical sensitivity",
        "independent_config": independent.config,
        "early_task": compare_plan(nominal, independent, [.37, 1.73, 4.], [.2, .3, .5], x),
        "later_task": compare_plan(nominal, independent, [1.73, 3.1, 4.], [.2, .3, .5], x)}
    times = [[.37, 1.73, 4.], [1.73, 3.1, 4.]]
    if mix == "W40":
        axis = surface_time_spline(nominal, [0.])
        nominal_excess = lambda t: float(.2 * axis(t)[0] + .3 * axis(1.73)[0]
                                         + .5 * axis(4.)[0] - 2.2)
        nominal_root = brentq(nominal_excess, .37, 1.73, xtol=1e-10)
        fixed = surface_response([independent.event(1.73), independent.event(4.)],
                                 [.3, .5], np.array([0.]))[0]
        def independent_excess(t):
            return float(.2 * surface_response([independent.event(t)], [1.],
                                               np.array([0.]))[0] + fixed - 2.2)
        root = brentq(independent_excess, .37, 1.73, xtol=2.5e-4)
        lower, upper = root - .0005, root + .0005
        lower_value, upper_value = independent_excess(lower), independent_excess(upper)
        if lower_value <= 0 or upper_value > 0:
            raise RuntimeError("Independent time bracket does not straddle the active constraint")
        result["first_deadline"] = {
            "nominal_root_h": nominal_root,
            "independent_bracket_h": [lower, upper],
            "independent_endpoint_excess_um_kpa": [lower_value, upper_value],
            "nominal_active_slope_um_kpa_h": float(.2 * axis.derivative()(nominal_root)[0]),
            "at_nominal_boundary": compare_plan(nominal, independent,
                [nominal_root, 1.73, 4.], [.2, .3, .5], x),
            "at_independent_feasible_endpoint": compare_plan(nominal, independent,
                [upper, 1.73, 4.], [.2, .3, .5], x)}
        pressure = PressureEventPINN.load(ROOT / f"results/pinn_support/{mix}/pressure/model.npz",
                                          thermal)
        result["loaded_continuation"] = compare_plan(pressure.modes["uniform"], independent,
            [.37, 1.73, 3.1, 4.], [.2, .1, .2, .5], x)
        times.append([nominal_root, 1.73, 4.])
        radius = np.linspace(5.8, 6.02, 201)
        state = thermal.query("paste", radius, nominal_root)
        result["first_deadline"]["equivalent_reference_age_h"] = [
            float(state["reference_age_s"].min() / 3600),
            float(state["reference_age_s"].max() / 3600)]
        result["first_deadline"]["thermal_consistency"] = assess_events(
            thermal, [nominal_root, lower, upper])
    result["ordering"] = ordering_checks(nominal, times, x)
    result["independent_ordering_bounds"] = independent_ordering_bounds(result)
    result["independent_event_sources"] = independent.sources
    write_json(OUTPUT / f"{mix}.json", result)
    print(json.dumps({"mix": mix,
        "early_margin": [result["early_task"]["stages"][-1][name]
            for name in ("nominal_margin_um_kpa", "independent_margin_um_kpa")],
        "first_deadline": result.get("first_deadline", {}).get("independent_bracket_h")}), flush=True)


def independent_ordering_bounds(result):
    """Bound richer task fields on the grid using retained absolute differences."""
    bounds = {}
    for name in ("early_task", "later_task"):
        task = result[name]
        fractions = np.asarray(task["increments"])
        if np.any(fractions <= 0):
            raise ValueError("The event-envelope bound requires positive increments")
        nominal = next(row for row in result["ordering"]["timetables"]
                       if row["times_h"] == task["times_h"])
        envelope = task["stages"][-1]["sampled_sum_absolute_event_differences_um_kpa"]
        errors = envelope / fractions
        bounds[name] = {
            "times_h": task["times_h"],
            "surface_points": result["ordering"]["surface_points"],
            "weighted_absolute_envelope_um_kpa": envelope,
            "event_difference_bounds_um_kpa": errors.tolist(),
            "minimum_event_lower_bound_um_kpa": float(
                nominal["minimum_event_um_kpa"] - errors.max()),
            "minimum_adjacent_difference_lower_bound_um_kpa": float(
                nominal["minimum_adjacent_difference_um_kpa"]
                - (errors[:-1] + errors[1:]).max()),
        }
    return bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", choices=["W32", "W40", "W48"], nargs="+",
                        default=["W32", "W40", "W48"])
    parser.add_argument("--summarize-ordering", action="store_true",
                        help="Update ordering bounds from retained JSON without solving fields")
    args = parser.parse_args()
    if args.summarize_ordering:
        for mix in args.mix:
            path = OUTPUT / f"{mix}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["independent_ordering_bounds"] = independent_ordering_bounds(result)
            write_json(path, result)
            print(json.dumps({"mix": mix, **result["independent_ordering_bounds"]}))
        return
    torch.set_num_threads(2)
    with threadpool_limits(limits=4):
        for mix in args.mix:
            assess_mix(mix)
            gc.collect()


if __name__ == "__main__":
    main()
