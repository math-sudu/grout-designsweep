"""Train a continuous event PINN and compare held-out full neural fields."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_event import EventPINN, clamped_knots, train_event_family
from grouting_pino.pinn_mechanics import (SectionPINN, DOMAINS, PRESSURE,
                                         annulus_points, sample_domains)
from grouting_pino.pinn_design import feasible_intervals, surface_maximum, surface_response
from run_pinn_design import EventFields, write_json
from check_pinn_mechanics import constitutive


def field_comparison(model, reference):
    config = model.config | {"domain_points": {"soil": 1200, "paste": 600, "lining": 600}}
    points = sample_domains(config, config["seed"] + 8901)
    result = {"domain": {}, "interfaces": {}}
    for domain, xy in points.items():
        field, baseline = [m.query(domain, xy, True) for m in (model, reference)]
        row = {}
        for name, keys in [("displacement", ["ux", "uy"]),
                           ("stress", ["sxx", "syy", "sxy", "szz"])]:
            actual = np.stack([field[key] for key in keys], axis=1)
            expected = np.stack([baseline[key] for key in keys], axis=1)
            row[name + "_relative_l2"] = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            row[name + "_max_abs_difference"] = float(np.max(np.abs(actual - expected)))
        for prefix, source in [("event", model), ("independent", reference)]:
            f = field if prefix == "event" else baseline
            calc = constitutive(source, domain, xy, f)
            eq = np.c_[f["sxx_x"] + f["sxy_y"], f["sxy_x"] + f["syy_y"]]
            con = np.c_[f["sxx"] - calc["sxx"], f["syy"] - calc["syy"],
                        np.sqrt(2) * (f["sxy"] - calc["sxy"])]
            row[prefix + "_equilibrium_rms_pa_m"] = float(np.sqrt(np.mean(eq**2)))
            row[prefix + "_constitutive_rms_pa"] = float(np.sqrt(np.mean(con**2)))
        result["domain"][domain] = row
    angle = np.pi * (np.arange(601) + .37) / 601
    nx, ny = np.sin(angle), np.cos(angle)
    for radius, inner, outer in [(5.8, "lining", "paste"), (6.02, "paste", "soil")]:
        xy = annulus_points(radius, angle)
        row = {}
        for prefix, source in [("event", model), ("independent", reference)]:
            a, b = source.query(inner, xy), source.query(outer, xy)
            tx = (a["sxx"] - b["sxx"]) * nx + (a["sxy"] - b["sxy"]) * ny
            ty = (a["sxy"] - b["sxy"]) * nx + (a["syy"] - b["syy"]) * ny
            target = -PRESSURE if inner == "paste" else 0.
            row[prefix + "_traction_error_rms_pa"] = float(np.sqrt(np.mean(
                (tx - target * nx)**2 + (ty - target * ny)**2)))
            row[prefix + "_displacement_jump_max_m"] = float(np.max(np.hypot(
                a["ux"] - b["ux"], a["uy"] - b["uy"])))
        result["interfaces"][inner + "_" + outer] = row
    x = np.linspace(-30., 30., 1201)
    for prefix, source in [("event", model), ("independent", reference)]:
        f = source.query("soil", np.c_[x, np.zeros_like(x)])
        result[prefix + "_surface_traction_rms_pa"] = float(np.sqrt(np.mean(f["sxy"]**2 + f["syy"]**2)))
    return result


def evaluate(model, series, training):
    retained = json.loads((ROOT / f"results/pinn_support/design/{series.mix}.json").read_text())
    times = [row["time_h"] for row in retained["events"]]
    x = np.linspace(-30., 30., 1201)
    records, surface_fields = [], []
    for t in times:
        event, reference = model.event(t), series.event(t)
        peak = surface_maximum(lambda at: surface_response([event], [1.], at))
        independent = surface_maximum(lambda at: surface_response([reference], [1.], at))
        response = surface_response([event], [1.], x)
        baseline = surface_response([reference], [1.], x)
        records.append({"time_h": t,
                        "training_time_distance_h": float(np.min(np.abs(np.array(training["time_h"]) - t))),
                        "peak_um_kpa": peak["peak_um_kpa"], "peak_x_m": peak["peak_x_m"],
                        "independent_peak_um_kpa": independent["peak_um_kpa"],
                        "surface_relative_l2": float(np.linalg.norm(response - baseline) / np.linalg.norm(baseline)),
                        "surface_max_difference_um_kpa": float(np.max(np.abs(response - baseline)))})
        surface_fields.append(response)
    def excess(t):
        event = model.event(t)
        return surface_maximum(lambda at: surface_response([event], [1.], at))["peak_um_kpa"] - retained["limit_um_kpa"]

    handover = feasible_intervals(excess, np.linspace(0, 4, 9), retained["crossing_resolution_h"])
    critical = retained["handover"]["intervals_h"][0][0]
    held = sorted(set([0., .5, 1.5, critical, 3.5, 4.]))
    fields = []
    for t in held:
        print(f"Held-out mixed field: {series.mix}, {t:g} h", flush=True)
        fields.append({"time_h": t, **field_comparison(model.event(t), series.event(t))})
    return {"mix_id": series.mix, "surfaces": records, "fields": fields,
            "handover": handover, "independent_handover": retained["handover"]}, dict(
                time_h=times, x_m=x, surface_um_kpa=surface_fields)


def main():
    settings = json.loads((ROOT / "configs/pinn_support.json").read_text())["event"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=["W32", "W40", "W48"], default="W40")
    parser.add_argument("--break-step", type=float, default=settings["break_step_h"])
    parser.add_argument("--quadrature-order", type=int, default=settings["quadrature_order"])
    parser.add_argument("--linear-rtol", type=float, default=settings["linear_rtol"])
    parser.add_argument("--continuity", type=int, choices=[0, 1, 2], default=settings["continuity"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Compare the retained independent event fields")
    args = parser.parse_args()
    directory = args.output or ROOT / f"results/pinn_support/{args.mix}/event"
    directory.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        series = EventFields(args.mix)
        breaks = np.unique(np.r_[np.arange(0, 4, args.break_step), 4.])
        config = series.config | {"time_h": 2.,
            "thermal_model": series.thermal_path.relative_to(ROOT).as_posix()}
        if args.train or not (directory / "model.npz").exists():
            section = SectionPINN(config, lambda r: series.thermal.query(
                "paste", np.clip(r, 5.8, 6.02), 2.)["modulus_pa"])
            knots = clamped_knots(breaks, settings["degree"], args.continuity)
            model = EventPINN(section, series.thermal, knots, degree=settings["degree"])
            training = train_event_family(model, breaks, args.quadrature_order, args.linear_rtol)
            training.update(config=config, break_step_h=args.break_step, continuity=args.continuity)
            model.save(directory / "model.npz")
            write_json(directory / "training.json", training)
            print(json.dumps({key: training[key] for key in ["elapsed_s", "linear_iterations",
                                                            "relative_normal_residual"]}), flush=True)
        else:
            model = EventPINN.load(directory / "model.npz", series.thermal)
            training = json.loads((directory / "training.json").read_text())
            if (not np.array_equal(model.knots, clamped_knots(breaks, settings["degree"], args.continuity))
                    or training["quadrature_order"] != args.quadrature_order):
                raise ValueError("Stored event basis differs; select its configuration or use --train")
            print(f"Loaded continuous event field: {directory / 'model.npz'}", flush=True)
        if args.evaluate:
            comparison, surfaces = evaluate(model, series, training)
            write_json(directory / "evaluation.json", comparison)
            np.savez_compressed(directory / "surface.npz", **surfaces)
            print(json.dumps({"mix_id": args.mix, "handover": comparison["handover"]["intervals_h"],
                              "max_surface_relative_l2": max(row["surface_relative_l2"]
                                                             for row in comparison["surfaces"])}), flush=True)


if __name__ == "__main__":
    main()
