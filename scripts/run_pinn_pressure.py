"""Train and assess continuous pressure modes over the complete section."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_event import EventPINN, clamped_knots, train_event_family
from grouting_pino.pinn_mechanics import (SectionPINN, PRESSURE, DOMAINS,
    annulus_points, sample_domains)
from grouting_pino.pinn_pressure import PressureEventPINN, PRESSURE_MODES
from grouting_pino.pinn_design import surface_extrema, surface_response
from run_pinn_design import EventFields, write_json
from check_pinn_mechanics import constitutive


def assess_pressure(section):
    """Balance and jump checks on both halves, retaining the endpoint profiles."""
    config = section.config | {"domain_points": {"soil": 1800, "paste": 900, "lining": 900}}
    points = sample_domains(config, config["seed"] + 8901, full_domain=True)
    result, fields = {"domain": {}, "interfaces": {}}, {}
    for domain, xy in points.items():
        f = section.query(domain, xy, True)
        calc = constitutive(section, domain, xy, f)
        eq = np.c_[f["sxx_x"] + f["sxy_y"], f["sxy_x"] + f["syy_y"]]
        con = np.c_[f["sxx"] - calc["sxx"], f["syy"] - calc["syy"],
                    np.sqrt(2) * (f["sxy"] - calc["sxy"])]
        result["domain"][domain] = {"equilibrium_rms_pa_m": float(np.sqrt(np.mean(eq**2))),
            "constitutive_rms_pa": float(np.sqrt(np.mean(con**2))),
            "stress_rms_pa": float(np.sqrt(np.mean(np.c_[f["sxx"], f["syy"], f["sxy"]]**2)))}
        for name, residual, scale in [("equilibrium", eq, PRESSURE/6.02),
                                      ("constitutive", con, PRESSURE)]:
            absolute = np.linalg.norm(residual, axis=1) / scale
            index = int(absolute.argmax())
            result["domain"][domain].update({
                name+"_normalized_p99": float(np.quantile(absolute, .99)),
                name+"_normalized_max": float(absolute[index]),
                name+"_maximum_xy_m": xy[index].tolist()})
        fields[domain + "_xy_m"] = xy
        fields.update({domain + "_" + key: f[key] for key in ("ux", "uy", "sxx", "syy", "sxy", "szz")})
    theta = -np.pi + 2 * np.pi * (np.arange(1440) + .37) / 1440
    nx, ny = np.sin(theta), np.cos(theta)
    distance = np.minimum(np.abs(theta), np.abs(theta - np.deg2rad(config["sector_angle_deg"])))
    for radius, inner, outer in [(5.8, "lining", "paste"), (6.02, "paste", "soil")]:
        xy = annulus_points(radius, theta)
        a, b = section.query(inner, xy), section.query(outer, xy)
        tx = (a["sxx"] - b["sxx"]) * nx + (a["sxy"] - b["sxy"]) * ny
        ty = (a["sxy"] - b["sxy"]) * nx + (a["syy"] - b["syy"]) * ny
        target = -PRESSURE * section.pressure(theta) if inner == "paste" else np.zeros_like(theta)
        radial, tangent = tx * nx + ty * ny, tx * ny - ty * nx
        name = inner + "_" + outer
        row = {"displacement_jump_max_m": float(np.max(np.hypot(a["ux"] - b["ux"], a["uy"] - b["uy"]))),
               "traction_error_rms_pa": float(np.sqrt(np.mean((tx - target * nx)**2 + (ty - target * ny)**2))),
               "traction_error_max_pa": float(np.max(np.hypot(tx - target*nx, ty - target*ny))),
               "radial_jump_mean_pa": float(np.mean(radial)), "prescribed_mean_pa": float(np.mean(target))}
        if inner == "paste":
            for label, mask in [("within_10deg_of_endpoint", distance < np.deg2rad(10.)),
                                ("away_from_endpoints", distance >= np.deg2rad(10.))]:
                row[label] = {"radial_error_rms_pa": float(np.sqrt(np.mean((radial[mask] - target[mask])**2))),
                              "tangential_rms_pa": float(np.sqrt(np.mean(tangent[mask]**2)))}
        result["interfaces"][name] = row
        fields.update({name + "_theta_rad": theta, name + "_radial_jump_pa": radial,
                       name + "_target_pa": target, name + "_tangential_jump_pa": tangent})
    xy = np.c_[np.r_[np.full(300, -30.), np.full(300, 30.)], np.tile(np.linspace(-60., 0., 300), 2)]
    side = section.query("soil", xy)
    top = section.query("soil", np.c_[np.linspace(-30., 30., 601), np.zeros(601)])
    result["boundary"] = {"both_side_horizontal_displacement_max_m": float(np.max(np.abs(side["ux"]))),
        "both_side_shear_rms_pa": float(np.sqrt(np.mean(side["sxy"]**2))),
        "surface_traction_rms_pa": float(np.sqrt(np.mean(top["sxy"]**2 + top["syy"]**2)))}
    nodes, weights = np.polynomial.legendre.leggauss(80)
    reaction = np.zeros(2)
    for xy, normal in [(np.c_[30 * nodes, np.full(80, -60.)], (0., -1.)),
                       (np.c_[30 * nodes, np.zeros(80)], (0., 1.)),
                       (np.c_[np.full(80, -30.), 30 * nodes - 30.], (-1., 0.)),
                       (np.c_[np.full(80, 30.), 30 * nodes - 30.], (1., 0.))]:
        f = section.query("soil", xy)
        reaction += 30 * np.array([weights @ (f["sxx"] * normal[0] + f["sxy"] * normal[1]),
                                  weights @ (f["sxy"] * normal[0] + f["syy"] * normal[1])])
    f = section.query("lining", annulus_points(5.25, theta))
    reaction -= 2 * np.pi * 5.25 * np.array([np.mean(f["sxx"] * nx + f["sxy"] * ny),
                                            np.mean(f["sxy"] * nx + f["syy"] * ny)])
    angle = np.deg2rad(config["sector_angle_deg"])
    applied = -PRESSURE * 6.02 * section.coefficients["sector"] * np.array([1 - np.cos(angle), np.sin(angle)])
    result["force_balance"] = {"applied_n_m": applied.tolist(), "reaction_n_m": reaction.tolist(),
        "defect_n_m": (applied + reaction).tolist(),
        "relative_defect": (float(np.linalg.norm(applied + reaction) / np.linalg.norm(applied))
                            if np.linalg.norm(applied) else None)}
    result["evaluation"] = {"domain_points": config["domain_points"],
        "domain_seed": config["seed"]+8901, "interface_points": len(theta),
        "equilibrium_scale_pa_m": PRESSURE/6.02, "constitutive_scale_pa": PRESSURE,
        "independent_of_training": True}
    return result, fields


def evaluate(family, directory, mix):
    thermal = family.thermal
    reference = EventPINN.load(ROOT / f"results/pinn_support/{mix}/event/model.npz", thermal)
    times = [0., .5, 1.5, 2., 2.3984375, 3., 3.5, 4.]
    reductions = [1., .8, .5, .2]
    x = np.linspace(-30., 30., 1201)
    rows, profiles, mode_profiles, uniform_errors = [], [], [], []
    for t in times:
        modes = {name: model.event(t) for name, model in family.modes.items()}
        components = np.array([surface_response([modes[name]], [1.], x) for name in PRESSURE_MODES])
        mode_profiles.append(components)
        baseline = surface_response([reference.event(t)], [1.], x)
        uniform_errors.append({"time_h": t, "surface_relative_l2": float(np.linalg.norm(components[0] - baseline) / np.linalg.norm(baseline)),
            "surface_max_difference_um_kpa": float(np.max(np.abs(components[0] - baseline)))})
        for k in reductions:
            coefficients = {"uniform": 1., "sector": k - 1.}
            section = family.event(t, coefficients)
            peak = surface_extrema(lambda at: surface_response([section], [1.], at), x)
            profile = components[0] + (k - 1) * (components[1] + components[2])
            rows.append({"time_h": t, "k": k, **peak,
                "axis_um_kpa": float(profile[len(x) // 2]),
                "mean_pressure_fraction": family.mean_pressure_fraction(coefficients)})
            profiles.append(profile)
    assessment, fields = assess_pressure(family.event(2., {"uniform": 1., "sector": -.8}))
    write_json(directory / "evaluation.json", {"mix_id": mix, "sector_angle_deg": family.angle_deg,
        "surfaces": rows, "uniform_recovery": uniform_errors,
        "field_time_h": 2., "field_k": .2, "full_section": assessment})
    np.savez_compressed(directory / "fields.npz", **fields)
    np.savez_compressed(directory / "surface.npz", x_m=x, time_h=times, k=reductions,
                        surface_um_kpa=profiles, mode_surface_um_kpa=mode_profiles)
    print(json.dumps({"mix_id": mix, "at_2h": [row for row in rows if row["time_h"] == 2.],
                      "max_uniform_recovery_l2": max(row["surface_relative_l2"] for row in uniform_errors)}), flush=True)


def main():
    settings_all = json.loads((ROOT / "configs/pinn_support.json").read_text())
    settings, pressure = settings_all["event"], settings_all["pressure"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=["W32", "W40", "W48"], default="W40")
    parser.add_argument("--angular-modes", type=int, default=pressure["angular_modes"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    directory = args.output or ROOT / f"results/pinn_support/{args.mix}/pressure"
    directory.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        series = EventFields(args.mix)
        if args.train or not (directory / "model.npz").exists():
            breaks = np.unique(np.r_[np.arange(0, 4, settings["break_step_h"]), 4.])
            knots = clamped_knots(breaks, settings["degree"], settings["continuity"])
            config = series.config | {"time_h": 2., "sector_angle_deg": pressure["sector_angle_deg"],
                "angular_modes": args.angular_modes,
                "thermal_model": series.thermal_path.relative_to(ROOT).as_posix()}
            modes = {}
            for name in PRESSURE_MODES:
                section = SectionPINN(config | {"pressure_mode": name,
                    "reflection": "odd" if name == "sector_odd" else "even"},
                    lambda r: series.thermal.query("paste", np.clip(r, 5.8, 6.02), 2.)["modulus_pa"])
                modes[name] = EventPINN(section, series.thermal, knots, settings["degree"])
            kwargs = {"quadrature_order": settings["quadrature_order"], "linear_rtol": settings["linear_rtol"]}
            even = train_event_family(modes["uniform"], breaks,
                additional_models=[modes["sector_even"]], **kwargs)
            odd = train_event_family(modes["sector_odd"], breaks, **kwargs)
            family = PressureEventPINN(modes)
            family.save(directory / "model.npz")
            write_json(directory / "training.json", {"config": config, "even": even, "odd": odd,
                "elapsed_s": even["elapsed_s"] + odd["elapsed_s"]})
        else:
            family = PressureEventPINN.load(directory / "model.npz", series.thermal)
            if family.modes["uniform"].section.config["angular_modes"] != args.angular_modes:
                raise ValueError("Stored angular order differs; use its setting or --train")
        evaluate(family, directory, args.mix)


if __name__ == "__main__":
    main()
