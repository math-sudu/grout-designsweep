"""Check thermal invariants and equivalent-age integration at decision events."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.paste_thermal import R_GAS
from grouting_pino.pinn_thermal import load_thermal


def assess_events(model, times, order=96, radial_points=201):
    """Evaluate local state errors in seconds and at the event's heat scale."""
    nodes, weights = leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    radius = np.linspace(*model.radii["paste"], radial_points)
    origin = float(model.material.heat_age_s[0])
    reference = model.material.reference_temperature_k
    device = next(model.parameters()).device
    events = []
    for time_h in sorted(set(times)):
        state = model.query("paste", radius, time_h)
        temperatures = model.query("paste", radius[None, :], time_h * nodes[:, None])["temperature_k"]
        acceleration = np.exp(40000 / R_GAS * (1 / reference - 1 / temperatures))
        integrated_age = origin + time_h * 3600 * (weights @ acceleration)
        age = state["reference_age_s"]
        error = age - integrated_age
        minimum = age - origin - time_h * 3600
        q_integrated = model.material.heat_at_reference_age_s(
            torch.tensor(integrated_age, dtype=torch.float64, device=device)).detach().cpu().numpy()
        q_error = state["heat_j_kg"] - q_integrated
        worst = int(np.abs(error).argmax())
        low = int(minimum.argmin())
        minimum_temperature = {}
        for domain, (left, right) in model.radii.items():
            radial = np.linspace(left, right, radial_points)
            values = model.query(domain, radial, time_h)["temperature_k"]
            index = int(values.argmin())
            minimum_temperature[domain] = {
                "value_k": float(values[index]),
                "excess_over_reservoir_k": float(values[index] - reference),
                "radius_m": float(radial[index])}
        row = {
            "time_h": time_h,
            "minimum_temperature": minimum_temperature,
            "equivalent_age_range_h": [float(age.min() / 3600), float(age.max() / 3600)],
            "minimum_age_excess_s": float(minimum[low]),
            "minimum_age_excess_radius_m": float(radius[low]),
            "maximum_absolute_age_integral_error_s": float(abs(error[worst])),
            "signed_age_integral_error_at_maximum_s": float(error[worst]),
            "age_integral_error_radius_m": float(radius[worst]),
            "maximum_absolute_heat_integral_error_j_kg": float(np.abs(q_error).max()),
            "maximum_relative_heat_integral_error": float(
                np.max(np.abs(q_error) / q_integrated)) if time_h else 0.,
            "modulus_range_mpa": [float(state["modulus_pa"].min() / 1e6),
                                  float(state["modulus_pa"].max() / 1e6)]}
        events.append(row)
    return {
        "mix_id": model.config["mix_id"],
        "thermal_config": model.config["thermal"],
        "initial_reference_age_h": origin / 3600,
        "evaluation": {"radial_points": radial_points, "time_quadrature_order": order,
            "age_relation": "a_e = a_0 + integral A(T) dt",
            "lower_bounds": "T >= T_ref and a_e >= a_0 + t for the specified initial and reservoir temperatures",
            "heat_error_denominator": "Q(a_0 + integral A(T) dt) at the same radius and event"},
        "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--order", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_thermal(args.directory / "model.pt", ROOT / "data/public_paste_benchmark")
    result = assess_events(model, args.times, args.order)
    output = args.output or args.directory / "event_consistency.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["events"]:
        print(json.dumps({key: row[key] for key in (
            "time_h", "minimum_age_excess_s", "maximum_absolute_age_integral_error_s",
            "maximum_relative_heat_integral_error")}), flush=True)


if __name__ == "__main__":
    main()
