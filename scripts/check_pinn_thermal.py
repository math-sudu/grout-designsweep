"""Independent quadrature and residual queries of the trained thermal PINN."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_thermal import load_thermal, derivative
from grouting_pino.paste_thermal import R_GAS


def assess(model, order=96):
    # Quadrature nodes are separate from all training collocation points.
    v, w = leggauss(order)
    v, w = (v + 1) / 2, w / 2
    times = np.linspace(0, model.time_scale / 3600, 41)
    stored, energies = {"time_h": times}, {}
    stats = {"quadrature_order": order, "domain": {}}
    device = next(model.parameters()).device
    for domain, (lo, hi) in model.radii.items():
        radius = lo + (hi - lo) * v
        weights = 2 * np.pi * radius * (hi-lo) * w
        fields = model.query(domain, radius[None,:], times[:,None])
        capacity = model.paste_properties["capacity_j_m3k"] if domain == "paste" else 2.2e6
        energies[domain] = capacity * ((fields["temperature_k"] - 293.15) @ weights)
        stored[domain + "_radius_m"] = radius
        stored.update({domain+"_"+key: value for key,value in fields.items()})
        z = torch.tensor(np.stack(np.meshgrid(v, v), -1).reshape(-1,2),
                         device=device,dtype=torch.float64,requires_grad=True)
        residuals = model.residuals(domain,z)
        stats["domain"][domain] = {key+"_normalized_rms": float(val.square().mean().sqrt().detach())
                                    for key,val in residuals.items()}
        for key, value in residuals.items():
            absolute = value.detach().abs().cpu().numpy()
            index = int(absolute.argmax())
            point = z[index].detach().cpu().numpy()
            stats["domain"][domain][key + "_normalized_p99"] = float(np.quantile(absolute, .99))
            stats["domain"][domain][key + "_normalized_max"] = float(absolute[index])
            stats["domain"][domain][key + "_maximum_at_radius_m_time_h"] = [
                float(lo + (hi - lo) * point[0]),
                float(model.time_scale / 3600 * point[1])]
        stats["domain"][domain]["residual_scales"] = {
            "flux_w_m2": model.flux_scale,
            "energy_w_m3": capacity * model.temperature_scale / model.time_scale}
        if domain == "paste":
            if model.state_representation == "equivalent_age":
                stats["domain"][domain]["residual_scales"]["reaction_definition"] = "a_e,t / A(T) - 1"
            else:
                stats["domain"][domain]["residual_scales"]["reaction_w_kg"] = model.heat_scale / model.time_scale
        stats["domain"][domain]["temperature_range_k"] = [float(fields["temperature_k"].min()),
                                                            float(fields["temperature_k"].max())]
        if domain == "paste":
            generated = model.paste_properties["cement_kg_m3"] * (fields["heat_j_kg"] @ weights)
            _, _, heat = model.fields(domain,z)
            qt = derivative(heat,z)[:,1]/model.time_scale
            stats["minimum_heat_rate_w_kg"] = float(qt.min().detach())
            stats["max_heat_j_kg"] = float(fields["heat_j_kg"].max())
            stats["modulus_4h_range_mpa"] = [float(fields["modulus_pa"][-1].min()/1e6),
                                               float(fields["modulus_pa"][-1].max()/1e6)]
            # Evaluate the actual measured rate via the tensor map, not an
            # approximation interpolated in heat coordinates.
            reaction = []
            losses = []
            for t in times:
                qfields = model.query("paste",radius[None,:],(v*t)[:,None])
                q = torch.tensor(qfields["heat_j_kg"],dtype=torch.float64,device=device)
                rate = model.material.rate(q).detach().cpu().numpy() * np.exp(
                    40000/R_GAS*(1/293.15-1/qfields["temperature_k"]))
                integrated = (w @ rate) * t*3600
                reaction.append(integrated)
                losses.append(np.max(np.abs(integrated-fields["heat_j_kg"][len(reaction)-1])))
            stats["reaction_integral_max_error_j_kg"] = float(max(losses))
            stats["reaction_integral_relative_to_final_heat"] = float(max(losses)/fields["heat_j_kg"][-1].max())
            stored["reaction_integral_j_kg"] = np.asarray(reaction)
    loss_integral = []
    jump_T, jump_J, robin_air, robin_soil = [], [], [], []
    for t in times:
        ti = model.query("lining",5.25,v*t)
        to = model.query("paste",6.02,v*t)
        power = 2*np.pi*(5.25*8*(ti["temperature_k"]-293.15)
                         +6.02*16*(to["temperature_k"]-293.15))
        loss_integral.append(float(w @ power*t*3600))
        li = model.query("lining",5.80,t)
        pa = model.query("paste",5.80,t)
        jump_T.append(float(li["temperature_k"]-pa["temperature_k"]))
        jump_J.append(float(li["flux_w_m2"]-pa["flux_w_m2"]))
        ai = model.query("lining",5.25,t)
        so = model.query("paste",6.02,t)
        robin_air.append(float(-ai["flux_w_m2"]-8*(ai["temperature_k"]-293.15)))
        robin_soil.append(float(so["flux_w_m2"]-16*(so["temperature_k"]-293.15)))
    stored["stored_energy_j_m"] = energies["lining"]+energies["paste"]
    stored["generated_heat_j_m"] = generated
    stored["boundary_loss_j_m"] = np.array(loss_integral)
    defect = stored["stored_energy_j_m"]+stored["boundary_loss_j_m"]-generated
    stored["energy_defect_j_m"] = defect
    stats.update(energy_final={"stored_j_m": float(stored["stored_energy_j_m"][-1]),
                              "generated_j_m":float(generated[-1]),
                              "lost_j_m":float(loss_integral[-1]),
                              "relative_defect":float(defect[-1]/generated[-1])},
                 max_energy_defect_over_final_heat=float(np.max(np.abs(defect))/generated[-1]),
                 max_interface_temperature_jump_k=float(np.max(np.abs(jump_T))),
                 max_interface_flux_jump_w_m2=float(np.max(np.abs(jump_J))),
                 max_air_robin_defect_w_m2=float(np.max(np.abs(robin_air))),
                 max_soil_robin_defect_w_m2=float(np.max(np.abs(robin_soil))))
    stats["maximum_energy_defect_time_h"] = float(times[np.argmax(np.abs(defect))])
    stats["evaluation"] = {"residual_points_per_domain": order**2,
        "radius_time_nodes": "tensor Gauss-Legendre, independent of training",
        "integral_times_h": times.tolist(), "initial_conditions": "imposed exactly"}
    return stats,stored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT/"results/pinn_support/W40/thermal_age")
    parser.add_argument("--order",type=int,default=96)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_thermal(args.directory/"model.pt",ROOT/"data/public_paste_benchmark", args.device)
    stats, fields = assess(model,args.order)
    (args.directory/"validation.json").write_text(json.dumps(stats,indent=2)+"\n")
    np.savez_compressed(args.directory/"fields.npz",**fields)
    print(json.dumps(stats,indent=2))


if __name__ == "__main__":
    main()
