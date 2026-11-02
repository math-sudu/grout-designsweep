"""Execute half-load, hold, reload and a nonzero checkpoint continuation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from check_pinn_mechanics import load_section
from grouting_pino.pinn_history import PINNHistory
from grouting_pino.pinn_thermal import load_thermal
from grouting_pino.pinn_mechanics import PRESSURE,annulus_points
from grouting_pino.pinn_pressure import PressureEventPINN
from grouting_pino.pinn_design import surface_extrema


def field_difference(a,b,probes):
    errors={"displacement_max_m":0.,"stress_max_pa":0.}
    for domain,xy in probes.items():
        left=a.query(domain,xy)
        right=b.query(domain,xy)
        for key in left:
            group="displacement_max_m" if key.startswith("u") else "stress_max_pa"
            errors[group]=max(errors[group],float(np.max(np.abs(left[key]-right[key]))))
    return errors


def main():
    torch.set_num_threads(4)
    folder=ROOT/"results/pinn_support/W40"
    output=folder/"history"
    output.mkdir(parents=True,exist_ok=True)
    thermal=load_thermal(folder/"thermal_age/model.pt",ROOT/"data/public_paste_benchmark")
    early=load_section(folder/"mechanics_age/t0_refined")
    late=load_section(folder/"mechanics_age/t4_refined")
    x=np.linspace(-30,30,1201)
    surface=np.c_[x,np.zeros_like(x)]
    probes={"soil":annulus_points(np.array([6.05,8.,15.])[:,None],
                                   np.array([0.,np.pi/2,np.pi])[None,:]),
            "paste":annulus_points(5.91,np.linspace(0,np.pi,19)),
            "lining":annulus_points(5.525,np.linspace(0,np.pi,19))}
    h=PINNHistory(thermal)
    h.apply(early,0.5)
    frames={"half_load":h.query("soil",surface)}
    initial=PINNHistory(thermal)
    initial.apply(early,0.5)
    h.advance(2.0)
    frames["hold"]=h.query("soil",surface)
    h.save(output/"loaded_checkpoint.npz")
    resumed=PINNHistory.load(output/"loaded_checkpoint.npz",ROOT/"data/public_paste_benchmark")
    hold_error=field_difference(initial,h,probes)
    restart_initial_error=field_difference(h,resumed,probes)
    for path in (h,resumed):
        path.advance(4.0)
        path.apply(late,1.0)
    frames["reload"]=h.query("soil",surface)
    restart_final_error=field_difference(h,resumed,probes)
    split=PINNHistory(thermal)
    split.apply(early,0.2)
    split.apply(early,0.5)
    split.advance(4.)
    split.apply(late,0.8)
    split.apply(late,1.)
    split_error=field_difference(h,split,probes)
    # Reopening the saved equilibrium after a future event cannot change it.
    old=PINNHistory.load(output/"loaded_checkpoint.npz",ROOT/"data/public_paste_benchmark")
    future_error=field_difference(initial,old,probes)
    old_material=old.material_state()
    new_material=h.material_state()
    h.save(output/"final_checkpoint.npz")
    # The historical stress differs from reconstructing D(4h):epsilon(u).
    xy=probes["paste"]
    early_f=early.query("paste",xy,True)
    late_f=late.query("paste",xy,True)
    E0=early.modulus("paste",xy)
    E4=late.modulus("paste",xy)
    memory=h.query("paste",xy)
    replacement={key:.5*early_f[key]*E4/E0+.5*late_f[key] for key in ("sxx","syy","sxy","szz")}
    stress_difference=max(float(np.max(np.abs(memory[key]-replacement[key]))) for key in replacement)
    fields={"surface_x_m":x}
    for label,frame in frames.items():
        fields.update({label+"_"+key:val for key,val in frame.items()})
    for domain,xy in probes.items():
        fields[domain+"_xy_m"]=xy
        fields.update({domain+"_final_"+key:val for key,val in h.query(domain,xy).items()})
    np.savez_compressed(output/"fields.npz",**fields)
    summary={"events":[{"time_h":0.,"total_fraction":.5},
                        {"time_h":2.,"total_fraction":.5,"operation":"hold and checkpoint"},
                        {"time_h":4.,"total_fraction":1.}],
             "mechanical_models":["mechanics_age/t0_refined/model.npz","mechanics_age/t4_refined/model.npz"],
             "hold_field_difference":hold_error,
             "restart_initial_field_difference":restart_initial_error,
             "restart_final_field_difference":restart_final_error,
             "split_field_difference":split_error,
             "future_event_effect_on_checkpoint":future_error,
             "checkpoint_heat_range_j_kg":[float(old_material["paste_heat_j_kg"].min()),
                                               float(old_material["paste_heat_j_kg"].max())],
             "final_heat_range_j_kg":[float(new_material["paste_heat_j_kg"].min()),
                                          float(new_material["paste_heat_j_kg"].max())],
             "stress_memory_vs_current_stiffness_max_pa":stress_difference,
             "surface_peak_um_kpa":{label:float(np.max(-frame["uy"])*1e9/PRESSURE)
                                     for label,frame in frames.items()}}
    (output/"validation.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def check_pressure_history():
    """Replay a prescribed loading task and restart from its loaded state."""
    torch.set_num_threads(4)
    folder = ROOT / "results/pinn_support/W40"
    output = folder / "history_pressure"
    output.mkdir(parents=True, exist_ok=True)
    thermal = load_thermal(folder / "thermal_age/model.pt", ROOT / "data/public_paste_benchmark")
    family = PressureEventPINN.load(folder / "pressure/model.npz", thermal)
    history, split = PINNHistory(thermal), PINNHistory(thermal)
    for current in (history, split):
        current.bind_family(family)
        current.advance(.37)
    split.apply_modes({"uniform": .1})
    for current in (history, split):
        current.apply_modes({"uniform": .2})
    probes = {"soil": annulus_points(np.array([6.05, 8., 15.])[:, None],
                                     np.linspace(-np.pi, np.pi, 19)[None, :]),
              "paste": annulus_points(5.91, np.linspace(-np.pi, np.pi, 37)),
              "lining": annulus_points(5.525, np.linspace(-np.pi, np.pi, 37))}
    x = np.linspace(-30., 30., 1201)
    surface = np.c_[x, np.zeros_like(x)]
    frames, stages = {}, []

    def retain_stage(label, current):
        frames[label] = current.query("soil", surface)
        peak = surface_extrema(lambda at: -current.query("soil", np.c_[at, np.zeros_like(at)])["uy"] * 1e9 / PRESSURE)
        stages.append({"label": label, "time_h": current.time_h,
                       "cumulative_pressure_coefficients": current.load_coefficients.copy(),
                       "mean_pressure_fraction": current.load_fraction, **peak,
                       "settlement_margin_um_kpa": 2.2 - peak["peak_um_kpa"],
                       "absolute_margin_um_kpa": 2.2 - peak["absolute_peak_um_kpa"]})

    retain_stage("initial", history)
    for current in (history, split):
        current.advance(1.)
    history.save(output / "loaded_checkpoint.npz")
    resumed = PINNHistory.load(output / "loaded_checkpoint.npz", ROOT / "data/public_paste_benchmark")
    retain_stage("hold", history)
    restart_initial = field_difference(history, resumed, probes)
    for current in (history, split, resumed):
        current.advance(1.73)
        current.apply_modes({"uniform": .45, "sector": -.2})
    retain_stage("intermediate", history)
    for current in (history, split, resumed):
        current.advance(4.)
        current.apply_modes({"uniform": 1., "sector": -.5})
    retain_stage("complete", history)
    history.save(output / "final_checkpoint.npz")
    old = PINNHistory.load(output / "loaded_checkpoint.npz", ROOT / "data/public_paste_benchmark")
    future_error = max(float(np.max(np.abs(old.query("soil", surface)[key] - value)))
                       for key, value in frames["initial"].items())
    # The same final pressure applied at the final time is a distinct history.
    delayed = PINNHistory(thermal)
    delayed.bind_family(family)
    delayed.advance(4.)
    delayed.apply_modes({"uniform": 1., "sector": -.5})
    retain_stage("all_at_completion", delayed)
    fields = {"surface_x_m": x}
    for label, frame in frames.items():
        fields.update({label + "_" + key: value for key, value in frame.items()})
    for domain, xy in probes.items():
        fields[domain + "_xy_m"] = xy
        fields.update({domain + "_final_" + key: value for key, value in history.query(domain, xy).items()})
    np.savez_compressed(output / "fields.npz", **fields)
    summary = {"task_source": "prescribed numerical loading task", "mix_id": "W40",
        "family": "results/pinn_support/W40/pressure/model.npz", "sector_angle_deg": family.angle_deg,
        "absolute_displacement_limit_um_kpa": 2.2, "stages": stages, "event_increments": history.mode_events,
        "hold_surface_max_difference": max(float(np.max(np.abs(frames["initial"][key] - frames["hold"][key]))) for key in frames["initial"]),
        "restart_initial_field_difference": restart_initial,
        "restart_final_field_difference": field_difference(history, resumed, probes),
        "split_field_difference": field_difference(history, split, probes),
        "future_event_effect_on_checkpoint": future_error,
        "different_history_field_difference": field_difference(history, delayed, probes),
        "all_stages_feasible": all(row["absolute_margin_um_kpa"] >= 0 for row in stages)}
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-pressure", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.continuous_pressure:
            check_pressure_history()
        else:
            main()
