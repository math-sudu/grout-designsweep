"""Train the nominal radial PINN and retain the actual model and field queries."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_thermal import ThermalPINN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pinn_support.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mix", choices=["W32", "W40", "W48"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.mix is not None:
        cfg["mix_id"] = args.mix
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.refine:
        cfg["thermal"].update(width=40, domain_points=1536, boundary_points=384)
    args.output = args.output or ROOT / f"results/pinn_support/{cfg['mix_id']}/thermal_age"
    c = cfg["thermal"]
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    model = ThermalPINN(cfg, ROOT / "data/public_paste_benchmark").double().to(cfg["device"])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "model.pt"
    if args.resume:
        model.load_state_dict(torch.load(path, map_location=cfg["device"], weights_only=True)["state_dict"])
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg["seed"])

    def sample():
        result = {d: engine.draw(c["domain_points"]).double().to(cfg["device"]).requires_grad_()
                  for d in model.radii}
        result["times"] = engine.draw(c["boundary_points"])[:, 1].double().to(cfg["device"])
        return result

    def save():
        torch.save({"config": cfg, "state_dict": model.state_dict()}, path)

    start = time.monotonic()
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    samples = sample()
    for step in range(c["adam_steps"]):
        if step % 100 == 0:
            samples = sample()
        optimizer.zero_grad(set_to_none=True)
        losses = model.losses(samples)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step + 1 == c["adam_steps"]:
            row = {"optimizer": "adam", "step": step + 1, "loss": loss.item(),
                   "elapsed_s": time.monotonic() - start}
            history.append(row)
            print(json.dumps(row), flush=True)
            save()
    # Fixed points/weights throughout line searches.
    samples = sample()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=c["lbfgs_steps"],
                                  max_eval=c["lbfgs_steps"] * 5 // 4,
                                  tolerance_grad=1e-10, tolerance_change=1e-13,
                                  history_size=50, line_search_fn="strong_wolfe")
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        loss = sum(model.losses(samples).values())
        loss.backward()
        calls += 1
        if calls % 250 == 0:
            print(json.dumps({"optimizer": "lbfgs", "closure": calls,
                              "loss": loss.item(), "elapsed_s": time.monotonic() - start}), flush=True)
        return loss

    optimizer.step(closure)
    save()
    final = {key: val.item() for key, val in model.losses(sample()).items()}
    summary = {"config": cfg, "training": history, "independent_losses": final,
               "elapsed_s": time.monotonic() - start}
    (args.output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
