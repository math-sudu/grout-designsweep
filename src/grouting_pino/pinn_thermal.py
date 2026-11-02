"""Mixed radial thermal/reaction PINN in SI units, with fixed measured kinetics."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from grouting_pino.paste_reference import load_paste_references
from grouting_pino.pinn_material import PasteStateMap
from grouting_pino.paste_thermal import PasteThermalProperties, R_GAS


def derivative(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


class DenseNet(nn.Sequential):
    def __init__(self, inputs, outputs, width, depth):
        layers = []
        for n in range(depth):
            layers.extend([nn.Linear(inputs if n == 0 else width, width), nn.Tanh()])
        layers.append(nn.Linear(width, outputs))
        super().__init__(*layers)
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self[-1].weight)


class ThermalPINN(nn.Module):
    """Nominal initial state at 293.15 K and zero observed heat increment.

    Inputs are each layer's local radius and elapsed time, both in [0, 1].
    The age representation imposes the initial state and the physical lower
    bounds for the prescribed reference-temperature reservoirs. Saved heat-state
    models remain readable for comparison with the earlier calculations.
    Predictions outside the trained time or material domain are rejected.
    """

    radii = {"lining": (5.25, 5.80), "paste": (5.80, 6.02)}

    def __init__(self, config, data_dir):
        super().__init__()
        self.config = config
        c = config["thermal"]
        self.state_representation = c.get("state_representation", "heat")
        if self.state_representation not in ("heat", "equivalent_age"):
            raise ValueError("Unknown thermal state representation")
        self.material = PasteStateMap(load_paste_references(Path(data_dir))[config["mix_id"]])
        self.networks = nn.ModuleDict({
            name: DenseNet(2, 3 if name == "paste" else 2, c["width"], c["depth"])
            for name in self.radii})
        self.time_scale = c["end_time_h"] * 3600
        self.temperature_scale = c["temperature_scale_k"]
        self.heat_scale = c["heat_scale_j_kg"]
        self.flux_scale = c["flux_scale_w_m2"]
        self.paste_properties = PasteThermalProperties().volume_properties(
            self.material.water_cement_ratio)
        if self.state_representation == "equivalent_age":
            with torch.no_grad():
                self.networks["paste"][-1].bias[2] = -3.

    def state(self, domain, z):
        raw = self.networks[domain](2 * z - 1)
        tau = z[:, 1]
        J = self.flux_scale * tau * raw[:, 1]
        if self.state_representation == "equivalent_age":
            T = 293.15 + self.temperature_scale * tau * nn.functional.softplus(raw[:, 0])
            age = (self.material.heat_age_s[0] + self.time_scale
                   * (tau + tau.square() * nn.functional.softplus(raw[:, 2]))) if domain == "paste" else None
            q = self.material.heat_at_reference_age_s(age) if age is not None else None
        else:
            T = 293.15 * torch.exp(tau * self.temperature_scale / 293.15 * raw[:, 0])
            q = self.heat_scale * tau * nn.functional.softplus(raw[:, 2]) if domain == "paste" else None
            age = None
        return T, J, q, age

    def fields(self, domain, z):
        T, J, q, _ = self.state(domain, z)
        return T, J, q

    def residuals(self, domain, z):
        left, right = self.radii[domain]
        T, J, q, age = self.state(domain, z)
        dT, dJ = derivative(T, z), derivative(J, z)
        r = left + (right - left) * z[:, 0]
        k = 1.0 if domain == "paste" else 1.8
        b = self.paste_properties["capacity_j_m3k"] if domain == "paste" else 2.2e6
        constitutive = (J + k * dT[:, 0] / (right - left)) / self.flux_scale
        energy = b * dT[:, 1] / self.time_scale + dJ[:, 0] / (right - left) + J / r
        result = {"flux": constitutive}
        if q is not None:
            qt = derivative(q, z)[:, 1] / self.time_scale
            acceleration = torch.exp(
                40000 / R_GAS * (1 / self.material.reference_temperature_k - 1 / T))
            if age is None:
                rate = self.material.trial_rate(q) * acceleration
                result["reaction"] = (qt - rate) / (self.heat_scale / self.time_scale)
            else:
                result["reaction"] = derivative(age, z)[:, 1] / self.time_scale / acceleration - 1
            energy = energy - self.paste_properties["cement_kg_m3"] * qt
        result["energy"] = energy / (b * self.temperature_scale / self.time_scale)
        return result

    def losses(self, samples):
        losses = {}
        for domain in self.radii:
            for name, res in self.residuals(domain, samples[domain]).items():
                losses[domain + "_" + name] = res.square().mean()
        t = samples["times"]
        zero, one = torch.zeros_like(t), torch.ones_like(t)
        Ti, Ji, _ = self.fields("lining", torch.stack([zero, t], 1))
        Tl, Jl, _ = self.fields("lining", torch.stack([one, t], 1))
        Tm, Jm, _ = self.fields("paste", torch.stack([zero, t], 1))
        To, Jo, _ = self.fields("paste", torch.stack([one, t], 1))
        losses["air"] = ((-Ji - 8 * (Ti - 293.15)) / self.flux_scale).square().mean()
        losses["soil"] = ((Jo - 16 * (To - 293.15)) / self.flux_scale).square().mean()
        losses["interface_T"] = ((Tl - Tm) / self.temperature_scale).square().mean()
        losses["interface_J"] = ((Jl - Jm) / self.flux_scale).square().mean()
        return losses

    def query(self, domain, radius_m, time_h):
        radius, time = np.broadcast_arrays(radius_m, time_h)
        left, right = self.radii[domain]
        if (np.any(radius < left) or np.any(radius > right) or np.any(time < 0)
                or np.any(time > self.time_scale / 3600)
                or not np.isfinite(radius).all() or not np.isfinite(time).all()):
            raise ValueError("Query outside the trained radius/time domain")
        device = next(self.parameters()).device
        z = torch.tensor(np.stack([(radius.ravel() - left) / (right - left),
                                   time.ravel() * 3600 / self.time_scale], 1),
                         dtype=torch.float64, device=device)
        with torch.no_grad():
            T, J, q = self.fields(domain, z)
            if not torch.isfinite(T).all() or torch.any(T >= 313.15):
                raise ValueError("Temperature leaves the selected calculation scope")
            result = {"temperature_k": T, "flux_w_m2": J}
            if q is not None:
                result.update(heat_j_kg=q, modulus_pa=self.material.modulus(q),
                              reference_age_s=self.material.reference_age_s(q))
        return {key: value.cpu().numpy().reshape(radius.shape) for key, value in result.items()}


def load_thermal(path, data_dir, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = ThermalPINN(saved["config"], data_dir).double().to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()
