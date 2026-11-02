"""Initial data, support interval and gradient flow of the mixed thermal PINN."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

from grouting_pino.pinn_thermal import ThermalPINN

ROOT=Path(__file__).resolve().parents[1]


def test_initial_fields_and_supported_query_interval():
    config=json.loads((ROOT/"configs/pinn_support.json").read_text())
    model=ThermalPINN(config,ROOT/"data/public_paste_benchmark").double()
    for domain,(lo,hi) in model.radii.items():
        fields=model.query(domain,np.linspace(lo,hi,13),0)
        np.testing.assert_array_equal(fields["temperature_k"],293.15)
        np.testing.assert_array_equal(fields["flux_w_m2"],0)
        if domain=="paste":
            np.testing.assert_array_equal(fields["heat_j_kg"],0)
        with pytest.raises(ValueError,match="trained"):
            model.query(domain,(lo+hi)/2,4.01)
    z=torch.tensor([[.1,.3],[.9,.8]],dtype=torch.float64,requires_grad=True)
    sum(value.square().sum() for value in model.residuals("paste",z).values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_equivalent_age_bounds_and_reaction_residual():
    config = json.loads((ROOT / "configs/pinn_support.json").read_text())
    model = ThermalPINN(config, ROOT / "data/public_paste_benchmark").double()
    with torch.no_grad():
        model.networks["paste"][-1].bias[:] = torch.tensor([-.7, .2, -2.])
    z = torch.tensor([[.1, .001], [.5, .1], [.9, 1.]],
                     dtype=torch.float64, requires_grad=True)
    temperature, _, heat, age = model.state("paste", z)
    elapsed = z[:, 1] * model.time_scale
    assert torch.all(temperature >= 293.15)
    assert torch.all(age >= model.material.heat_age_s[0] + elapsed)
    torch.testing.assert_close(model.material.reference_age_s(heat), age)
    from grouting_pino.paste_thermal import R_GAS
    acceleration = torch.exp(40000 / R_GAS * (1 / 293.15 - 1 / temperature))
    age_rate = 1 + 2 * z[:, 1] * torch.nn.functional.softplus(
        model.networks["paste"][-1].bias[2])
    torch.testing.assert_close(model.residuals("paste", z)["reaction"],
                               age_rate / acceleration - 1)


def test_saved_heat_representation_keeps_original_outputs():
    config = json.loads((ROOT / "configs/pinn_support.json").read_text())
    config["thermal"].pop("state_representation")
    model = ThermalPINN(config, ROOT / "data/public_paste_benchmark").double()
    z = torch.tensor([[.2, 0.], [.8, .1]], dtype=torch.float64)
    temperature, flux, heat = model.fields("paste", z)
    torch.testing.assert_close(temperature, torch.full_like(temperature, 293.15))
    torch.testing.assert_close(flux, torch.zeros_like(flux))
    torch.testing.assert_close(heat, model.heat_scale * z[:, 1] * np.log(2))


@pytest.mark.parametrize("age_rate", [1., .9])
def test_event_diagnostic_resolves_an_early_age_bias(age_rate):
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_pinn_thermal_events import assess_events
    config = json.loads((ROOT / "configs/pinn_support.json").read_text())
    model = ThermalPINN(config, ROOT / "data/public_paste_benchmark").double()
    origin = float(model.material.heat_age_s[0])

    def prescribed_state(domain, radius_m, time_h):
        radius, time = np.broadcast_arrays(radius_m, time_h)
        age = torch.tensor(origin + age_rate * time * 3600, dtype=torch.float64)
        heat = model.material.heat_at_reference_age_s(age)
        return {"temperature_k": np.full(radius.shape, 293.15),
                "reference_age_s": age.numpy(), "heat_j_kg": heat.numpy(),
                "modulus_pa": model.material.modulus(heat).numpy()}

    model.query = prescribed_state
    event = assess_events(model, [.37], order=12, radial_points=7)["events"][0]
    expected_error = (age_rate - 1) * .37 * 3600
    assert event["minimum_age_excess_s"] == pytest.approx(expected_error, abs=1e-10)
    assert event["signed_age_integral_error_at_maximum_s"] == pytest.approx(expected_error, abs=1e-10)
    assert event["maximum_absolute_age_integral_error_s"] == pytest.approx(abs(expected_error), abs=1e-10)
    actual, exact = model.material.heat_at_reference_age_s(
        torch.tensor([origin + age_rate * .37 * 3600, origin + .37 * 3600],
                     dtype=torch.float64)).numpy()
    assert event["maximum_relative_heat_integral_error"] == pytest.approx(abs(actual / exact - 1), abs=1e-12)
