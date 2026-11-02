"""Check measured-state values, SI units and the derivatives used in PINNs."""

from pathlib import Path

import numpy as np
import pytest
import torch

from grouting_pino.paste_reference import PasteReference, load_paste_references
from grouting_pino.pinn_material import PasteStateMap


REFERENCES = load_paste_references(
    Path(__file__).resolve().parents[1] / "data/public_paste_benchmark")


@pytest.fixture(params=tuple(REFERENCES))
def material(request):
    reference = REFERENCES[request.param]
    return reference, PasteStateMap(reference)


def test_values_match_the_given_material_curves(material):
    reference, mapping = material
    age_h = np.unique(np.r_[np.linspace(reference.start_age_h, reference.end_age_h, 301),
                           mapping.heat_age_s.numpy() / 3600,
                           mapping.modulus_age_s.numpy() / 3600])
    # Keep the declared endpoints exactly, including floating-point h/s conversion.
    age_h = np.clip(age_h, reference.start_age_h, reference.end_age_h)
    heat = reference.heat_at_age(age_h)
    q = torch.tensor(heat, dtype=torch.float64)
    age_s = torch.tensor(age_h * 3600, dtype=torch.float64)
    torch.testing.assert_close(mapping.heat_at_reference_age_s(age_s), q,
                               rtol=2e-13, atol=1e-9)
    np.testing.assert_allclose(mapping.reference_age_s(q).numpy() / 3600, age_h,
                               rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(mapping.rate(q).numpy(), reference.rate_at_heat(heat),
                               rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(mapping.modulus(q).numpy(), reference.modulus_at_heat(heat),
                               rtol=2e-12, atol=1e-4)
    assert mapping.modulus(q[:1]).item() > 0
    assert not list(mapping.parameters())


def test_chain_rule_in_seconds_and_state_gradients(material):
    reference, mapping = material
    knots = np.unique(np.r_[mapping.heat_age_s.numpy(), mapping.modulus_age_s.numpy()])
    mid_age = (knots[:-1] + knots[1:]) / 2
    age = torch.tensor(mid_age, dtype=torch.float64, requires_grad=True)
    q = mapping.heat_at_reference_age_s(age)
    dq_da, = torch.autograd.grad(q.sum(), age, create_graph=True)
    torch.testing.assert_close(dq_da, mapping.rate(q), rtol=1e-11, atol=1e-11)

    heat = q.detach().requires_grad_()
    recovered = mapping.reference_age_s(heat)
    da_dq, = torch.autograd.grad(recovered.sum(), heat, create_graph=True)
    torch.testing.assert_close(da_dq * mapping.rate(heat), torch.ones_like(heat),
                               rtol=2e-12, atol=2e-12)

    modulus = mapping.modulus(heat)
    de_dq, = torch.autograd.grad(modulus.sum(), heat, create_graph=True)
    i = np.searchsorted(reference.modulus_age_h, mid_age / 3600, side="right") - 1
    slope = (np.diff(reference.modulus_pa) / np.diff(reference.modulus_age_h * 3600))[i]
    expected = torch.tensor(slope, dtype=torch.float64) / mapping.rate(heat)
    torch.testing.assert_close(de_dq, expected, rtol=2e-12, atol=1e-8)
    second, = torch.autograd.grad(de_dq.sum(), heat)
    assert torch.isfinite(second).all()


def test_origin_restart_and_invalid_predictions(material):
    reference, mapping = material
    q0 = torch.tensor(0., dtype=torch.float64)
    assert mapping.reference_age_s(q0).item() == pytest.approx(reference.start_age_h * 3600)
    later = torch.tensor(reference.heat_at_age(3.), dtype=torch.float64)
    assert mapping.reference_age_s(later).item() == pytest.approx(3 * 3600)
    assert later > q0
    for value in (-1., reference.max_heat_j_kg_cement + 1., float("nan")):
        heat = torch.tensor(value, dtype=torch.float64)
        assert not mapping.valid_heat(heat)
        for query in (mapping.reference_age_s, mapping.rate, mapping.modulus):
            with pytest.raises(ValueError, match="observation domain"):
                query(heat)
    trial = torch.tensor([-100., reference.max_heat_j_kg_cement + 100.], dtype=torch.float64)
    before = trial.clone()
    assert torch.isfinite(mapping.trial_rate(trial)).all()
    torch.testing.assert_close(trial, before)
    assert not mapping.valid_heat(trial).any()


def test_constant_rate_and_decreasing_modulus_are_preserved():
    reference = PasteReference("synthetic", .4, 20., np.array([1., 2., 3.]),
                               np.array([2., 2., 2.]), np.array([1., 2., 3.]),
                               np.array([1e8, 2e8, 1.5e8]), 1., 3.)
    mapping = PasteStateMap(reference)
    heat = torch.tensor([0., 3600., 7200., 10800., 14400.], dtype=torch.float64,
                         requires_grad=True)
    torch.testing.assert_close(mapping.reference_age_s(heat), 3600 + heat / 2)
    modulus = mapping.modulus(heat)
    torch.testing.assert_close(modulus,
                               torch.tensor([1e8, 1.5e8, 2e8, 1.75e8, 1.5e8], dtype=torch.float64))
    derivative, = torch.autograd.grad(modulus.sum(), heat)
    assert derivative[1] > 0 and derivative[3] < 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mapping_keeps_network_gradients(material):
    reference, mapping = material
    mapping = mapping.cuda()
    network_state = torch.nn.Parameter(torch.tensor([.01, .03], dtype=torch.float64, device="cuda"))
    heat = mapping.max_heat * network_state
    loss = (mapping.modulus(heat) / 1e9).square().mean() + mapping.rate(heat).square().mean()
    loss.backward()
    assert torch.isfinite(network_state.grad).all()
    assert torch.count_nonzero(network_state.grad) == 2
