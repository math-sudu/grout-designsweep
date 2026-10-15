"""Physical balances, material-clock continuation, and censored endpoints."""

from pathlib import Path

import numpy as np
import pytest

from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import PasteThermalProperties, simulate_paste
from grouting_pino.support_response import SectionResponse, first_attainment

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = load_paste_references(ROOT / "data/public_paste_benchmark")["W40"]
GEOMETRY = SectionResponse.load(ROOT / "results/support_design/reference_response.npz").geometry


def test_observed_age_is_carried_through_reference_temperature():
    time = np.linspace(0, 2, 41)
    run = simulate_paste(REFERENCE, GEOMETRY, time, 20, 20,
                         initial_reference_age_h=6, heat_release=False, rtol=1e-7)
    q = REFERENCE.heat_at_age(6+time)
    np.testing.assert_allclose(run.heat_j_kg_cement[:, 0], q, rtol=3e-5)
    np.testing.assert_allclose(run.effective_modulus_pa,
                               REFERENCE.modulus_at_heat(q), rtol=5e-5)
    assert run.effective_modulus_pa[0] > 0


def test_composite_energy_matches_heat_release_and_boundary_flux():
    run = simulate_paste(REFERENCE, GEOMETRY, np.linspace(0, 4, 81), 16, 18,
                         initial_paste_temperature_c=22, initial_ring_temperature_c=18)
    stored = (run.temperature_c-run.temperature_c[0]) @ run.mesh["capacity"]
    released = ((run.heat_j_kg_cement-run.heat_j_kg_cement[0])
                @ run.mesh["paste_volume"] * run.volume_properties["cement_kg_m3"])
    np.testing.assert_allclose(stored, released+run.boundary_energy_j, atol=0.1, rtol=1e-7)
    volume = run.mesh["paste_volume"].sum()
    assert volume == pytest.approx((6.02**2-5.8**2)/2)


def test_nonuniform_state_restart_preserves_the_same_physical_history():
    whole = simulate_paste(REFERENCE, GEOMETRY, np.linspace(0, 4, 81), 18, 20, rtol=1e-7)
    first = simulate_paste(REFERENCE, GEOMETRY, np.linspace(0, 2, 41), 18, 20, rtol=1e-7)
    second = simulate_paste(REFERENCE, GEOMETRY, np.linspace(0, 2, 41), 18, 20,
                            initial_heat_j_kg_cement=first.heat_j_kg_cement[-1],
                            initial_temperature_c=first.temperature_c[-1],
                            state_time_h=first.construction_time_h[-1], rtol=1e-7)
    np.testing.assert_allclose(second.temperature_c[-1], whole.temperature_c[-1], atol=0.002)
    np.testing.assert_allclose(second.modulus_pa[-1], whole.modulus_pa[-1], rtol=5e-4)
    section = SectionResponse.load(ROOT / "results/support_design/reference_response.npz")
    original = first_attainment(whole.time_h, section.compliance(whole.effective_modulus_pa), 2.2e-9)
    updated = first_attainment(second.time_h, section.compliance(second.effective_modulus_pa), 2.2e-9)
    assert second.construction_time_h[0] == 2
    assert second.construction_time_h[-1] == whole.construction_time_h[-1]
    assert updated.time_h + second.state_time_h == pytest.approx(original.time_h, abs=0.001)


def test_material_and_temperature_endpoints_terminate_without_flat_tails():
    run = simulate_paste(REFERENCE, GEOMETRY, np.linspace(0, 12, 121), 20, 20)
    assert run.status == "temperature_scope_end"
    assert run.time_h[-1] < 12
    assert run.temperature_c[-1].max() == pytest.approx(40, abs=1e-6)
    material = simulate_paste(REFERENCE, GEOMETRY, [0, 1, 2], 20, 20,
                              initial_reference_age_h=REFERENCE.end_age_h-0.5,
                              heat_release=False, rtol=1e-7)
    assert material.status == "material_domain_end"
    assert material.time_h[-1] == pytest.approx(0.5, abs=2e-4)
    assert material.heat_j_kg_cement.max() <= REFERENCE.max_heat_j_kg_cement


def test_volume_dosage_uses_measured_ratio_and_a_common_fill_volume():
    props = PasteThermalProperties()
    for wc in (0.32, 0.40, 0.48):
        row = props.volume_properties(wc)
        mc = row["cement_kg_m3"]
        assert mc/3150 + wc*mc/998.2 == pytest.approx(1)
        assert mc/row["density_kg_m3"] == pytest.approx(1/(1+wc))
