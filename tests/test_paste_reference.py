from pathlib import Path

import numpy as np
import pytest

from grouting_pino.paste_reference import PasteReference, load_paste_references


def test_heat_increment_uses_seconds_and_retains_initial_age():
    material = PasteReference(
        "linear", 0.4, 20, np.array([0., 2.]), np.array([4., 2.]),
        np.array([0., 1., 2.]), np.array([0., 100e6, 200e6]), 1., 2.)
    # Between 1 and 2 h the heat flow falls from 3 to 2 W/kg.
    assert material.heat_at_age(2.) == pytest.approx(2.5 * 3600)
    assert material.age_at_heat(9000.) == pytest.approx(2.)
    assert material.rate_at_heat(0.) == pytest.approx(3.)
    assert material.modulus_at_heat(0.) == pytest.approx(100e6)
    assert material.modulus_at_heat(9000.) == pytest.approx(200e6)


def test_constant_heat_flow_has_a_finite_inverse():
    material = PasteReference(
        "constant", 0.5, 20, np.array([0., 8.]), np.array([2., 2.]),
        np.array([0., 8.]), np.array([0., 8e9]), 1., 7.)
    assert material.heat_at_age(7.) == pytest.approx(43200.)
    assert material.age_at_heat(21600.) == pytest.approx(4.)
    assert material.cement_mass_fraction == pytest.approx(2/3)


def test_public_curves_preserve_recipe_order_and_measured_domain():
    root = Path(__file__).resolve().parents[1]
    materials = load_paste_references(root / "data/public_paste_benchmark")
    assert list(materials) == ["W32", "W40", "W48"]
    moduli = [float(m.modulus_at_heat(m.heat_at_age(24))) for m in materials.values()]
    # Figure 10 at one day: approximately 12, 9 and 7 GPa.
    assert 11e9 < moduli[0] < 13e9
    assert 8e9 < moduli[1] < 10e9
    assert 6e9 < moduli[2] < 8e9
    for m in materials.values():
        assert m.start_age_h > 0
        assert m.end_age_h < 60
        assert m.modulus_at_heat(0.) > 0
        with pytest.raises(ValueError, match="domain"):
            m.heat_at_age(0.)
        with pytest.raises(ValueError, match="domain"):
            m.heat_at_age(72.)
        with pytest.raises(ValueError, match="domain"):
            m.modulus_at_heat(m.max_heat_j_kg_cement + 1)
