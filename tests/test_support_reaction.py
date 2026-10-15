"""Heat requirements on plateaued curves and nonuniform radial states."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from grouting_pino.paste_reference import PasteReference
from run_support_reaction import reaction_at_event


def test_first_modulus_crossing_uses_the_beginning_of_a_plateau():
    reference = PasteReference("test", .4, 20., np.array([0., 4.]), np.array([1., 1.]),
                               np.arange(5.), np.array([1., 2., 2., 2., 4.]), 0., 4.)
    assert reference.first_heat_at_modulus(2.) == pytest.approx(3600.)
    assert reference.first_heat_at_modulus(3.) == pytest.approx(3.5*3600.)
    assert reference.first_heat_at_modulus(1.) == 0.
    assert reference.first_heat_at_modulus(5.) is None


def test_reaction_uses_annular_volume_weights_and_subtracts_initial_heat():
    history = SimpleNamespace(
        time_h=np.array([0., 1., 2.]),
        heat_j_kg_cement=np.array([[100., 500.], [200., 700.], [400., 1100.]]),
        mesh={"paste_volume": np.array([1., 3.])},
        volume_properties={"cement_kg_m3": 1200.})
    result = reaction_at_event(history, 1.5)
    assert result["mean_heat_increment_j_kg_cement"] == pytest.approx(350.)
    assert result["released_heat_mj_m3"] == pytest.approx(.42)
    restarted = SimpleNamespace(time_h=history.time_h[1:]-1.,
                                heat_j_kg_cement=history.heat_j_kg_cement[1:],
                                mesh=history.mesh, volume_properties=history.volume_properties)
    remaining = reaction_at_event(restarted, .5)
    assert remaining["mean_heat_increment_j_kg_cement"] == pytest.approx(175.)
    assert remaining["released_heat_mj_m3"] == pytest.approx(.21)
    with pytest.raises(ValueError, match="within"):
        reaction_at_event(history, 3.)
