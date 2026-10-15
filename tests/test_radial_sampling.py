"""Event interpolation must preserve the decisions the design map consumes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_radial_support import sample_response
from grouting_pino.support_response import first_attainment


def test_curved_response_crossings_and_censored_demand():
    targets = np.linspace(1.5, 3.5, 41)
    points = sample_response(lambda t: 1 + 4*np.exp(-t), 2., targets)
    times = np.array(sorted(points))
    response = np.array([points[t] for t in times])
    for target in targets:
        hit = first_attainment(times, response, target)
        exact = -np.log((target-1)/4)
        if exact > 2:
            assert hit.time_h is None
        else:
            assert hit.time_h == pytest.approx(exact, abs=0.002)


def test_near_boundary_choice_uses_the_direct_response():
    crossing = 4.001
    curve = lambda t: 1.2 + np.exp((crossing-t)/2)
    points = sample_response(curve, 6., np.array([2.2]))
    assert points[4.] > 2.2
    times = np.array(sorted(points))
    assert first_attainment(times, [points[t] for t in times], 2.2).time_h > 4.


def test_initially_sufficient_material_and_short_history():
    points = sample_response(lambda t: 2-t, 0.13, np.array([2.2, 1.5]))
    times = np.array(sorted(points))
    response = np.array([points[t] for t in times])
    assert first_attainment(times, response, 2.2).time_h == 0
    assert first_attainment(times, response, 1.5).time_h is None
    assert times[-1] == 0.13
