"""Pressure reuse must preserve mixed fields and frozen event semantics."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from grouting_pino.pinn_event import EventPINN, SpatialEventOperator, clamped_knots
from grouting_pino.pinn_mechanics import SectionPINN, annulus_points
from grouting_pino.pinn_pressure import PressureEventPINN, PRESSURE_MODES
from grouting_pino.pinn_reuse import IndependentPressureEvents, InterpolatedPressureEvents
from test_pinn_event import ThermalTrajectory


def test_independent_cache_and_interpolated_pressure_fields():
    thermal = ThermalTrajectory()
    config = {"mix_id": "W40", "time_h": 1., "seed": 23,
              "radial_neurons": {d: 2 for d in ("soil", "paste", "lining")},
              "angular_modes": 2, "domain_points": {"soil": 50, "paste": 30, "lining": 30},
              "boundary_points": 24, "domain_weight": {"soil": 20., "paste": 1., "lining": .2},
              "interface_weight": 30., "boundary_weight": 3., "svd_cutoff": 1e-11,
              "sector_angle_deg": 62.5}
    modes = {}
    for name in PRESSURE_MODES:
        section = SectionPINN(config | {"pressure_mode": name,
            "reflection": "odd" if name == "sector_odd" else "even"},
            lambda r: thermal.query("paste", r, 1.)["modulus_pa"])
        modes[name] = EventPINN(section, thermal, clamped_knots([0., 1., 2.]))
    template = PressureEventPINN(modes)
    with threadpool_limits(limits=2):
        operators = [SpatialEventOperator([modes["uniform"], modes["sector_even"]]),
                     SpatialEventOperator([modes["sector_odd"]])]
        reference = IndependentPressureEvents(template, operators)
        reference.event(.37, {"uniform": 1.})
        assert set(reference.parts[.37]) == {"uniform"}
        frozen = reference.event(.37, {"uniform": .8, "sector": -.5})
        xy = annulus_points(5.91, [-2.7, -.4, .4, 2.7])
        expected = frozen.query("paste", xy, True)
        records = len(reference.training)
        reference.event(.37, {"uniform": 1., "sector": -.8})
        assert len(reference.training) == records
        bank = InterpolatedPressureEvents(reference, [0., .5, 1., 1.5, 2.])
        coefficients = {"uniform": .8, "sector": -.5}
        for t in (0., .5, 1.5, 2.):
            actual = bank.event(t, coefficients).query("paste", xy, True)
            direct = reference.event(t, coefficients).query("paste", xy, True)
            for key in actual:
                np.testing.assert_allclose(actual[key], direct[key], atol=2e-9, rtol=1e-12)
        for key, value in expected.items():
            np.testing.assert_array_equal(frozen.query("paste", xy, True)[key], value)
        with pytest.raises(ValueError, match="interpolation library"):
            bank.event(2.01)
