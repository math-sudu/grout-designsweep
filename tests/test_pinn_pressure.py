"""Full-section parity, discontinuous forcing and shared pressure readouts."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from grouting_pino.pinn_event import EventPINN, clamped_knots, train_event_family
from grouting_pino.pinn_mechanics import (COMPONENTS, NeuralBasis, SectionPINN,
    annulus_points, assemble_section_system, boundary_angles, pressure_profile, section_target)
from grouting_pino.pinn_pressure import PressureEventPINN
from grouting_pino.pinn_design import surface_extrema
from test_pinn_event import ThermalTrajectory


@pytest.mark.parametrize("domain", ["soil", "paste", "lining"])
def test_odd_pressure_branch_reflection_and_derivatives(domain):
    basis = NeuralBasis.create(domain, 6, 4, np.random.default_rng(17), "odd")
    radius = {"soil": 12., "paste": 5.91, "lining": 5.52}[domain]
    xy = annulus_points(radius, [-2.7, -.4, .4, 2.7])
    mirrored = xy * [-1., 1.]
    for comp in COMPONENTS:
        value, dx, dy = basis.evaluate(xy, comp)
        sign = 1 if comp in ("ux", "sxy") else -1
        np.testing.assert_allclose(basis.evaluate(mirrored, comp)[0], sign * value, atol=1e-12)
        for axis, derivative in [(0, dx), (1, dy)]:
            shift = np.zeros_like(xy)
            shift[:, axis] = 1e-5
            fd = (basis.evaluate(xy + shift, comp)[0] - basis.evaluate(xy - shift, comp)[0]) / 2e-5
            np.testing.assert_allclose(fd, derivative, rtol=2e-4, atol=3e-5)


def test_sector_pressure_and_segment_quadrature():
    config = {"boundary_points": 31, "sector_angle_deg": 62.5}
    theta, weights = boundary_angles(config)
    sector = pressure_profile(config | {"pressure_mode": "sector_even"}, theta)
    assert weights @ sector == pytest.approx(62.5 / 360.)
    theta = np.deg2rad([-100., -20., 20., 100.])
    right = sum(pressure_profile(config | {"pressure_mode": name}, theta)
                for name in ("sector_even", "sector_odd"))
    np.testing.assert_array_equal(right, [0., 0., 1., 0.])


def test_signed_surface_search_preserves_uplift():
    result = surface_extrema(lambda x: 1. - 2. * np.cos(x), np.linspace(-np.pi, np.pi, 101))
    assert result["peak_um_kpa"] == pytest.approx(3.)
    assert result["uplift_um_kpa"] == pytest.approx(1.)
    assert result["uplift_x_m"] == pytest.approx(0., abs=1e-6)


def test_shared_rhs_matches_separate_physics_solves_and_restores(tmp_path):
    thermal = ThermalTrajectory()
    config = {"mix_id": "W40", "time_h": 1., "seed": 23,
              "radial_neurons": {d: 2 for d in ("soil", "paste", "lining")},
              "angular_modes": 2, "domain_points": {"soil": 50, "paste": 30, "lining": 30},
              "boundary_points": 24, "domain_weight": {"soil": 20., "paste": 1., "lining": .2},
              "interface_weight": 30., "boundary_weight": 3., "svd_cutoff": 1e-11,
              "sector_angle_deg": 62.5}

    def make(mode):
        section = SectionPINN(config | {"pressure_mode": mode,
            "reflection": "odd" if mode == "sector_odd" else "even"},
            lambda r: thermal.query("paste", r, 1.)["modulus_pa"])
        return EventPINN(section, thermal, clamped_knots([0., 1., 2.], 1), 1)

    modes = {name: make(name) for name in ("uniform", "sector_even", "sector_odd")}
    with threadpool_limits(limits=2):
        a, b, labels = assemble_section_system(modes["sector_even"].section)
        np.testing.assert_allclose(b, section_target(modes["sector_even"].section, labels), atol=1e-15)
        train_event_family(modes["uniform"], [0., 1., 2.], linear_rtol=1e-12,
                           additional_models=[modes["sector_even"]])
        separate = make("sector_even")
        train_event_family(separate, [0., 1., 2.], linear_rtol=1e-12)
        np.testing.assert_allclose(a @ separate.coefficients, a @ modes["sector_even"].coefficients,
                                   atol=2e-10, rtol=2e-8)
        train_event_family(modes["sector_odd"], [0., 1., 2.], linear_rtol=1e-12)
        family = PressureEventPINN(modes)
        section = family.event(.37, {"uniform": 1., "sector": -.8})
        xy = annulus_points(5.91, [-2.7, -.4, .4, 2.7])
        field = section.query("paste", xy, True)
        path = tmp_path / "pressure.npz"
        family.save(path)
        restored = PressureEventPINN.load(path, thermal)
        for key, value in field.items():
            np.testing.assert_array_equal(restored.event(.37, section.coefficients).query("paste", xy, True)[key], value)
        assert family.mean_pressure_fraction(section.coefficients) == pytest.approx(1 - .8 * 62.5 / 360)
