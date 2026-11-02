"""The compressed joint solve must represent the full mixed physics system."""

import numpy as np
import pytest
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

from grouting_pino.pinn_event import (EventPINN, SpatialEventOperator, clamped_knots,
                                      time_quadrature, train_event_family)
from grouting_pino.pinn_mechanics import (SectionPINN, FieldMatrixCache,
    assemble_section_system, annulus_points)


class ThermalTrajectory:
    config = {"mix_id": "W40"}
    time_scale = 7200.

    def query(self, domain, radius, time_h):
        radius, time_h = np.broadcast_arrays(radius, time_h)
        return {"modulus_pa": 1e8 * (1 + .7 * time_h) * (1 + .1 * (radius - 5.8))}


@pytest.mark.parametrize("degree,continuity", [(1, 0), (3, 0), (3, 2)])
def test_joint_solve_matches_explicit_space_time_system_and_reloads(tmp_path, degree, continuity):
    thermal = ThermalTrajectory()
    config = {"mix_id": "W40", "time_h": 1., "seed": 23,
              "radial_neurons": {d: 2 for d in ("soil", "paste", "lining")},
              "angular_modes": 1, "domain_points": {"soil": 50, "paste": 30, "lining": 30},
              "boundary_points": 18, "domain_weight": {"soil": 20., "paste": 1., "lining": .2},
              "interface_weight": 30., "boundary_weight": 3., "svd_cutoff": 1e-11}
    section = SectionPINN(config, lambda r: thermal.query("paste", r, 1.)["modulus_pa"])
    model = EventPINN(section, thermal, clamped_knots([0., 1., 2.], degree, continuity), degree=degree)
    with threadpool_limits(limits=2):
        stats = train_event_family(model, [0., 1., 2.], quadrature_order=4, linear_rtol=1e-12)
        times, qweights = time_quadrature([0., 1., 2.], 4)
        rows, targets = [], []
        for t, weight, basis in zip(times, qweights, model.time_basis(times)):
            event = model.event(t)
            A, b, _ = assemble_section_system(event)
            rows.append(np.einsum("ij,k->ijk", A, basis).reshape(len(b), -1) * np.sqrt(weight))
            targets.append(b * np.sqrt(weight))
        matrix, target = np.concatenate(rows), np.concatenate(targets)
        expected = lstsq(matrix, target, cond=1e-11)[0]
        np.testing.assert_allclose(matrix @ model.coefficients.ravel(), matrix @ expected,
                                   atol=2e-9, rtol=2e-7)
        assert stats["relative_normal_residual"] < 2e-12
        # An untrained time remains a full field, and an earlier event is frozen.
        xy = annulus_points([5.85, 5.93, 5.99], [.2, 1., 2.9])
        early = model.event(.37)
        before = early.query("paste", xy, True)
        model.event(1.79)
        for key, value in before.items():
            np.testing.assert_array_equal(early.query("paste", xy, True)[key], value)
        path = tmp_path / "event.npz"
        model.save(path)
        restored = EventPINN.load(path, thermal)
        for key, value in before.items():
            np.testing.assert_array_equal(restored.query("paste", xy, .37, True)[key], value)
        with pytest.raises(ValueError):
            restored.event(2.01)


def test_cached_spatial_operator_matches_direct_independent_physics():
    thermal = ThermalTrajectory()
    config = {"mix_id": "W40", "time_h": 1., "seed": 23,
              "radial_neurons": {d: 2 for d in ("soil", "paste", "lining")},
              "angular_modes": 2, "domain_points": {"soil": 50, "paste": 30, "lining": 30},
              "boundary_points": 18, "domain_weight": {"soil": 20., "paste": 1., "lining": .2},
              "interface_weight": 30., "boundary_weight": 3., "svd_cutoff": 1e-11}
    section = SectionPINN(config, lambda r: thermal.query("paste", r, 1.)["modulus_pa"])
    model = EventPINN(section, thermal, clamped_knots([0., 1., 2.]))
    with threadpool_limits(limits=2):
        operator = SpatialEventOperator([model])
        for t in (0., .37, 1.73, 2.):
            parts, _ = operator.independent(t)
            part = parts["uniform"]
            A, b, _ = assemble_section_system(part)
            norm = np.linalg.norm(A, axis=0)
            expected = lstsq(A / norm, b, cond=1e-11)[0] / norm
            np.testing.assert_allclose(A @ part.weights, A @ expected, atol=2e-8, rtol=2e-7)
            xy = annulus_points([5.85, 5.93, 5.99], [.2, 1., 2.9])
            uncached = part.query("paste", xy, True)
            cache = FieldMatrixCache()
            cache.bind(part)
            for key, value in uncached.items():
                np.testing.assert_array_equal(part.query("paste", xy, True)[key], value)
            assert cache.bytes > 0


def test_continuous_design_queries_do_not_train(monkeypatch):
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    if not (root / "results/pinn_support/W40/event/model.npz").exists():
        pytest.skip("Retained continuous W40 field is required")
    sys.path.insert(0, str(root / "scripts"))
    import run_pinn_design

    def forbidden_training(*args, **kwargs):
        raise AssertionError("A new event time must use the retained readout")

    monkeypatch.setattr(run_pinn_design, "SpatialEventOperator", forbidden_training)
    with threadpool_limits(limits=2):
        series = run_pinn_design.ContinuousEventFields("W40")
        response = series.maximum([.37, 2.71], [.2, .8])
        assert np.isfinite(response["peak_um_kpa"])
        assert set(series.models) == {.37, 2.71}
        assert all(path == series.directory for path in series.paths.values())
