"""Exercise event semantics using the actual trained W40 fields."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from check_pinn_mechanics import load_section
from grouting_pino.pinn_history import PINNHistory
from grouting_pino.pinn_thermal import load_thermal
from grouting_pino.pinn_pressure import PressureEventPINN


@pytest.fixture(scope="module")
def fields():
    torch.set_num_threads(2)
    folder=ROOT/"results/pinn_support/W40"
    required=[folder/"thermal_age/model.pt",folder/"mechanics_age/t0_refined/model.npz",folder/"mechanics_age/t4_refined/model.npz"]
    if not all(path.exists() for path in required):
        pytest.skip("W40 trained field integration test requires the retained model files")
    return (load_thermal(required[0],ROOT/"data/public_paste_benchmark"),
            load_section(folder/"mechanics_age/t0_refined"),load_section(folder/"mechanics_age/t4_refined"))


def test_loaded_restart_retains_fields_load_and_material_clock(fields,tmp_path):
    thermal,early,late=fields
    history=PINNHistory(thermal)
    xy=np.array([[0.,0.],[10.,0.],[5.91,-25.52]])
    history.apply(early,.5)
    held=history.query("soil",xy[:2])
    history.advance(2.)
    checkpoint=tmp_path/"loaded.npz"
    history.save(checkpoint)
    restart=PINNHistory.load(checkpoint,ROOT/"data/public_paste_benchmark")
    assert restart.time_h==2 and restart.load_fraction==.5
    assert restart.material_state()["paste_heat_j_kg"].min()>0
    for key,value in held.items():
        np.testing.assert_array_equal(restart.query("soil",xy[:2])[key],value)
    for path in (history,restart):
        path.advance(4.)
        path.apply(late,1.)
    for domain,points in [("soil",xy[:2]),("paste",xy[2:])]:
        for key,value in history.query(domain,points).items():
            np.testing.assert_array_equal(restart.query(domain,points)[key],value)
    # A new load is measured from the applied starting load of 0.5.
    assert restart.events[-1][0]==.5
    restart.apply(late,.75)
    assert restart.events[-1][0]==-.25
    restart.apply(late,1.)
    for key,value in history.query("soil",xy[:2]).items():
        np.testing.assert_allclose(restart.query("soil",xy[:2])[key],value,rtol=1e-13,atol=1e-10)


def test_event_fields_are_frozen_and_invalid_time_does_not_advance(fields):
    thermal,early,_=fields
    history=PINNHistory(thermal)
    history.apply(early,.5)
    assert not np.shares_memory(history.events[0][1].weights,early.weights)
    with pytest.raises(ValueError,match="trained"):
        history.advance(4.5)
    assert history.time_h==0
    history.advance(2.)
    with pytest.raises(ValueError,match="current material time"):
        history.apply(early,1.)
    assert history.load_fraction==.5


def test_retained_legacy_checkpoint_loads(fields):
    path = ROOT / "results/pinn_support/W40/history/loaded_checkpoint.npz"
    if not path.exists():
        pytest.skip("Retained W40 restart is required")
    thermal, early, _ = fields
    old = PINNHistory.load(path, ROOT / "data/public_paste_benchmark")
    assert old.time_h == 2. and old.load_fraction == .5
    xy = np.array([[-10., 0.], [0., 0.], [10., 0.]])
    expected = PINNHistory(thermal)
    expected.apply(early, .5)
    for key, value in expected.query("soil", xy).items():
        np.testing.assert_allclose(old.query("soil", xy)[key], value, atol=1e-12, rtol=1e-12)


def test_conditional_history_restarts_splits_and_preserves_both_fields(fields, tmp_path):
    path = ROOT / "results/pinn_support/W40/pressure/model.npz"
    if not path.exists():
        pytest.skip("Trained pressure family is required")
    thermal, _, _ = fields
    family = PressureEventPINN.load(path, thermal)
    history, split = PINNHistory(thermal), PINNHistory(thermal)
    for current in (history, split):
        current.bind_family(family)
        current.advance(.37)
    assert history.family is split.family is family
    split.apply_modes({"uniform": .1, "sector": -.05})
    for current in (history, split):
        current.apply_modes({"uniform": .2, "sector": -.1})
    probes = {"soil": np.array([[-10., 0.], [0., 0.], [10., 0.]]),
              "paste": np.array([[-5.91, -25.52], [5.91, -25.52]])}
    before = {domain: history.query(domain, xy) for domain, xy in probes.items()}
    for current in (history, split):
        current.advance(1.73)
    for domain, xy in probes.items():
        for key, value in before[domain].items():
            np.testing.assert_array_equal(history.query(domain, xy)[key], value)
    checkpoint = tmp_path / "modes.npz"
    history.save(checkpoint)
    restored = PINNHistory.load(checkpoint, ROOT / "data/public_paste_benchmark")
    assert restored.load_coefficients == {"uniform": .2, "sector": -.1}
    assert restored.load_fraction == pytest.approx(.2 - .1 * 62.5 / 360.)
    assert len(restored.mode_events) == 1
    # Histories share frozen readouts; an attempted update fails explicitly.
    with pytest.raises(ValueError):
        family.modes["uniform"].coefficients[:] = 0.
    with pytest.raises(ValueError, match="frozen"):
        family.modes["uniform"].coefficients = np.zeros_like(family.modes["uniform"].coefficients)
    for current in (history, split, restored):
        current.apply_modes({"uniform": .45, "sector": -.225})
        current.advance(4.)
        current.apply_modes({"uniform": 1., "sector": -.5})
    for domain, xy in probes.items():
        for key, value in history.query(domain, xy).items():
            np.testing.assert_allclose(split.query(domain, xy)[key], value, rtol=1e-12, atol=1e-9)
            np.testing.assert_array_equal(restored.query(domain, xy)[key], value)
        old = PINNHistory.load(checkpoint, ROOT / "data/public_paste_benchmark")
        for key, value in before[domain].items():
            np.testing.assert_array_equal(old.query(domain, xy)[key], value)


def test_legacy_loaded_state_can_continue_with_pressure_modes(fields):
    model_path = ROOT / "results/pinn_support/W40/pressure/model.npz"
    checkpoint = ROOT / "results/pinn_support/W40/history/loaded_checkpoint.npz"
    if not model_path.exists() or not checkpoint.exists():
        pytest.skip("Retained W40 family and loaded state are required")
    thermal, _, _ = fields
    history = PINNHistory.load(checkpoint, ROOT / "data/public_paste_benchmark")
    family = PressureEventPINN.load(model_path, thermal)
    xy = np.array([[-10., 0.], [0., 0.], [10., 0.]])
    start = history.query("soil", xy)
    history.bind_family(family)
    history.advance(4.)
    history.apply_modes({"uniform": 1., "sector": -.5})
    increment = family.event(4., {"uniform": .5, "sector": -.5}).query("soil", xy, True)
    for key, value in start.items():
        np.testing.assert_allclose(history.query("soil", xy)[key], value + increment[key], rtol=1e-12, atol=1e-9)
    assert history.load_fraction == pytest.approx(1. - .5 * 62.5 / 360.)


def test_planning_and_replay_accept_restored_family_and_reject_mismatches(fields, tmp_path):
    from copy import deepcopy
    from threadpoolctl import threadpool_limits
    from grouting_pino.pinn_design import solve_prescribed_plan, replay_prescribed_plan

    thermal, _, _ = fields
    path = ROOT / "results/pinn_support/W40/pressure/model.npz"
    family = PressureEventPINN.load(path, thermal)
    history = PINNHistory(thermal)
    history.bind_family(family)
    history.advance(.37)
    history.apply_modes({"uniform": .2})
    history.advance(1.)
    checkpoint = tmp_path / "history.npz"
    history.save(checkpoint)
    restored = PINNHistory.load(checkpoint, ROOT / "data/public_paste_benchmark")
    assert restored.family is not family
    with threadpool_limits(limits=2):
        expected = solve_prescribed_plan(restored.family, [1.73, 3.1, 4.], [.3, .5, 1.],
                                         {"uniform": 1.}, initial=restored)
        actual = solve_prescribed_plan(family, [1.73, 3.1, 4.], [.3, .5, 1.],
                                       {"uniform": 1.}, initial=restored)
        assert actual["events"] == expected["events"]
        assert actual["minimum_margin_um_kpa"] == expected["minimum_margin_um_kpa"]
        from run_pinn_plans import CachedPressureFields
        cached = CachedPressureFields(family)
        cached_plan = solve_prescribed_plan(cached, [1.73, 3.1, 4.], [.3, .5, 1.],
                                            {"uniform": 1.}, initial=restored)
        assert cached_plan["events"] == expected["events"]
        assert cached_plan["stages"] == expected["stages"]
        assert cached.workload()["event_requests"] == 3
        assert cached.workload()["unique_pressure_events"] == 3
        assert cached.workload()["unique_time_mode_pairs"] == 3
        solve_prescribed_plan(cached, [1.73, 3.1, 4.], [.3, .5, 1.],
                               {"uniform": 1.}, initial=restored)
        assert cached.workload()["event_requests"] == 6
        assert cached.workload()["unique_pressure_events"] == 3
        final = list(replay_prescribed_plan(family, actual, restored))[-1]
        assert final.family is restored.family
        assert final.load_fraction == pytest.approx(1.)
        assert restored.time_h == 1. and restored.load_fraction == .2

        changed_thermal = deepcopy(thermal)
        with torch.no_grad():
            next(changed_thermal.parameters()).add_(1e-4)
        thermal_mismatch = PressureEventPINN.load(path, changed_thermal)
        basis_mismatch = PressureEventPINN.load(path, thermal)
        for mode in ("sector_even", "sector_odd"):
            basis_mismatch.modes[mode].section.config["sector_angle_deg"] = 60.
        basis_mismatch.angle_deg = 60.
        readout_mismatch = PressureEventPINN.load(path, thermal)
        readout_mismatch.modes["uniform"].coefficients[0, 0] += 1e-4
        for candidate, message in [(thermal_mismatch, "thermal trajectory"),
                                    (basis_mismatch, "pressure family retained"),
                                    (readout_mismatch, "pressure family retained")]:
            with pytest.raises(ValueError, match=message):
                solve_prescribed_plan(candidate, [4.], [1.], {"uniform": 1.}, initial=restored)
            with pytest.raises(ValueError, match=message):
                list(replay_prescribed_plan(candidate, actual, restored))
            with pytest.raises(ValueError, match=message):
                solve_prescribed_plan(CachedPressureFields(candidate), [4.], [1.],
                                       {"uniform": 1.}, initial=restored)
