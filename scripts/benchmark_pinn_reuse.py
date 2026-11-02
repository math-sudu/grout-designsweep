"""Measure three neural-field reuse strategies on the prescribed design batch."""

import argparse
from contextlib import contextmanager
from copy import copy
import ctypes
import gc
import json
import os
from pathlib import Path
import platform
import sys
import threading
from time import perf_counter, process_time

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.pinn_event import SpatialEventOperator, train_event_family
from grouting_pino.pinn_history import PINNHistory
from grouting_pino.pinn_mechanics import FieldMatrixCache, SectionPINN, annulus_points
from grouting_pino.pinn_pressure import PressureEventPINN, PRESSURE_MODES
from grouting_pino.pinn_reuse import IndependentPressureEvents, InterpolatedPressureEvents
from grouting_pino.pinn_thermal import load_thermal
from run_pinn_plans import CachedPressureFields, run

OUT = ROOT / "results/pinn_support/design/workload"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def process_memory():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in
                ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
                 "nonpaged", "pagefile", "peak_pagefile", "private")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not query(process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.working
    # Unix reports the process high-water mark, as labelled in the output.
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


@contextmanager
def measure():
    result = {"start_rss_bytes": process_memory(), "peak_rss_bytes": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            result["peak_rss_bytes"] = max(result["peak_rss_bytes"], process_memory())
            stop.wait(.05)

    worker = threading.Thread(target=sample, daemon=True)
    started = perf_counter()
    cpu_started = process_time()
    worker.start()
    try:
        yield result
    finally:
        result["elapsed_s"] = perf_counter() - started
        result["cpu_s"] = process_time() - cpu_started
        result["peak_rss_bytes"] = max(result["peak_rss_bytes"], process_memory())
        stop.set()
        worker.join()


def frozen_initial():
    """Keep the supplied loaded displacement/stress identical in all strategies."""
    source = PINNHistory.load(ROOT / "results/pinn_support/W40/history_pressure/loaded_checkpoint.npz",
                              ROOT / "data/public_paste_benchmark")
    result = PINNHistory(source.thermal)
    result.time_h = source.time_h
    result.load_fraction = source.load_fraction
    result.load_coefficients = source.load_coefficients.copy()
    result.events = list(source.events)
    for time_h, coefficients in source.mode_events:
        parts = source.family.event(time_h, coefficients).parts
        for name, section in parts.items():
            amplitude = coefficients["uniform" if name == "uniform" else "sector"]
            if amplitude:
                result.events.append((amplitude, copy(section)))
    return result


def online(provider, mix, directory):
    cache = FieldMatrixCache()
    template = provider if isinstance(provider, PressureEventPINN) else provider.template
    for model in template.modes.values():
        cache.bind(model.section)
    if isinstance(provider, IndependentPressureEvents):
        for parts in provider.parts.values():
            for section in parts.values():
                cache.bind(section)
    initial = frozen_initial() if mix == "W40" else None
    if initial is not None:
        for _, section in initial.events:
            cache.bind(section)
    weights = {"cached": "cached_readouts.npz", "interpolated": "interpolation_readouts.npz"}
    source = {"strategy": directory.name,
              "spatial_template": f"results/pinn_support/{mix}/pressure/model.npz"}
    if directory.name in weights:
        source["readouts"] = (directory.parent / weights[directory.name]).relative_to(ROOT).as_posix()
    with measure() as measured:
        result = run(directory, lambda _mix, _thermal: CachedPressureFields(provider), (mix,), initial, source)
    measured["feature_cache_peak_bytes"] = cache.peak_bytes
    return result, measured


def restore_independent(template, path):
    events = IndependentPressureEvents(template, None)
    with np.load(path) as saved:
        for key in saved.files:
            time_string, name = key.split("__")
            time_h = float(time_string)
            section = copy(template.modes[name].section)
            section.config = section.config | {"time_h": time_h}
            section.weights = saved[key].copy()
            section.paste_modulus = lambda r, t=time_h: template.thermal.query(
                "paste", np.clip(r, 5.8, 6.02), t)["modulus_pa"]
            events.parts.setdefault(time_h, {})[name] = section
    return events


def replay(mix):
    directory = OUT / mix
    costs = json.loads((directory / "costs.json").read_text())
    thermal = load_thermal(ROOT / f"results/pinn_support/{mix}/thermal_age/model.pt",
                           ROOT / "data/public_paste_benchmark")
    template = PressureEventPINN.load(ROOT / f"results/pinn_support/{mix}/pressure/model.npz", thermal)
    providers = {"cached": restore_independent(template, directory / "cached_readouts.npz")}
    with np.load(directory / "interpolation_readouts.npz") as saved:
        library = IndependentPressureEvents(template, None)
        for i, time_h in enumerate(saved["times_h"]):
            library.parts[time_h] = {}
            for name in PRESSURE_MODES:
                section = copy(template.modes[name].section)
                section.weights = saved[name][i].copy()
                library.parts[time_h][name] = section
        providers["interpolated"] = InterpolatedPressureEvents(library, saved["times_h"].copy())
    providers["continuous"] = template
    for name, provider in providers.items():
        previous = json.loads((directory / name / "plans.json").read_text())
        result, timing = online(provider, mix, directory / name)
        old_plans = {row["id"]: row for row in previous["plans"]}
        change = max(abs(stage[key] - old[key]) for plan in result["plans"]
            for stage, old in zip(plan["stages"], old_plans[plan["id"]]["stages"], strict=True)
            for key in ("peak_um_kpa", "uplift_um_kpa"))
        if change > 1e-7:
            raise RuntimeError(f"Cached replay changed a stage field: {name}, {change}")
        costs["methods"][name]["task_execution"] = timing
        costs["methods"][name]["replay_maximum_stage_change_um_kpa"] = change
    costs["protocol"]["online_scope"] = (
        "One replay of the actual batch after preparing its unique cached events or the reusable library; "
        "includes planning, history replay, field exports and serialization. Total costs add cached event training once.")
    costs["protocol"]["event_cache"] = "The same pressure-event cache serves planning and history replay for all strategies"
    costs["protocol"]["continuous_replay"] = "Retained pressure family; stage fields checked against fresh joint training"
    write_json(directory / "costs.json", costs)


def field_comparison(reference, candidates, mix):
    """Compare actual off-grid loads and the full-handover controlling state."""
    result = {}
    probes = {"soil": np.r_[np.c_[np.linspace(-30., 30., 1201), np.zeros(1201)],
                                 annulus_points([6.05, 8., 15.], [.2, 1., 2.9])],
              "paste": annulus_points(5.91, np.linspace(-np.pi, np.pi, 37)),
              "lining": annulus_points(5.525, np.linspace(-np.pi, np.pi, 37))}
    for t in (.37, 1.73, 3.1):
        for ratio in (1., .5):
            coefficients = {"uniform": 1., "sector": ratio - 1.}
            expected = reference.event(t, coefficients)
            for name, candidate in candidates.items():
                actual = candidate.event(t, coefficients)
                row = {"time_h": t, "pressure_ratio": ratio, "domains": {}}
                for domain, xy in probes.items():
                    a, b = actual.query(domain, xy, True), expected.query(domain, xy, True)
                    differences = {}
                    for label, keys in (("displacement", ("ux", "uy")),
                                        ("stress", ("sxx", "syy", "sxy", "szz"))):
                        av, bv = [np.column_stack([f[key] for key in keys]) for f in (a, b)]
                        differences[label + "_relative_l2"] = float(np.linalg.norm(av - bv) / np.linalg.norm(bv))
                    if domain == "soil":
                        differences["surface_max_difference_um_kpa"] = float(np.max(np.abs(a["uy"][:1201] - b["uy"][:1201])) * 1e4)
                    row["domains"][domain] = differences
                result.setdefault(name, []).append(row)
    if mix == "W40":
        direct = {}
        for name in ("sector_even", "sector_odd"):
            path = ROOT / f"results/pinn_support/W40/pressure/thermal_age/check_order12_t0.37/{name}.npz"
            modulus = lambda r: reference.thermal.query("paste", r, .37)["modulus_pa"]
            expected = SectionPINN.load(path, modulus)
            actual = reference.parts[.37][name]
            if expected.config != actual.config:
                raise ValueError("Direct PINN check uses a different spatial configuration")
            direct[name] = {}
            for domain, xy in probes.items():
                a, b = [section.query(domain, xy, True) for section in (actual, expected)]
                for label, keys in (("displacement", ("ux", "uy")), ("stress", ("sxx", "syy", "sxy", "szz"))):
                    av, bv = [np.column_stack([f[key] for key in keys]) for f in (a, b)]
                    direct[name][domain + "_" + label + "_relative_l2"] = float(np.linalg.norm(av-bv) / np.linalg.norm(bv))
        result["shared_operator_vs_direct_event"] = direct
    return result


def benchmark(mix, bank_step):
    directory = OUT / mix
    directory.mkdir(parents=True, exist_ok=True)
    with measure() as loading:
        thermal = load_thermal(ROOT / f"results/pinn_support/{mix}/thermal_age/model.pt",
                               ROOT / "data/public_paste_benchmark")
        template = PressureEventPINN.load(ROOT / f"results/pinn_support/{mix}/pressure/model.npz", thermal)
    settings = json.loads((ROOT / "configs/pinn_support.json").read_text())["event"]
    with measure() as preparation:
        even = SpatialEventOperator([template.modes["uniform"], template.modes["sector_even"]])
        odd = SpatialEventOperator([template.modes["sector_odd"]])
    operators = [even, odd]
    common = sum(sum(operator.timing_s.values()) for operator in operators)
    measured = {"loading": loading, "spatial_preparation": preparation,
                "common_preparation_s": common,
                "spatial_operator_bytes": sum(value.nbytes for operator in operators
                    for value in vars(operator).values() if isinstance(value, np.ndarray)), "methods": {}}
    independent = IndependentPressureEvents(template, operators)
    _, timing = online(independent, mix, directory / "cached")
    training = list(independent.training)
    measured["methods"]["cached"] = {"task_execution": timing, "event_training": training,
        "event_training_s": sum(row["elapsed_s"] for row in training),
        "offline_s": common,
        "cached_readout_bytes": sum(a.nbytes for a in independent.arrays().values())}
    np.savez_compressed(directory / "cached_readouts.npz", **independent.arrays())
    write_json(directory / "costs.json", measured)

    times = np.unique(np.r_[np.arange(0., 4., bank_step), 4.])
    library = IndependentPressureEvents(template, operators)
    with measure() as bank_training:
        interpolated = InterpolatedPressureEvents(library, times)
    _, timing = online(interpolated, mix, directory / "interpolated")
    measured["methods"]["interpolated"] = {"task_execution": timing, "library_training": bank_training,
        "bank_step_h": bank_step, "event_training": library.training,
        "offline_s": common + bank_training["elapsed_s"],
        "readout_bytes": sum(a.nbytes for a in interpolated.readouts.values()),
        "interpolant_bytes": sum(spline.c.nbytes for spline in interpolated.interpolants.values())}
    np.savez_compressed(directory / "interpolation_readouts.npz", times_h=times, **interpolated.readouts)
    write_json(directory / "costs.json", measured)

    # Use the identical spatial setup, residual weights and stopping setting.
    breaks = np.unique(np.r_[np.arange(0., 4., settings["break_step_h"]), 4.])
    with measure() as joint_training:
        joint_records = [train_event_family(operator.models[0], breaks,
            quadrature_order=settings["quadrature_order"], linear_rtol=settings["linear_rtol"],
            additional_models=operator.models[1:], spatial_operator=operator) for operator in operators]
    _, timing = online(template, mix, directory / "continuous")
    measured["methods"]["continuous"] = {"task_execution": timing, "joint_training": joint_training,
        "training": joint_records, "offline_s": common + joint_training["elapsed_s"],
        "readout_bytes": sum(model.coefficients.nbytes for model in template.modes.values())}
    measured["field_comparison"] = field_comparison(independent,
        {"interpolated": interpolated, "continuous": template}, mix)
    measured["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "threadpools": [{key: value for key, value in pool.items() if key != "filepath"}
                        for pool in threadpool_info()], "memory_measurement":
        "Process working set sampled every 0.05 s" if os.name == "nt" else "Process high-water resident memory"}
    measured["protocol"] = {
        "tasks": "Unchanged complete_plans inputs; each method performs its own time-window searches and full history replays",
        "initial_state": "Identical frozen loaded W40 displacement, stress, clock and pressure in all methods",
        "shared_work": "One even and one odd spatial decomposition per material, shared pressure RHS, 128 MiB feature cache per task run",
        "common_cost": "The measured spatial preparation is charged once to every strategy",
        "thermal_cost": "The same retained thermal trajectory supplies all strategies; thermal training is a common upstream cost",
        "online_scope": "Includes planning, full history replay, field outputs and serialization; cached event training occurs inside task execution",
        "bank": "Not-a-knot cubic interpolation of five mixed neural output coefficient vectors; no extrapolation",
        "order": "Cached, interpolated, continuous; one controlled run, no repeated identical tasks used for a speedup"}
    write_json(directory / "costs.json", measured)
    print(json.dumps({"completed": mix, "costs": str(directory / "costs.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", choices=("W32", "W40", "W48"), required=True)
    parser.add_argument("--bank-step", type=float, default=.5)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=4):
        if args.replay:
            replay(args.mix)
        else:
            benchmark(args.mix, args.bank_step)
            gc.collect()
            replay(args.mix)
