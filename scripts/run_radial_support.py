"""Compute radial-FE loading events for every common-origin design scenario.

Run in the dolfinx container. Each worker reuses one section mesh and elastic
problem. Adaptive FE samples resolve the demand crossings; existing thermal
scenarios and loading-event semantics are shared with the uniform queries.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import simulate_paste
from grouting_pino.support_response import SectionResponse, first_attainment
from run_paste_support import thermal_scenarios
from support_section import RadialSection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/support_design"
KEYS = ("scenario", "case", "mix_id")
DEMANDS = np.linspace(1.5, 3.5, 41)


def sample_response(evaluate, end_h, targets, time_tolerance_h=0.002):
    """Sample the response where interpolation affects a requested crossing.

    Midpoint evaluation compares the piecewise-linear inverse with the endpoint
    inverse at the actual demand levels. A 0.5 h seed grid also samples intervals
    outside the demand range. Near a 0.1 h decision boundary, evaluate that time
    directly so the selected material does not depend on which side a small
    interpolation error falls. All evaluated points remain in the history.
    """
    cache = {}

    def value(t):
        t = float(t)
        if t not in cache:
            cache[t] = float(evaluate(t))
        return cache[t]

    def refine(a, b):
        ca, cb = value(a), value(b)
        middle = (a + b) / 2
        cm = value(middle)
        if b - a <= 2 * time_tolerance_h:
            return
        nonmonotone = not min(ca, cb) <= cm <= max(ca, cb)
        relevant = targets[(targets >= min(ca, cb, cm)) & (targets <= max(ca, cb, cm))]
        if nonmonotone and relevant.size:
            refine(a, middle)
            refine(middle, b)
        elif relevant.size and ca > cb:
            coarse = np.interp(relevant, [cb, ca], [b, a])
            split = np.interp(relevant, [cb, cm, ca], [b, middle, a])
            if np.max(np.abs(split - coarse)) > time_tolerance_h:
                refine(a, middle)
                refine(middle, b)

    seeds = np.unique(np.r_[np.arange(0, end_h, 0.5), end_h])
    for a, b in zip(seeds[:-1], seeds[1:]):
        refine(a, b)
    times = np.array(sorted(cache))
    values = np.array([cache[t] for t in times])
    for target in targets:
        hit = first_attainment(times, values, target)
        if hit.time_h is not None:
            boundary = round(hit.time_h * 10) / 10
            if 0 <= boundary <= end_h and abs(boundary - hit.time_h) <= 2*time_tolerance_h:
                value(boundary)
    return cache


def initialize_worker():
    global SECTION, REFERENCES, SCENARIOS, RADIAL
    SECTION = SectionResponse.load(OUT / "reference_response.npz")
    REFERENCES = load_paste_references(ROOT / "data/public_paste_benchmark")
    SCENARIOS = {key: (props, options) for key, props, options in thermal_scenarios()}
    RADIAL = RadialSection()


def compute_history(task):
    started = perf_counter()
    row, tolerance = task
    props, options = SCENARIOS[row["scenario"]]
    inputs = {k: float(row[k]) for k in ("initial_reference_age_h", "initial_paste_temperature_c",
                                       "initial_ring_temperature_c", "soil_temperature_c")}
    run = simulate_paste(REFERENCES[row["mix_id"]], SECTION.geometry, np.linspace(0, 12, 1201),
                         air_temperature_c=20., properties=props, **inputs, **options)
    radius = run.mesh["radius_m"][run.mesh["interface"]:]
    end = min(6., run.time_h[-1])
    fields = {}

    def evaluate(t):
        t = float(t)
        if t not in fields:
            e = np.array([np.interp(t, run.time_h, column) for column in run.modulus_pa.T])
            fields[t] = (RADIAL.response(radius, e), e)
        return fields[t][0]

    sample_response(evaluate, end, DEMANDS, tolerance)
    times = np.array(sorted(fields))
    responses = np.array([fields[t][0] for t in times])
    hit = first_attainment(times, responses, 2.2)
    exact_hit = hit.time_h
    if hit.status == "attained":
        i = int(np.flatnonzero(responses <= 2.2)[0])
        exact_hit = brentq(lambda t: evaluate(t)-2.2, times[i-1], times[i], xtol=0.0002)
        evaluate(exact_hit)
    common = {k: row[k] for k in KEYS}
    history = []
    for t in sorted(fields):
        response, e = fields[t]
        mean = float(e @ run.mesh["paste_volume"] / run.mesh["paste_volume"].sum())
        history.append(dict(**common, time_h=t, response_um_kpa=response,
                            uniform_response_um_kpa=float(SECTION.compliance(mean))*1e9,
                            mean_modulus_gpa=mean/1e9, min_modulus_gpa=float(e.min()/1e9),
                            max_modulus_gpa=float(e.max()/1e9),
                            paste_max_temperature_c=float(max(np.interp(t, run.time_h, col)
                                for col in run.temperature_c[:, run.mesh["interface"]:].T))))
    times = np.array([r["time_h"] for r in history])
    responses = np.array([r["response_um_kpa"] for r in history])
    crossings = []
    for target in DEMANDS:
        event = first_attainment(times, responses, target)
        crossing = exact_hit if np.isclose(target, 2.2) else event.time_h
        crossings.append(dict(**common, target_um_kpa=float(target), time_h=crossing,
                              status=event.status, supported_end_h=end))
    summary = dict(**common, **inputs, thermal_end_h=float(run.time_h[-1]),
                   supported_end_h=end, end_reason=run.status, fe_solves=len(fields),
                   interpolation_time_tolerance_h=tolerance, wall_s=perf_counter()-started)
    return summary, history, crossings


def read_rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--case", action="append")
    parser.add_argument("--mix", action="append")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--time-tolerance", type=float, default=0.002)
    parser.add_argument("--resume", action="store_true", help="Reuse complete saved trajectories")
    args = parser.parse_args()
    rows = [r for r in read_rows("query_runs.csv") if r["case"] not in ("age_3", "age_6")]
    for column, selected in (("case", args.case), ("mix_id", args.mix), ("scenario", args.scenario)):
        if selected:
            rows = [r for r in rows if r[column] in selected]
    completed = set()
    if args.resume:
        completed = {tuple(r[k] for k in KEYS) for r in read_rows("radial_runs.csv")}
    pending = [r for r in rows if tuple(r[k] for k in KEYS) not in completed]
    if not pending:
        print("All requested radial histories are already complete.", flush=True)
        return
    streams = [(OUT / name).open("a" if args.resume else "w", newline="", encoding="utf-8")
               for name in ("radial_runs.csv", "radial_histories.csv", "radial_attainment.csv")]
    writers = None
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker) as pool:
            tasks = ((r, args.time_tolerance) for r in pending)
            for n, (summary, history, crossings) in enumerate(pool.map(compute_history, tasks), 1):
                if writers is None:
                    writers = [csv.DictWriter(s, fieldnames=sample)
                               for s, sample in zip(streams, (summary, history[0], crossings[0]))]
                    if not args.resume:
                        for writer in writers:
                            writer.writeheader()
                writers[1].writerows(history)
                writers[2].writerows(crossings)
                streams[1].flush()
                streams[2].flush()
                writers[0].writerow(summary)
                streams[0].flush()
                print(f"{n}/{len(pending)} {summary['scenario']} / {summary['case']} / "
                      f"{summary['mix_id']}: {summary['fe_solves']} FE states, "
                      f"{summary['wall_s']:.1f} s", flush=True)
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
