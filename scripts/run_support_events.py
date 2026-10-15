"""Select materials for one loading event before a prescribed latest time.

Reuse radial-FE first attainments and history endpoints. The prospective
candidate set starts from the common observed state of the placed annulus;
the independent age_3/age_6 initializations are not construction checkpoints.
"""

import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.paste_thermal import PasteThermalProperties
from grouting_pino.support_response import Attainment, loading_event

OUT = ROOT / "results/support_design"


def summarize_events(attainments, latest_times_h, cement_mass, assumption_set):
    """Minimize cement while requiring an available event in every scenario.

    One observed failure excludes a candidate from the all-scenario set. A
    candidate with missing events and no observed failure remains unresolved.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in attainments:
        grouped[(row["case"], row["target_um_kpa"])][row["mix_id"]].append(row)
    designs = []
    for (case, target), candidates in grouped.items():
        for latest in latest_times_h:
            feasible, excluded, unresolved, latest_hits = [], [], [], {}
            for mix, rows in candidates.items():
                events = [loading_event(Attainment(r["time_h"], r["status"]),
                                        r["supported_end_h"], latest) for r in rows]
                statuses = {e.status for e in events}
                if statuses == {"available"}:
                    feasible.append(mix)
                    latest_hits[mix] = max(e.earliest_time_h for e in events)
                elif "unavailable" in statuses:
                    excluded.append(mix)
                else:
                    unresolved.append(mix)
            feasible.sort(key=cement_mass.get)
            best = feasible[0] if feasible else None
            resolved = (not unresolved if best is None else
                        all(cement_mass[m] >= cement_mass[best] for m in unresolved))
            designs.append(dict(
                assumption_set=assumption_set, case=case, target_um_kpa=target,
                latest_loading_time_h=float(latest),
                feasible_all_evaluated=";".join(feasible),
                infeasible_some_evaluated=";".join(excluded), domain_limited=";".join(unresolved),
                least_cement_all_evaluated_mix=best or "",
                cement_kg_m3=cement_mass[best] if best else None,
                selected_latest_attainment_h=latest_hits.get(best),
                least_cement_choice_resolved=resolved))
    return designs


def frontier_segments(attainments, cement_mass, assumption_set, latest_h=6.0):
    """Exact constant-choice intervals, including censored-window boundaries.

    An attained event becomes available at its crossing. An unobserved event
    remains excluded at its history endpoint and becomes unknown after it.
    Evaluate both boundary points and open intervals to retain that distinction.
    """
    grouped = defaultdict(list)
    for row in attainments:
        grouped[(row["case"], row["target_um_kpa"])].append(row)
    segments = []
    for rows in grouped.values():
        boundaries = {0.0, latest_h}
        for row in rows:
            boundary = row["time_h"] if row["time_h"] is not None else row["supported_end_h"]
            if 0 <= boundary <= latest_h:
                boundaries.add(boundary)
        boundaries = sorted(boundaries)
        pieces = []
        for i, start in enumerate(boundaries):
            pieces.append((start, start, True, True, start))
            if i + 1 < len(boundaries):
                end = boundaries[i + 1]
                pieces.append((start, end, False, False, (start + end) / 2))
        current = None
        for start, end, left, right, query in pieces:
            choice = summarize_events(rows, [query], cement_mass, assumption_set)[0]
            del choice["latest_loading_time_h"]
            if current is not None and choice == current["choice"]:
                current["end_h"], current["end_inclusive"] = end, right
            else:
                current = dict(start_h=start, end_h=end, start_inclusive=left,
                               end_inclusive=right, choice=choice)
                segments.append(current)
    return [dict(**s["choice"], start_h=s["start_h"], end_h=s["end_h"],
                 start_inclusive=s["start_inclusive"], end_inclusive=s["end_inclusive"])
            for s in segments]


def adjacent_tradeoffs(segments):
    """Cement reduction and added time between adjacent resolved choices."""
    trades = []
    for previous, current in zip(segments[:-1], segments[1:]):
        keys = ("assumption_set", "case", "target_um_kpa")
        if any(previous[k] != current[k] for k in keys):
            continue
        if not all(r["least_cement_choice_resolved"] and r["cement_kg_m3"] is not None
                   for r in (previous, current)):
            continue
        saving = previous["cement_kg_m3"] - current["cement_kg_m3"]
        if saving <= 0:
            continue
        extra = current["selected_latest_attainment_h"] - previous["selected_latest_attainment_h"]
        trades.append(dict(
            **{k: current[k] for k in keys},
            from_mix=previous["least_cement_all_evaluated_mix"],
            to_mix=current["least_cement_all_evaluated_mix"],
            from_time_h=previous["selected_latest_attainment_h"],
            to_time_h=current["selected_latest_attainment_h"],
            extra_time_h=extra, cement_saved_kg_m3=saving,
            cement_saved_per_added_hour_kg_m3_h=saving / extra))
    return trades


def load_events():
    with (OUT / "radial_attainment.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["target_um_kpa"] = float(row["target_um_kpa"])
        row["time_h"] = float(row["time_h"]) if row["time_h"] else None
        row["supported_end_h"] = float(row["supported_end_h"])
    with (OUT / "query_runs.csv").open(newline="", encoding="utf-8") as stream:
        expected = {(r["scenario"], r["case"], r["mix_id"], round(float(target), 9))
                    for r in csv.DictReader(stream) if r["case"] not in ("age_3", "age_6")
                    for target in np.linspace(1.5, 3.5, 41)}
    actual = [(r["scenario"], r["case"], r["mix_id"], round(r["target_um_kpa"], 9)) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Radial events must cover each material, case, scenario and demand exactly once")
    return rows


def main():
    rows = load_events()
    props = PasteThermalProperties()
    masses = {m: props.volume_properties(r.water_cement_ratio)["cement_kg_m3"]
              for m, r in load_paste_references(ROOT / "data/public_paste_benchmark").items()}
    latest = np.linspace(0, 6, 61)
    designs, frontier = [], []
    for assumption_set, selected in (("nominal", [r for r in rows if r["scenario"] == "nominal"]),
                                     ("thermal", rows)):
        designs += summarize_events(selected, latest, masses, assumption_set)
        frontier += frontier_segments(selected, masses, assumption_set)
    trades = adjacent_tradeoffs(frontier)
    for name, records in (("event_designs.csv", designs), ("cement_time_frontier.csv", frontier),
                          ("cement_time_tradeoffs.csv", trades)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    print(f"Saved {len(designs)} event selections, {len(frontier)} frontier intervals "
          f"and {len(trades)} adjacent material tradeoffs.")


if __name__ == "__main__":
    main()
