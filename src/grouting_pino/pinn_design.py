"""Surface-wise design queries for frozen neural event fields.

Time scans resolve feasible sets without assuming a single successor window.
Crossing brackets retain both signs; the reported admissible endpoint is the
evaluated feasible side. The scan spacing is a numerical resolution, not a
proof that no narrower component exists between samples.
"""

from copy import copy
from time import perf_counter

import numpy as np
from scipy.optimize import linprog, minimize_scalar

from grouting_pino.pinn_mechanics import PRESSURE
from grouting_pino.pinn_pressure import pressure_coefficients


def surface_response(models, fractions, x):
    x = np.asarray(x, dtype=float)
    xy = np.column_stack([x, np.zeros_like(x)])
    return sum(-weight * model.query("soil", xy)["uy"] * 1e9 / PRESSURE
               for model, weight in zip(models, fractions, strict=True))


def surface_maximum(function, x=None):
    """Refine every sampled local maximum, including possible off-axis peaks."""
    x = np.linspace(-30, 30, 1201) if x is None else np.asarray(x, dtype=float)
    values = np.asarray(function(x))
    candidates = [(float(values[i]), float(x[i])) for i in (0, int(values.argmax()), len(x) - 1)]
    local = np.flatnonzero((values[1:-1] >= values[:-2])
                           & (values[1:-1] >= values[2:])
                           & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))) + 1
    for i in local:
        candidates.append((float(values[i]), float(x[i])))
        result = minimize_scalar(lambda at: -float(function(np.array([at]))[0]),
                                 bounds=(x[i - 1], x[i + 1]), method="bounded",
                                 options={"xatol": 1e-6})
        candidates.append((-float(result.fun), float(result.x)))
    peak, location = max(candidates)
    return {"peak_um_kpa": peak, "peak_x_m": location,
            "grid_peak_um_kpa": float(values.max()),
            "peak_refinement_um_kpa": peak - float(values.max())}


def surface_extrema(function, x=None):
    """Resolve settlement and uplift from the signed downward surface field."""
    down = surface_maximum(function, x)
    up = surface_maximum(lambda at: -np.asarray(function(at)), x)
    controlling = down if down["peak_um_kpa"] >= up["peak_um_kpa"] else up
    return {**down, "uplift_um_kpa": up["peak_um_kpa"], "uplift_x_m": up["peak_x_m"],
            "absolute_peak_um_kpa": controlling["peak_um_kpa"],
            "absolute_peak_x_m": controlling["peak_x_m"]}


def displacement_margins(extrema, lower, upper):
    """Bounds use signed downward displacement in micrometres per kPa."""
    settlement = upper - extrema["peak_um_kpa"]
    uplift = -lower - extrema["uplift_um_kpa"]
    return {"settlement_margin_um_kpa": settlement, "uplift_margin_um_kpa": uplift,
            "minimum_margin_um_kpa": min(settlement, uplift)}


def solve_prescribed_plan(family, times, minimum_fractions, target_pressure,
                          lower=-2.2, upper=2.2, initial=None, x=None):
    """Allocate a prescribed remaining pressure shape over fixed event times.

    Fractions partition target minus saved pressure, with nonnegative physical
    pressure increments inside and outside the sector. Minimum cumulative
    fractions refer to circumferential mean pressure / final mean pressure.
    The linear program maximizes the smallest signed displacement margin over
    the initial state and all stages. A negative optimum retains a complete
    minimum-violation witness, explicitly distinguished from a feasible plan.
    """
    started = perf_counter()
    times = np.asarray(times, dtype=float)
    minima = np.asarray(minimum_fractions, dtype=float)
    target = pressure_coefficients(target_pressure)
    if (times.ndim != 1 or not len(times) or times.shape != minima.shape
            or not np.isfinite(times).all() or not np.isfinite(minima).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(minima) < 0)
            or np.any((minima < 0) | (minima > 1))):
        raise ValueError("Need increasing event times and cumulative requirements in [0, 1]")
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Need finite, ordered displacement bounds")
    initial_time = initial.time_h if initial is not None else 0.
    if times[0] < initial_time:
        raise ValueError("Plan cannot precede the saved material clock")
    if initial is not None:
        family = initial.compatible_family(family)
    saved = pressure_coefficients(initial.load_coefficients if initial is not None else {})
    remaining = {key: target[key] - saved[key] for key in target}
    for coefficients in (saved, target, remaining):
        if min(coefficients["uniform"], coefficients["uniform"] + coefficients["sector"]) < -1e-12:
            raise ValueError("Prescribed withdrawal must be nonnegative inside and outside the sector")
    target_mean = family.mean_pressure_fraction(target)
    if target_mean <= 0:
        raise ValueError("The prescribed final mean pressure must be positive")
    saved_mean = family.mean_pressure_fraction(saved)
    remaining_mean = family.mean_pressure_fraction(remaining)
    readout_start = perf_counter()
    models = [family.event(float(t), remaining) for t in times]
    readout_s = perf_counter() - readout_start
    query_s = 0.

    def base(at):
        if initial is None:
            return np.zeros(len(at))
        return -initial.query("soil", np.c_[at, np.zeros_like(at)])["uy"] * 1e9 / PRESSURE

    def evaluate(at):
        nonlocal query_s
        tick = perf_counter()
        result = base(at), np.array([surface_response([model], [1.], at) for model in models])
        query_s += perf_counter() - tick
        return result

    x = np.linspace(-30., 30., 1201) if x is None else np.asarray(x, dtype=float)
    grid = x.copy()
    s0, fields = evaluate(grid)
    n = len(times)
    cumulative = np.tril(np.ones((n, n)))
    demand = minima * target_mean - saved_mean
    solve_s, solves = 0., 0
    # This tolerance resolves LP/field roundoff, not model accuracy.
    tolerance = 1e-7
    while True:
        constraints, rhs = [], []
        for stage in range(n + 1):
            prefix = np.arange(n) < stage
            response = (fields * prefix[:, None]).T
            constraints.extend([np.c_[response, np.ones(len(grid))],
                                np.c_[-response, np.ones(len(grid))]])
            rhs.extend([upper - s0, s0 - lower])
        constraints.append(np.c_[-remaining_mean * cumulative, np.zeros(n)])
        rhs.append(-demand)
        tick = perf_counter()
        solution = linprog(np.r_[np.zeros(n), -1.], A_ub=np.vstack(constraints),
            b_ub=np.concatenate(rhs), A_eq=[np.r_[np.ones(n), 0.]], b_eq=[1.],
            bounds=[(0., 1.)] * n + [(None, None)], method="highs",
            options={"primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9})
        solve_s += perf_counter() - tick
        solves += 1
        if not solution.success:
            return {"status": "load_requirements_infeasible" if solution.status == 2 else "solver_failure",
                    "solver_status": int(solution.status), "solver_message": solution.message,
                    "events": [], "stages": [], "timing_s": {"total": perf_counter() - started}}
        fractions = solution.x[:n]
        extrema = []
        for stage in range(n + 1):
            def response(at):
                nonlocal query_s
                tick = perf_counter()
                values = base(at)
                if stage:
                    values = values + surface_response(models[:stage], fractions[:stage], at)
                query_s += perf_counter() - tick
                return values
            extrema.append(surface_extrema(response, grid))
        margins = [displacement_margins(row, lower, upper) for row in extrema]
        minimum = min(row["minimum_margin_um_kpa"] for row in margins)
        if solution.x[-1] - minimum <= tolerance:
            break
        # Add the actual active positions; do not substitute a sum of maxima.
        points = np.unique([value for row in extrema for value in
                            (row["peak_x_m"], row["uplift_x_m"])])
        points = points[~np.isin(points, grid)]
        if not len(points):
            return {"status": "solver_failure", "solver_message": "Spatial refinement stagnated",
                    "events": [], "stages": [], "timing_s": {"total": perf_counter() - started}}
        extra0, extra = evaluate(points)
        order = np.argsort(np.r_[grid, points])
        grid = np.r_[grid, points][order]
        s0 = np.r_[s0, extra0][order]
        fields = np.c_[fields, extra][:, order]
    stages = [{"time_h": float(initial_time), "cumulative_pressure": saved,
               "cumulative_fraction": saved_mean / target_mean,
               "remaining_pressure": remaining, **extrema[0], **margins[0]}]
    events = []
    for i, (t, fraction) in enumerate(zip(times, fractions, strict=True)):
        cumulative_fraction = float(fractions[:i + 1].sum())
        total = {key: saved[key] + cumulative_fraction * remaining[key] for key in target}
        increment = {key: float(fraction * remaining[key]) for key in target}
        events.append({"time_h": float(t), "remaining_fraction_increment": float(fraction),
                       "pressure_increment": increment, "cumulative_pressure": total})
        achieved = family.mean_pressure_fraction(total) / target_mean
        stages.append({"time_h": float(t), "cumulative_pressure": total,
                       "cumulative_fraction": achieved,
                       "minimum_cumulative_fraction": float(minima[i]),
                       "load_requirement_margin": achieved - float(minima[i]),
                       "remaining_pressure": {key: target[key] - total[key] for key in target},
                       **extrema[i + 1], **margins[i + 1]})
    return {"status": "feasible" if minimum >= -tolerance else "displacement_infeasible",
            "solution_role": "plan" if minimum >= -tolerance else "minimum_violation_witness",
            "target_pressure": target, "initial_pressure": saved, "remaining_pressure": remaining,
            "target_mean_pressure_fraction": target_mean,
            "displacement_bounds_um_kpa": [lower, upper],
            "minimum_margin_um_kpa": minimum, "sampled_margin_bound_um_kpa": float(solution.x[-1]),
            "numerical_tolerance_um_kpa": tolerance, "events": events, "stages": stages,
            "spatial_constraint_points": len(grid), "linear_solves": solves,
            "timing_s": {"event_readout": readout_s, "surface_query": query_s,
                         "linear_solve": solve_s, "total": perf_counter() - started}}


def replay_prescribed_plan(family, plan, initial=None):
    """Replay saved cumulative events without changing the supplied history."""
    from grouting_pino.pinn_history import PINNHistory

    if not plan["events"]:
        raise ValueError("No complete event list is available to replay")
    if initial is None:
        history = PINNHistory(family.thermal)
        history.bind_family(family)
    else:
        family = initial.compatible_family(family)
        history = copy(initial)
        history.events = list(initial.events)
        history.mode_events = list(initial.mode_events)
        history.load_coefficients = initial.load_coefficients.copy()
        if history.family is None:
            history.bind_family(family)
    for event in plan["events"]:
        history.advance(event["time_h"])
        history.apply_modes(event["cumulative_pressure"])
        yield history


def feasible_intervals(excess, times, bracket_width_h=0.01):
    """Resolve all sign changes in an ordered scan and keep feasible endpoints."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or bracket_width_h <= 0:
        raise ValueError("Need increasing scan times and a positive bracket width")
    values = np.array([excess(float(t)) for t in times])
    if not np.isfinite(values).all():
        raise ValueError("Design margins must be finite on the valid time interval")
    intervals, crossings = [], []
    start = float(times[0]) if values[0] <= 0 else None
    for a, b, fa, fb in zip(times[:-1], times[1:], values[:-1], values[1:]):
        if (fa <= 0) == (fb <= 0):
            continue
        entering = fb <= 0
        while b - a > bracket_width_h:
            mid = float((a + b) / 2)
            fm = float(excess(mid))
            if not np.isfinite(fm):
                raise ValueError("A crossing query returned a nonfinite margin")
            if (fm <= 0) == (fa <= 0):
                a, fa = mid, fm
            else:
                b, fb = mid, fm
        crossings.append({"lower_h": float(a), "upper_h": float(b),
                          "lower_excess_um_kpa": float(fa),
                          "upper_excess_um_kpa": float(fb),
                          "direction": "enter" if entering else "leave"})
        if entering:
            start = float(b)
        else:
            intervals.append([start, float(a)])
            start = None
    if start is not None:
        intervals.append([start, float(times[-1])])
    return {"intervals_h": intervals, "crossings": crossings,
            "scan_time_h": times.tolist(), "scan_excess_um_kpa": values.tolist()}


def earliest_in_window(intervals, plan, deadline):
    candidates = [max(plan, a) for a, b in intervals if max(plan, a) <= min(deadline, b)]
    return min(candidates) if candidates else None


def compare_plans(materials, planned_responses, plans, deadline, reference_mix):
    """B0 feasibility, B1 at the plan, B2 over each material's valid time sets."""
    by_mix = {row["mix_id"]: row for row in materials}
    result = []
    for plan in plans:
        fixed = [by_mix[row["mix_id"]] for row in planned_responses
                 if row["planned_handover_h"] == plan and row["status"] == "feasible"]
        extended = [(row, earliest_in_window(row["handover"]["intervals_h"], plan, deadline))
                    for row in materials]
        extended = [(row, time) for row, time in extended if time is not None]
        b0 = by_mix[reference_mix] if by_mix[reference_mix] in fixed else None
        b1 = min(fixed, key=lambda row: row["cement_kg_m3"]) if fixed else None
        b2, actual = min(extended, key=lambda pair: pair[0]["cement_kg_m3"]) if extended else (None, None)
        result.append({"planned_handover_h": plan, "latest_handover_h": deadline,
                       "B0_status": "feasible" if b0 else "reference_infeasible",
                       "B0_mix": b0["mix_id"] if b0 else None,
                       "B1_mix": b1["mix_id"] if b1 else None,
                       "B2_mix": b2["mix_id"] if b2 else None,
                       "material_saving_kg_m3": b0["cement_kg_m3"] - b1["cement_kg_m3"] if b0 and b1 else None,
                       "timing_saving_kg_m3": b1["cement_kg_m3"] - b2["cement_kg_m3"] if b1 and b2 else None,
                       "adjusted_handover_h": actual,
                       "added_cycle_time_min": (actual - plan) * 60 if actual is not None else None})
    return result
