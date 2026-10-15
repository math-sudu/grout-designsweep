"""Section demand and first attainment under a shared support target.

Compliance is the magnitude of the peak downward surface displacement per
unit inward interface traction, in m/Pa. Each section is tabulated against
grout modulus. Interpolation is linear in log modulus, as in the FE tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _target(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Target compliance must be finite and nonnegative")
    return value


@dataclass
class SectionResponse:
    modulus_pa: np.ndarray
    compliance_m_per_pa: np.ndarray
    geometry: dict

    def __post_init__(self) -> None:
        self.modulus_pa = np.asarray(self.modulus_pa, dtype=float)
        self.compliance_m_per_pa = np.asarray(self.compliance_m_per_pa, dtype=float)
        e, c = self.modulus_pa, self.compliance_m_per_pa
        if e.ndim != 1 or e.size < 2 or c.shape != e.shape:
            raise ValueError("Response table needs matching one-dimensional arrays")
        if not np.all(np.isfinite(e)) or np.any(e <= 0) or np.any(np.diff(e) <= 0):
            raise ValueError("Tabulated moduli must be positive and strictly increasing")
        if not np.all(np.isfinite(c)) or np.any(c < 0):
            raise ValueError("Tabulated compliance must be finite and nonnegative")

    @classmethod
    def load(cls, path: str | Path) -> SectionResponse:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["modulus_pa"], data["compliance_m_per_pa"],
                       json.loads(str(data["geometry_json"])))

    def compliance(self, modulus_pa: np.ndarray | float) -> np.ndarray | float:
        """Evaluate within the tabulated modulus range, without extrapolation."""
        e = np.asarray(modulus_pa, dtype=float)
        if (not np.all(np.isfinite(e)) or np.any(e < self.modulus_pa[0])
                or np.any(e > self.modulus_pa[-1])):
            raise ValueError(
                f"Modulus lies outside the response table "
                f"[{self.modulus_pa[0]}, {self.modulus_pa[-1]}] Pa")
        return np.interp(np.log(e), np.log(self.modulus_pa), self.compliance_m_per_pa)

    def admissible_modulus_intervals(self, target_m_per_pa: float) -> list[tuple[float, float]]:
        """Closed modulus intervals satisfying C(E) <= target within this table.

        Segment crossings preserve the original table, including disconnected
        feasible intervals when a section response is nonmonotone.
        """
        target = _target(target_m_per_pa)
        e, c = self.modulus_pa, self.compliance_m_per_pa
        intervals: list[tuple[float, float]] = []
        for i in range(e.size - 1):
            left, right = c[i] <= target, c[i + 1] <= target
            if not left and not right:
                continue
            lo, hi = float(e[i]), float(e[i + 1])
            if left != right:
                fraction = (target - c[i]) / (c[i + 1] - c[i])
                if fraction == 0:
                    crossing = lo
                elif fraction == 1:
                    crossing = hi
                else:
                    crossing = float(np.exp(np.log(lo) + fraction * np.log(hi / lo)))
                if left:
                    hi = crossing
                else:
                    lo = crossing
            if intervals and lo <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], hi)
            else:
                intervals.append((lo, hi))
        return intervals

    def required_modulus_pa(self, target_m_per_pa: float) -> float | None:
        """Lowest admissible tabulated modulus for a nonincreasing response.

        None means the target is stricter than this table covers. A nonmonotone
        response must be queried through admissible_modulus_intervals instead.
        """
        if np.any(np.diff(self.compliance_m_per_pa) > 0):
            raise ValueError("Response is nonmonotone; use admissible_modulus_intervals")
        intervals = self.admissible_modulus_intervals(target_m_per_pa)
        return intervals[0][0] if intervals else None


@dataclass(frozen=True)
class Attainment:
    time_h: float | None
    status: str


def _history(time_h: np.ndarray, compliance_m_per_pa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_h, dtype=float)
    response = np.asarray(compliance_m_per_pa, dtype=float)
    if time.ndim != 1 or time.size < 2 or response.shape != time.shape:
        raise ValueError("History needs matching one-dimensional time and response arrays")
    if (not np.all(np.isfinite(time)) or time[0] != 0
            or np.any(np.diff(time) <= 0)):
        raise ValueError("History time must start at the specified initial state (0 h) and strictly increase")
    if not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("History compliance must be finite and nonnegative")
    return time, response


def first_attainment(time_h: np.ndarray, compliance_m_per_pa: np.ndarray,
                     target_m_per_pa: float) -> Attainment:
    """First C(t) <= target, linearly interpolating sampled response in time."""
    time, response = _history(time_h, compliance_m_per_pa)
    target = _target(target_m_per_pa)
    reached = np.flatnonzero(response <= target)
    if reached.size == 0:
        return Attainment(None, "not_reached_in_window")
    i = int(reached[0])
    if i == 0:
        return Attainment(0.0, "initially_met")
    fraction = (response[i - 1] - target) / (response[i - 1] - response[i])
    crossing = time[i - 1] + fraction * (time[i] - time[i - 1])
    return Attainment(float(crossing), "attained")


@dataclass(frozen=True)
class LoadingEvent:
    earliest_time_h: float | None
    remaining_time_h: float | None
    status: str


def loading_event(attainment: Attainment, supported_end_h: float,
                  latest_loading_time_h: float, *, state_time_h: float = 0.0) -> LoadingEvent:
    """Schedule one load increment by its latest time on the construction clock.

    Attainment and supported_end_h are relative to the supplied state. A known
    event can be scheduled at its observed time even when the latest allowed
    time is beyond the evaluated history. A state update queries the trajectory
    starting at that checkpoint, so it never schedules an action in the past.
    """
    if (not np.all(np.isfinite([supported_end_h, latest_loading_time_h, state_time_h]))
            or supported_end_h < 0 or state_time_h < 0
            or latest_loading_time_h < state_time_h):
        raise ValueError("Loading window must start at a finite construction time and end at or after it")
    wait = attainment.time_h
    if wait is not None:
        if not np.isfinite(wait) or not 0 <= wait <= supported_end_h:
            raise ValueError("Attainment must lie within the evaluated history")
        earliest = state_time_h + wait
        status = "available" if earliest <= latest_loading_time_h else "unavailable"
        return LoadingEvent(earliest, wait, status)
    status = "unavailable" if state_time_h + supported_end_h >= latest_loading_time_h else "unresolved"
    return LoadingEvent(None, None, status)


def meets_deadline(time_h: np.ndarray, compliance_m_per_pa: np.ndarray,
                   target_m_per_pa: float, deadline_h: float) -> bool:
    """Evaluate support at the deadline, including histories that lose support."""
    time, response = _history(time_h, compliance_m_per_pa)
    target = _target(target_m_per_pa)
    if not np.isfinite(deadline_h) or not time[0] <= deadline_h <= time[-1]:
        raise ValueError("Deadline must be within the sampled history")
    return bool(np.interp(deadline_h, time, response) <= target)


def deadline_responses(time_h, compliance_m_per_pa, deadlines_h):
    """Reuse a supported trajectory; later instantaneous responses are unknown.

    A first attainment already observed in this trajectory remains a known
    event. It does not supply a response at a later, censored deadline.
    """
    time, response = _history(time_h, compliance_m_per_pa)
    deadlines = np.asarray(deadlines_h, dtype=float)
    if deadlines.ndim != 1 or not np.all(np.isfinite(deadlines)) or np.any(deadlines < 0):
        raise ValueError("Deadlines must be a finite nonnegative one-dimensional array")
    supported = deadlines <= time[-1]
    values = np.full(deadlines.shape, np.nan)
    values[supported] = np.interp(deadlines[supported], time, response)
    return values


def classify_scenarios(responses, target_m_per_pa):
    """Classify one material under a finite collection of model assumptions.

    NaN marks a deadline outside a scenario's supported history. Agreement
    across these evaluations is not a probability or a bound over untested
    parameters. Known disagreement remains visible even if others are censored.
    """
    values = np.asarray(responses, dtype=float)
    target = _target(target_m_per_pa)
    if values.ndim != 1 or not values.size or np.any(np.isinf(values)) or np.any(values < 0):
        raise ValueError("Scenario responses must be nonnegative values or censored NaNs")
    known = np.isfinite(values)
    feasible = int(np.count_nonzero(values[known] <= target))
    infeasible = int(known.sum()) - feasible
    unknown = len(values) - int(known.sum())
    if feasible and infeasible:
        state = "assumption_sensitive"
    elif unknown:
        state = "domain_limited"
    elif feasible:
        state = "feasible_all_evaluated"
    else:
        state = "infeasible_all_evaluated"
    return dict(status=state, feasible_count=feasible, infeasible_count=infeasible,
                outside_domain_count=unknown)
