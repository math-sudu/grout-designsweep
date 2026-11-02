"""Cached independent events and interpolation of their mixed neural readouts."""

from copy import copy

import numpy as np
from scipy.interpolate import CubicSpline

from grouting_pino.pinn_pressure import PressureSection, PRESSURE_MODES, pressure_coefficients


class IndependentPressureEvents:
    """Train only requested time/mode pairs, sharing the spatial operators."""

    def __init__(self, template, operators):
        self.template = template
        self.operators = operators
        self.thermal = template.thermal
        self.angle_deg = template.angle_deg
        self.parts = {}
        self.training = []

    def mean_pressure_fraction(self, coefficients):
        return self.template.mean_pressure_fraction(coefficients)

    def freeze(self):
        return self

    def event(self, time_h, coefficients=None):
        values = pressure_coefficients({"uniform": 1.} if coefficients is None else coefficients)
        required = {"uniform"} if values["uniform"] else set()
        if values["sector"]:
            required.update(("sector_even", "sector_odd"))
        current = self.parts.setdefault(float(time_h), {})
        missing = required - current.keys()
        if missing and self.operators is None:
            raise ValueError(f"Independent event {time_h} lacks modes {sorted(missing)}")
        for operator in self.operators or ():
            names = {model.section.config.get("pressure_mode", "uniform") for model in operator.models}
            if names & missing:
                parts, record = operator.independent(time_h, modes=missing)
                current.update(parts)
                self.training.append(record)
        # The uniform template supplies configuration when its amplitude is zero.
        return PressureSection({"uniform": self.template.modes["uniform"].section, **current}, values)

    def arrays(self):
        return {f"{t:.12g}__{name}": section.weights for t, parts in self.parts.items()
                for name, section in parts.items()}


class InterpolatedPressureEvents:
    """Cubic interpolation of independently trained mixed output coefficients."""

    def __init__(self, events, times):
        self.thermal = events.thermal
        self.template = events.template
        self.angle_deg = events.angle_deg
        self.times = np.asarray(times, dtype=float)
        if len(self.times) < 4 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Cubic event interpolation requires at least four increasing times")
        for t in self.times:
            events.event(t, {"uniform": 1., "sector": 1.})
        self.readouts = {name: np.array([events.parts[t][name].weights for t in self.times])
                         for name in PRESSURE_MODES}
        self.interpolants = {name: CubicSpline(self.times, weights, axis=0, extrapolate=False)
                             for name, weights in self.readouts.items()}

    def mean_pressure_fraction(self, coefficients):
        return self.template.mean_pressure_fraction(coefficients)

    def freeze(self):
        return self

    def event(self, time_h, coefficients=None):
        if not np.isfinite(time_h) or time_h < self.times[0] or time_h > self.times[-1]:
            raise ValueError("Event time exceeds the interpolation library")
        values = {"uniform": 1.} if coefficients is None else coefficients
        parts = {}
        for name in PRESSURE_MODES:
            section = copy(self.template.modes[name].section)
            section.config = section.config | {"time_h": float(time_h)}
            section.weights = self.interpolants[name](time_h)
            section.paste_modulus = lambda r: self.thermal.query(
                "paste", np.clip(r, 5.8, 6.02), time_h)["modulus_pa"]
            parts[name] = section
        return PressureSection(parts, values)
