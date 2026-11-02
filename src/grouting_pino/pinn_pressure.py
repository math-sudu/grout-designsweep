"""Continuous event fields for uniform and prescribed right-sector pressures."""

import numpy as np

from grouting_pino.pinn_event import EventPINN
from grouting_pino.pinn_mechanics import pressure_profile


PRESSURE_MODES = ("uniform", "sector_even", "sector_odd")


def pressure_coefficients(values):
    if set(values) - {"uniform", "sector"}:
        raise ValueError("Pressure coefficients use uniform and sector modes")
    result = {key: float(values.get(key, 0.)) for key in ("uniform", "sector")}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Pressure coefficients must be finite")
    return result


class PressureSection:
    """A frozen linear combination of even and odd mixed spatial readouts."""

    def __init__(self, parts, coefficients):
        self.parts = parts
        self.coefficients = pressure_coefficients(coefficients)
        self.config = parts["uniform"].config | {"pressure_coefficients": self.coefficients}

    def query(self, domain, xy, derivatives=False):
        result = None
        for name, part in self.parts.items():
            amplitude = self.coefficients["uniform" if name == "uniform" else "sector"]
            if amplitude:
                fields = part.query(domain, xy, derivatives)
                if result is None:
                    result = {key: amplitude * value for key, value in fields.items()}
                else:
                    for key, value in fields.items():
                        result[key] += amplitude * value
        if result is None:
            result = {key: np.zeros_like(value) for key, value in
                      self.parts["uniform"].query(domain, xy, derivatives).items()}
        return result

    def modulus(self, domain, xy):
        return self.parts["uniform"].modulus(domain, xy)

    def pressure(self, theta):
        return sum(self.coefficients["uniform" if name == "uniform" else "sector"]
                   * pressure_profile(part.config, theta) for name, part in self.parts.items())


class PressureEventPINN:
    """The same trained pressure modes serve new event times and histories."""

    def __init__(self, modes):
        if set(modes) != set(PRESSURE_MODES):
            raise ValueError("Uniform, even-sector and odd-sector fields are required")
        self.modes = modes
        self.thermal = modes["uniform"].thermal
        self.angle_deg = modes["sector_even"].section.config["sector_angle_deg"]
        for name, model in modes.items():
            if model.section.config.get("pressure_mode", "uniform") != name:
                raise ValueError("Pressure mode metadata disagrees with its readout")
            parity = "odd" if name == "sector_odd" else "even"
            if model.section.config.get("reflection", "even") != parity:
                raise ValueError("Pressure and field reflection must agree")
            if (model.thermal.config["mix_id"] != self.thermal.config["mix_id"]
                    or not np.array_equal(model.knots, modes["uniform"].knots)
                    or model.section.config.get("sector_angle_deg", self.angle_deg) != self.angle_deg):
                raise ValueError("Pressure modes must share material, time interval and sector")

    def event(self, time_h, coefficients=None):
        coefficients = {"uniform": 1.} if coefficients is None else coefficients
        return PressureSection({name: model.event(time_h) for name, model in self.modes.items()},
                               coefficients)

    def mean_pressure_fraction(self, coefficients):
        values = pressure_coefficients(coefficients)
        return values["uniform"] + values["sector"] * self.angle_deg / 360.

    def freeze(self):
        for model in self.modes.values():
            model.freeze()
        return self

    def arrays(self):
        return {name + "__" + key: value for name, model in self.modes.items()
                for key, value in model.arrays().items()}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def from_arrays(cls, saved, thermal):
        modes = {}
        for name in PRESSURE_MODES:
            prefix = name + "__"
            arrays = {key[len(prefix):]: value for key, value in saved.items() if key.startswith(prefix)}
            modes[name] = EventPINN.from_arrays(arrays, thermal)
        return cls(modes)

    @classmethod
    def load(cls, path, thermal):
        with np.load(path) as saved:
            return cls.from_arrays(dict(saved), thermal)
