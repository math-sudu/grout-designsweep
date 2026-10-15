"""Paired 20 C heat-flow and EMM-ARM curves for three public paste recipes.

The reaction coordinate is observed heat increment, J/kg dry cement, from
the common first supported age. It is not an absolute hydration fraction.
Piecewise-linear heat flow is integrated and inverted analytically, so the
heat rate and stiffness use the same material age without a fitted lag.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np


def _curves(path: Path, quantity: str) -> dict[str, np.ndarray]:
    rows: dict[str, list] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.setdefault(row["mix_id"], []).append(
                (float(row["age_h"]), float(row[quantity])))
    return {key: np.array(value) for key, value in rows.items()}


def _in_range(value, low: float, high: float, quantity: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < low) or np.any(values > high):
        raise ValueError(f"{quantity} must be in the measured curve domain [{low}, {high}]")
    return values


@dataclass
class PasteReference:
    mix_id: str
    water_cement_ratio: float
    temperature_c: float
    heat_age_h: np.ndarray
    heat_rate_w_kg_cement: np.ndarray
    modulus_age_h: np.ndarray
    modulus_pa: np.ndarray
    start_age_h: float
    end_age_h: float

    def __post_init__(self) -> None:
        for time, value in [(self.heat_age_h, self.heat_rate_w_kg_cement),
                            (self.modulus_age_h, self.modulus_pa)]:
            if (time.ndim != 1 or len(time) < 2 or value.shape != time.shape
                    or not np.all(np.isfinite(time)) or not np.all(np.isfinite(value))
                    or np.any(np.diff(time) <= 0)):
                raise ValueError("Material curves need finite values and increasing ages")
        if (np.any(self.heat_rate_w_kg_cement <= 0) or np.any(self.modulus_pa < 0)
                or not max(self.heat_age_h[0], self.modulus_age_h[0])
                <= self.start_age_h < self.end_age_h
                <= min(self.heat_age_h[-1], self.modulus_age_h[-1])):
            raise ValueError("Invalid heat rate, modulus or paired material domain")
        seconds = self.heat_age_h * 3600
        self._slopes = np.diff(self.heat_rate_w_kg_cement) / np.diff(seconds)
        areas = np.diff(seconds) * (self.heat_rate_w_kg_cement[:-1]
                                    + self.heat_rate_w_kg_cement[1:]) / 2
        self._integral = np.r_[0.0, np.cumsum(areas)]
        self._offset = float(self._raw_heat(self.start_age_h))
        self.max_heat_j_kg_cement = float(self._raw_heat(self.end_age_h) - self._offset)

    @property
    def cement_mass_fraction(self) -> float:
        return 1 / (1 + self.water_cement_ratio)

    def _raw_heat(self, age_h) -> np.ndarray:
        age = np.asarray(age_h)
        i = np.clip(np.searchsorted(self.heat_age_h, age, side="right") - 1,
                    0, len(self.heat_age_h) - 2)
        dt = (age - self.heat_age_h[i]) * 3600
        return self._integral[i] + self.heat_rate_w_kg_cement[i] * dt + self._slopes[i] * dt**2 / 2

    def heat_at_age(self, age_h) -> np.ndarray:
        age = _in_range(age_h, self.start_age_h, self.end_age_h, "Age (h)")
        return self._raw_heat(age) - self._offset

    def age_at_heat(self, heat_j_kg_cement) -> np.ndarray:
        heat = _in_range(heat_j_kg_cement, 0, self.max_heat_j_kg_cement,
                         "Heat increment (J/kg cement)") + self._offset
        i = np.clip(np.searchsorted(self._integral, heat, side="right") - 1,
                    0, len(self._integral) - 2)
        dq = heat - self._integral[i]
        q0 = self.heat_rate_w_kg_cement[i]
        q1 = np.sqrt(q0**2 + 2 * self._slopes[i] * dq)
        # This form also covers constant heat-flow segments without dividing
        # by their zero slope or subtracting nearly equal square roots.
        dt = 2 * dq / (q0 + q1)
        return self.heat_age_h[i] + dt / 3600

    def rate_at_heat(self, heat_j_kg_cement) -> np.ndarray:
        age = self.age_at_heat(heat_j_kg_cement)
        return np.interp(age, self.heat_age_h, self.heat_rate_w_kg_cement)

    def modulus_at_heat(self, heat_j_kg_cement) -> np.ndarray:
        age = self.age_at_heat(heat_j_kg_cement)
        return np.interp(age, self.modulus_age_h, self.modulus_pa)

    def first_heat_at_modulus(self, modulus_pa: float) -> float | None:
        """First observed heat state reaching a modulus, retaining plateau starts."""
        if not np.isfinite(modulus_pa) or modulus_pa < 0:
            raise ValueError("Required modulus must be finite and nonnegative")
        ages = np.r_[self.start_age_h,
                     self.modulus_age_h[(self.modulus_age_h > self.start_age_h)
                                        & (self.modulus_age_h < self.end_age_h)],
                     self.end_age_h]
        moduli = np.interp(ages, self.modulus_age_h, self.modulus_pa)
        reached = np.flatnonzero(moduli >= modulus_pa)
        if not reached.size:
            return None
        i = reached[0]
        if i == 0:
            return 0.0
        age = ages[i-1] + (ages[i]-ages[i-1]) * (modulus_pa-moduli[i-1]) / (moduli[i]-moduli[i-1])
        return float(self.heat_at_age(age))


def load_paste_references(directory: str | Path) -> dict[str, PasteReference]:
    """Load all recipes on their shared published age interval."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    heat = _curves(directory / "heat_flow.csv", "heat_rate_w_kg_cement")
    modulus = _curves(directory / "modulus.csv", "modulus_pa")
    recipes = metadata["recipes"]
    start = max(curves[m][0, 0] for m in recipes for curves in (heat, modulus))
    end = min(curves[m][-1, 0] for m in recipes for curves in (heat, modulus))
    return {
        m: PasteReference(m, r["water_kg"] / r["cement_kg"],
                          metadata["reference_temperature_c"],
                          heat[m][:, 0], heat[m][:, 1],
                          modulus[m][:, 0], modulus[m][:, 1], start, end)
        for m, r in recipes.items()
    }
