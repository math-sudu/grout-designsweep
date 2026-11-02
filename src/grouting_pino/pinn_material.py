"""Differentiable measured paste maps for the support PINNs.

Age is in seconds, heat in J/kg dry cement, heat rate in W/kg dry cement,
and modulus in Pa. The heat origin remains the common observation start.
The data buffers are fixed; derivatives with respect to the state are retained.
"""

import numpy as np
import torch
from torch import nn

from grouting_pino.paste_reference import PasteReference


class PasteStateMap(nn.Module):
    """Tensor version of the measured heat integral and paired modulus curve."""

    def __init__(self, reference: PasteReference):
        super().__init__()
        self.mix_id = reference.mix_id
        self.water_cement_ratio = reference.water_cement_ratio
        self.reference_temperature_k = reference.temperature_c + 273.15

        def paired_ages(ages):
            interior = ages[(ages > reference.start_age_h)
                            & (ages < reference.end_age_h)]
            return np.r_[reference.start_age_h, interior, reference.end_age_h]

        heat_age = paired_ages(reference.heat_age_h)
        modulus_age = paired_ages(reference.modulus_age_h)
        rate = np.interp(heat_age, reference.heat_age_h,
                         reference.heat_rate_w_kg_cement)
        buffers = {
            "heat_age_s": heat_age * 3600,
            "heat_nodes": reference.heat_at_age(heat_age),
            "rate_nodes": rate,
            "rate_slopes": np.diff(rate) / np.diff(heat_age * 3600),
            "modulus_age_s": modulus_age * 3600,
            "modulus_nodes": np.interp(modulus_age, reference.modulus_age_h,
                                       reference.modulus_pa),
        }
        for name, values in buffers.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.float64))

    @property
    def max_heat(self):
        return self.heat_nodes[-1]

    def valid_heat(self, heat):
        """Validity per queried state; no extrapolation or heat saturation."""
        return torch.isfinite(heat) & (heat >= 0) & (heat <= self.max_heat)

    def _check_heat(self, heat):
        if not self.valid_heat(heat).all():
            raise ValueError("Heat state is outside the paired observation domain")

    @staticmethod
    def _interval(knots, values):
        return (torch.searchsorted(knots, values.contiguous(), right=True) - 1
                ).clamp(0, len(knots) - 2)

    def _age_and_rate(self, heat):
        i = self._interval(self.heat_nodes, heat)
        delta_heat = heat - self.heat_nodes[i]
        rate0 = self.rate_nodes[i]
        rate = torch.sqrt(rate0.square() + 2 * self.rate_slopes[i] * delta_heat)
        # Stable also on a constant-rate segment: no division by its slope.
        delta_age = 2 * delta_heat / (rate0 + rate)
        return self.heat_age_s[i] + delta_age, rate

    def reference_age_s(self, heat):
        self._check_heat(heat)
        return self._age_and_rate(heat)[0]

    def heat_at_reference_age_s(self, age_s):
        valid = (torch.isfinite(age_s) & (age_s >= self.heat_age_s[0])
                 & (age_s <= self.heat_age_s[-1]))
        if not valid.all():
            raise ValueError("Reference age is outside the paired observation domain")
        i = self._interval(self.heat_age_s, age_s)
        delta_age = age_s - self.heat_age_s[i]
        return (self.heat_nodes[i] + self.rate_nodes[i] * delta_age
                + self.rate_slopes[i] * delta_age.square() / 2)

    def rate(self, heat):
        self._check_heat(heat)
        return self._age_and_rate(heat)[1]

    def modulus(self, heat):
        age = self.reference_age_s(heat)
        i = self._interval(self.modulus_age_s, age)
        fraction = ((age - self.modulus_age_s[i])
                    / (self.modulus_age_s[i + 1] - self.modulus_age_s[i]))
        return (self.modulus_nodes[i]
                + fraction * (self.modulus_nodes[i + 1] - self.modulus_nodes[i]))

    def trial_rate(self, heat):
        """Endpoint continuation for optimizer trials only.

        This keeps the reaction residual evaluable outside the observation
        interval. It does not clip the evolving heat state. Predictions must
        pass valid_heat before using the measured age or modulus maps.
        """
        if not torch.isfinite(heat).all():
            raise ValueError("Optimizer heat state must be finite")
        bounded = heat.clamp(min=0, max=self.max_heat)
        return self._age_and_rate(bounded)[1]
