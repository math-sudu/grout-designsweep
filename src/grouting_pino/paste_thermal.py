"""Constituent thermal properties for the measured paste PINN model."""

from dataclasses import dataclass

R_GAS = 8.314462618


@dataclass(frozen=True)
class PasteThermalProperties:
    cement_density_kg_m3: float = 3150.0
    water_density_kg_m3: float = 998.2
    cement_cp_j_kgk: float = 750.0
    water_cp_j_kgk: float = 4180.0
    paste_cp_factor: float = 1.0
    conductivity_w_mk: float = 1.0
    activation_energy_j_mol: float = 40000.0

    def volume_properties(self, water_cement_ratio):
        """Absolute-volume closure for the de-aired two-constituent paste."""
        wc = water_cement_ratio
        cement_mass = 1 / (1 / self.cement_density_kg_m3 + wc / self.water_density_kg_m3)
        rho = (1 + wc) * cement_mass
        capacity = cement_mass * (self.cement_cp_j_kgk + wc * self.water_cp_j_kgk)
        return {"cement_kg_m3": cement_mass, "density_kg_m3": rho,
                "capacity_j_m3k": capacity * self.paste_cp_factor}
