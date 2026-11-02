"""Explicit pressure-withdrawal events with frozen mixed fields and material clock."""

from copy import deepcopy
import json

import numpy as np
import torch

from grouting_pino.pinn_mechanics import SectionPINN, PRESSURE, CENTER_Y
from grouting_pino.pinn_thermal import ThermalPINN
from grouting_pino.pinn_pressure import PressureEventPINN, pressure_coefficients


LOAD_PATTERN = {
    "operation": "withdraw outward support pressure",
    "interface": "soil-paste", "radius_m": 6.02,
    "center_m": [0.0, CENTER_Y], "direction": "minus radial unit vector",
    "jump": "(sigma_paste - sigma_soil) @ e_r = -p_star * e_r",
    "reference_pressure_pa": PRESSURE,
    "geometry": {"width_m": 60.0, "height_m": 60.0,
                 "lining_inner_radius_m": 5.25, "lining_outer_radius_m": 5.80},
}


def pressure_load_pattern(angle_deg):
    return LOAD_PATTERN | {"jump": "(sigma_paste - sigma_soil) @ e_r = -p_star * (a_uniform + a_sector * chi_A) * e_r",
                           "sector_angle_deg": angle_deg,
                           "sector": "rightward from the crown", "load_fraction": "circumferential mean / p_star"}


class PINNHistory:
    """An equilibrated pressure history, retaining each event's two fields.

    The coefficients of displacement and stress are stored independently;
    no current-stiffness reconstruction of old stress is performed. The frozen
    thermal trajectory supplies the continued state over its trained interval.
    """

    def __init__(self, thermal):
        self.thermal = deepcopy(thermal).cpu().eval()
        self.time_h = 0.0
        self.load_fraction = 0.0
        self.load_coefficients = {"uniform": 0., "sector": 0.}
        self.events = []
        self.family = None
        self.mode_events = []

    def advance(self, time_h):
        if not np.isfinite(time_h) or time_h < self.time_h:
            raise ValueError("History time must not move backwards")
        # Checking the full radial profile also checks the material domain.
        self.thermal.query("paste", np.linspace(5.8,6.02,201),time_h)
        self.time_h = float(time_h)

    def apply(self, section, total_load_fraction):
        if self.family is not None:
            raise ValueError("Use apply_modes after binding a pressure family")
        if section.config.get("pressure_mode", "uniform") != "uniform":
            raise ValueError("Use a pressure family for nonuniform loads")
        if not np.isfinite(total_load_fraction):
            raise ValueError("Load fraction must be finite")
        if section.config["mix_id"] != self.thermal.config["mix_id"]:
            raise ValueError("Event and history material must match")
        if section.config["time_h"] != self.time_h:
            raise ValueError("Event field must use the current material time")
        delta = float(total_load_fraction - self.load_fraction)
        if delta:
            frozen = deepcopy(section)
            event_time = self.time_h
            frozen.paste_modulus = lambda r: self.thermal.query(
                "paste",np.clip(r,5.8,6.02),event_time)["modulus_pa"]
            self.events.append((delta, frozen))
        self.load_fraction = float(total_load_fraction)
        self.load_coefficients["uniform"] = self.load_fraction

    def bind_family(self, family):
        """Freeze one conditional family for all subsequent events in this history."""
        if self.family is not None:
            raise ValueError("This history already has a pressure family")
        self.compatible_family(family)
        self.family = family.freeze()

    def compatible_family(self, family):
        """Validate retained state without replacing the supplied query provider."""
        saved_thermal = self.thermal.state_dict()
        supplied_thermal = family.thermal.state_dict()
        if (family.thermal.config != self.thermal.config
                or saved_thermal.keys() != supplied_thermal.keys()
                or any(not torch.equal(value.cpu(), saved_thermal[key].cpu())
                       for key, value in supplied_thermal.items())):
            raise ValueError("Pressure family must use the history's thermal trajectory")
        if self.family is not None and self.family is not family:
            saved, supplied = self.family.arrays(), family.arrays()
            if (saved.keys() != supplied.keys()
                    or any(not np.array_equal(value, supplied[key])
                           for key, value in saved.items())):
                raise ValueError("Continuation must use the pressure family retained in the history")
        return family

    def apply_modes(self, total_coefficients):
        """Apply cumulative uniform/sector coefficients relative to the saved load."""
        if self.family is None:
            raise ValueError("Bind a pressure family before applying its modes")
        totals = pressure_coefficients(total_coefficients)
        self.family.event(self.time_h, totals)
        delta = {key: totals[key] - self.load_coefficients[key] for key in totals}
        if any(delta.values()):
            self.mode_events.append((self.time_h, delta))
        self.load_coefficients = totals
        self.load_fraction = self.family.mean_pressure_fraction(totals)

    def query(self, domain, xy):
        result = {key: np.zeros(len(xy)) for key in ("ux","uy","sxx","syy","sxy","szz")}
        for amplitude, section in self.events:
            fields = section.query(domain,xy,derivatives=True)
            for key in result:
                result[key] += amplitude * fields[key]
        for event_time, coefficients in self.mode_events:
            fields = self.family.event(event_time, coefficients).query(domain, xy, derivatives=True)
            for key in result:
                result[key] += fields[key]
        return result

    def material_state(self):
        result = {"time_h":self.time_h,
                  "observation_origin_age_s":float(self.thermal.material.heat_age_s[0])}
        for domain,(left,right) in self.thermal.radii.items():
            radius=np.linspace(left,right,201)
            result[domain+"_radius_m"]=radius
            result.update({domain+"_"+key:value for key,value in
                           self.thermal.query(domain,radius,self.time_h).items()})
        return result

    def save(self, path):
        metadata={"load_pattern":LOAD_PATTERN,"time_h":self.time_h,
                  "load_fraction":self.load_fraction,
                  "load_coefficients": self.load_coefficients,
                  "mode_events": self.mode_events,
                  "has_pressure_family": self.family is not None,
                  "event_amplitudes":[amplitude for amplitude,_ in self.events],
                  "thermal_config":self.thermal.config,
                  "initial_field_representation":"sum of frozen mixed event readouts",
                  "thermal_continuation":"frozen learned trajectory over the trained time interval"}
        if self.family is not None:
            metadata["load_pattern"] = pressure_load_pattern(self.family.angle_deg)
        payload={"metadata_json":np.array(json.dumps(metadata))}
        for key,value in self.thermal.state_dict().items():
            payload["thermal_"+key]=value.cpu().numpy()
        for index,(_,section) in enumerate(self.events):
            payload.update({f"event_{index}_"+key:value for key,value in section.arrays().items()})
        if self.family is not None:
            payload.update({"family_" + key: value for key, value in self.family.arrays().items()})
        payload.update({"state_"+key:np.asarray(value) for key,value in self.material_state().items()})
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path,data_dir):
        with np.load(path) as saved:
            metadata=json.loads(str(saved["metadata_json"]))
            expected_pattern = (pressure_load_pattern(metadata["load_pattern"]["sector_angle_deg"])
                                if metadata.get("has_pressure_family", False) else LOAD_PATTERN)
            if metadata["load_pattern"] != expected_pattern:
                raise ValueError("Checkpoint load pattern differs from this section")
            thermal=ThermalPINN(metadata["thermal_config"],data_dir).double()
            state={key[len("thermal_"):]:torch.tensor(saved[key]) for key in saved.files
                   if key.startswith("thermal_")}
            thermal.load_state_dict(state)
            result=cls(thermal)
            result.time_h=metadata["time_h"]
            result.load_fraction=metadata["load_fraction"]
            result.load_coefficients = metadata.get("load_coefficients", {"uniform": result.load_fraction, "sector": 0.})
            for index,amplitude in enumerate(metadata["event_amplitudes"]):
                prefix=f"event_{index}_"
                arrays={key[len(prefix):]:saved[key] for key in saved.files if key.startswith(prefix)}
                event_time=json.loads(str(arrays["config_json"]))["time_h"]
                def modulus(r,time=event_time):
                    return result.thermal.query("paste",np.clip(r,5.8,6.02),time)["modulus_pa"]
                result.events.append((amplitude,SectionPINN.from_arrays(arrays,modulus)))
            if metadata.get("has_pressure_family", False):
                arrays = {key[len("family_"):]: saved[key] for key in saved.files if key.startswith("family_")}
                result.family = PressureEventPINN.from_arrays(arrays, result.thermal).freeze()
                if result.family.angle_deg != metadata["load_pattern"]["sector_angle_deg"]:
                    raise ValueError("Checkpoint pressure sector differs from its family")
                result.mode_events = [(t, coefficients) for t, coefficients in metadata["mode_events"]]
            # The stored nonzero state is required to agree with its frozen
            # field representation, preventing a reset of the material clock.
            for key,value in result.material_state().items():
                if not np.allclose(value,saved["state_"+key],rtol=2e-12,atol=1e-9):
                    raise ValueError("Checkpoint material state is inconsistent")
        return result
