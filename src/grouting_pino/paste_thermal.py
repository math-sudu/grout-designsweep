"""Nonisothermal continuation of paired paste heat and modulus observations.

The state q is heat since the first supported material age (J/kg cement).
It remains nonzero when a history is restarted. Temperature and q evolve
together in a radial lining/paste finite-volume model; no pre-observation
heat is invented. Integration ends when any paste cell reaches the last
supported material state.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from grouting_pino.paste_reference import PasteReference

R_GAS = 8.314462618


def arrhenius(temperature_c, activation_energy_j_mol, reference_c=20.0):
    return np.exp(activation_energy_j_mol / R_GAS
                  * (1 / (reference_c + 273.15) - 1 / (np.asarray(temperature_c) + 273.15)))


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


def radial_mesh(geometry, paste, n_paste=25, n_ring=49,
                ring_conductivity=1.8, ring_capacity=2.2e6):
    """Adapt ring_chain's shared-interface grid to annular volume/resistance.

    Volumes and heat capacities are per radian and per metre of tunnel.
    The shared interface node contains both material half-cells.
    """
    props, k = paste
    lr = geometry["lining_thickness_m"]
    lg = geometry["grout_thickness_m"]
    if n_paste < 3 or n_ring < 2:
        raise ValueError("The composite grid needs at least 3 paste and 2 ring nodes")
    r = np.r_[np.linspace(0, lr, n_ring), lr+np.linspace(0, lg, n_paste)[1:]]
    r += geometry["lining_inner_diameter_m"] / 2
    interface = n_ring - 1
    edges = np.r_[r[0], (r[1:] + r[:-1]) / 2, r[-1]]
    ring_edge = r[interface]
    vr = (np.minimum(edges[1:], ring_edge)**2
          - np.minimum(edges[:-1], ring_edge)**2) / 2
    vg = (np.maximum(edges[1:], ring_edge)**2
          - np.maximum(edges[:-1], ring_edge)**2) / 2
    capacities = vr * ring_capacity + vg * props["capacity_j_m3k"]
    conductivity = np.where(np.arange(len(r)-1) < interface, ring_conductivity, k)
    conductance = conductivity / np.log(r[1:] / r[:-1])
    return {"radius_m": r, "capacity": capacities, "conductance": conductance,
            "paste_volume": vg[interface:], "ring_volume": vr,
            "interface": interface}


@dataclass
class PasteHistory:
    time_h: np.ndarray
    temperature_c: np.ndarray
    heat_j_kg_cement: np.ndarray
    modulus_pa: np.ndarray
    effective_modulus_pa: np.ndarray
    harmonic_modulus_pa: np.ndarray
    boundary_energy_j: np.ndarray
    mesh: dict
    volume_properties: dict
    status: str
    state_time_h: float = 0.0

    @property
    def construction_time_h(self):
        """Time on the shared construction clock, including elapsed history."""
        return self.state_time_h + self.time_h


def simulate_paste(reference: PasteReference, geometry: dict, time_h,
                   air_temperature_c, soil_temperature_c,
                   properties=PasteThermalProperties(), *,
                   initial_reference_age_h=None, initial_heat_j_kg_cement=None,
                   initial_paste_temperature_c=20.0, initial_ring_temperature_c=20.0,
                   initial_temperature_c=None, state_time_h=0.0,
                   air_h_w_m2k=8.0, soil_h_w_m2k=16.0,
                   n_paste=25, n_ring=49, ring_conductivity=1.8, ring_capacity=2.2e6,
                   heat_release=True, temperature_ceiling_c=40.0,
                   rtol=2e-5, max_step_h=0.1):
    """Continue a specified material state under air/soil reservoir histories.

    h values are the coefficients actually applied at the two surfaces.
    q and temperature fields from a previous run may be supplied unchanged
    for restart, with their construction time in state_time_h. Input time and
    reservoir histories are relative to that state; construction_time_h carries
    the common clock. Reference age only locates a material state.
    With heat_release=False, only the self-heating feedback is
    disabled; maturation still follows the local temperature.
    """
    time = np.asarray(time_h, dtype=float)
    if not np.isfinite(state_time_h) or state_time_h < 0:
        raise ValueError("Construction time of the supplied state must be finite and nonnegative")
    air = np.broadcast_to(air_temperature_c, time.shape)
    soil = np.broadcast_to(soil_temperature_c, time.shape)
    if (time.ndim != 1 or time.size < 2 or time[0] != 0
            or np.any(np.diff(time) <= 0) or not np.all(np.isfinite(time))
            or not np.all(np.isfinite(air)) or not np.all(np.isfinite(soil))):
        raise ValueError("Boundary histories need finite increasing times starting at zero")
    if initial_heat_j_kg_cement is not None and initial_reference_age_h is not None:
        raise ValueError("Supply either reference age or heat state, not both")
    vp = properties.volume_properties(reference.water_cement_ratio)
    mesh = radial_mesh(geometry, (vp, properties.conductivity_w_mk), n_paste, n_ring,
                       ring_conductivity, ring_capacity)
    cap, conduct = mesh["capacity"], mesh["conductance"]
    ix, r, volume = mesh["interface"], mesh["radius_m"], mesh["paste_volume"]
    nt = len(r)
    if initial_heat_j_kg_cement is None:
        age = reference.start_age_h if initial_reference_age_h is None else initial_reference_age_h
        initial_heat_j_kg_cement = reference.heat_at_age(age)
    q0 = np.broadcast_to(initial_heat_j_kg_cement, (n_paste,)).copy()
    reference.age_at_heat(q0)
    if np.max(q0) >= reference.max_heat_j_kg_cement:
        raise ValueError("Initial state has no remaining measured heat interval")
    temp = np.full(nt, float(initial_paste_temperature_c))
    temp[:ix] = initial_ring_temperature_c
    share = volume[0] * vp["capacity_j_m3k"] / cap[ix]
    temp[ix] = share * initial_paste_temperature_c + (1-share) * initial_ring_temperature_c
    if initial_temperature_c is not None:
        temp = np.asarray(initial_temperature_c, dtype=float).copy()
        if temp.shape != (nt,) or not np.all(np.isfinite(temp)):
            raise ValueError("Restart temperature must match the composite grid")
    if temperature_ceiling_c is not None and temp[ix:].max() >= temperature_ceiling_c:
        raise ValueError("Initial paste temperature is outside the selected temperature scope")
    surface = np.array([air_h_w_m2k*r[0], soil_h_w_m2k*r[-1]])
    heat_mass = vp["cement_kg_m3"] * volume

    def rhs(t, y):
        T, q = y[:nt], y[nt:nt+n_paste]
        # BDF trial states may cross the event. Only the rate evaluation is
        # bounded here; the solver terminates at the measured-domain event.
        qs = np.clip(q, 0, reference.max_heat_j_kg_cement)
        rate = reference.rate_at_heat(qs) * arrhenius(
            T[ix:], properties.activation_energy_j_mol, reference.temperature_c)
        flow = conduct * np.diff(T)
        power = np.r_[flow, 0.0] - np.r_[0.0, flow]
        boundary = surface * (np.array([np.interp(t/3600, time, air),
                                        np.interp(t/3600, time, soil)]) - T[[0, -1]])
        power[[0, -1]] += boundary
        if heat_release:
            power[ix:] += heat_mass * rate
        return np.r_[power / cap, rate, np.sum(boundary)]

    def domain_end(t, y):
        return reference.max_heat_j_kg_cement - np.max(y[nt:nt+n_paste])

    domain_end.terminal = True
    domain_end.direction = -1
    events = [domain_end]
    if temperature_ceiling_c is not None:
        def temperature_end(t, y):
            return temperature_ceiling_c - np.max(y[ix:nt])
        temperature_end.terminal = True
        temperature_end.direction = -1
        events.append(temperature_end)
    size = nt + n_paste + 1
    sparse = lil_matrix((size, size), dtype=int)
    for i in range(nt):
        sparse[i, max(0, i-1):min(nt, i+2)] = 1
    for j in range(n_paste):
        i = ix+j
        sparse[i, nt+j] = 1
        sparse[nt+j, i] = sparse[nt+j, nt+j] = 1
    sparse[-1, 0] = sparse[-1, nt-1] = 1
    solution = solve_ivp(rhs, (0, time[-1]*3600), np.r_[temp, q0, 0.0],
                         method="BDF", rtol=rtol,
                         atol=np.r_[np.full(nt, 1e-5), np.full(n_paste, 0.01), 0.1],
                         max_step=max_step_h*3600, dense_output=True,
                         jac_sparsity=sparse.tocsr(), events=events)
    if not solution.success:
        raise RuntimeError(solution.message)
    end = solution.t[-1]/3600
    times = time[time < end]
    times = np.r_[times, end]
    y = solution.sol(times*3600)
    q = y[nt:nt+n_paste].T
    # Event/root rounding only; no saturation or extrapolation past the event.
    q = np.clip(q, 0, reference.max_heat_j_kg_cement)
    modulus = reference.modulus_at_heat(q)
    weights = volume / volume.sum()
    status = "completed"
    if solution.t_events[0].size:
        status = "material_domain_end"
    elif len(events) > 1 and solution.t_events[1].size:
        status = "temperature_scope_end"
    return PasteHistory(times, y[:nt].T, q, modulus, modulus @ weights,
                        1 / ((1/modulus) @ weights), y[-1], mesh, vp,
                        status, float(state_time_h))
