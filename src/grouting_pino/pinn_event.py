"""Continuous event-time readouts of the mixed spatial neural fields."""

from copy import copy
import json
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular, svd
from scipy.sparse.linalg import LinearOperator, cg

from grouting_pino.pinn_mechanics import (
    CENTER_Y, SectionPINN, assemble_section_system, sample_domains, section_target,
)


def clamped_knots(breaks, degree=3, continuity=None):
    breaks = np.asarray(breaks, dtype=float)
    if breaks.ndim != 1 or len(breaks) < 2 or np.any(np.diff(breaks) <= 0):
        raise ValueError("Event-time breaks must increase")
    continuity = degree - 1 if continuity is None else continuity
    if continuity < 0 or continuity >= degree:
        raise ValueError("Spline continuity must lie between zero and degree minus one")
    return np.r_[np.repeat(breaks[0], degree + 1),
                 np.repeat(breaks[1:-1], degree - continuity),
                 np.repeat(breaks[-1], degree + 1)]


def time_quadrature(breaks, order):
    nodes, weights = leggauss(order)
    breaks = np.asarray(breaks)
    times = np.concatenate([a + (nodes + 1) * (b - a) / 2
                            for a, b in zip(breaks[:-1], breaks[1:])])
    weights = np.concatenate([weights * (b - a) / 2
                              for a, b in zip(breaks[:-1], breaks[1:])])
    return times, weights / (breaks[-1] - breaks[0])


class EventPINN:
    """A joint spatial/time field; each event is a frozen spatial readout.

    Event time indexes the material state at loading. The displacement and
    stress coefficients at that time are subsequently held by the history.
    """

    def __init__(self, section, thermal, knots, degree=3, coefficients=None):
        self.section = section
        self.thermal = thermal
        self.knots = np.asarray(knots, dtype=float)
        self.degree = degree
        if section.config["mix_id"] != thermal.config["mix_id"]:
            raise ValueError("Mechanical and thermal materials must match")
        if self.knots[0] < 0 or self.knots[-1] > thermal.time_scale / 3600:
            raise ValueError("Event interval exceeds the thermal trajectory")
        ntime = len(self.knots) - degree - 1
        self.coefficients = (np.zeros((section.nweights, ntime)) if coefficients is None
                             else np.asarray(coefficients))
        if self.coefficients.shape != (section.nweights, ntime):
            raise ValueError("Event coefficients do not match the spatial/time basis")

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        if getattr(self, "_frozen", False):
            raise ValueError("A frozen event family cannot be retrained")
        self._coefficients = value

    def freeze(self):
        self.coefficients.setflags(write=False)
        self._frozen = True
        return self

    def time_basis(self, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if not np.isfinite(times).all():
            raise ValueError("Event times must be finite")
        return BSpline.design_matrix(times, self.knots, self.degree,
                                     extrapolate=False).toarray()

    def event(self, time_h):
        time_h = float(time_h)
        basis = self.time_basis([time_h])[0]
        self.thermal.query("paste", np.linspace(5.8, 6.02, 201), time_h)
        section = copy(self.section)
        section.config = self.section.config | {"time_h": time_h}
        section.weights = self.coefficients @ basis
        section.paste_modulus = lambda r: self.thermal.query(
            "paste", np.clip(r, 5.8, 6.02), time_h)["modulus_pa"]
        return section

    def query(self, domain, xy, time_h, derivatives=False):
        return self.event(time_h).query(domain, xy, derivatives)

    def arrays(self):
        payload = self.section.arrays()
        payload.pop("weights")
        payload.update(event_coefficients=self.coefficients, time_knots=self.knots,
                       time_degree=np.array(self.degree))
        return payload

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path, thermal):
        with np.load(path) as saved:
            return cls.from_arrays(saved, thermal)

    @classmethod
    def from_arrays(cls, saved, thermal):
        config = json.loads(str(saved["config_json"]))
        modulus = lambda r: thermal.query("paste", np.clip(r, 5.8, 6.02),
                                          config["time_h"])["modulus_pa"]
        arrays = dict(saved)
        size = saved["event_coefficients"].shape[0]
        arrays["weights"] = np.zeros(size)
        section = SectionPINN.from_arrays(arrays, modulus)
        return cls(section, thermal, saved["time_knots"].copy(), int(saved["time_degree"]),
                   saved["event_coefficients"].copy())


def whitened_normal_action(x, temporal, constant_rows, varying_rows, change):
    """Apply the joint normal operator after spatial/time whitening.

    A(t) P = U + diag(change(t)) H on the changing paste rows, with
    U.T U = I. All remaining spatial residuals are represented by that I.
    """
    at_time = x @ temporal.T
    fixed = constant_rows @ at_time
    varying = varying_rows @ at_time
    correction = (constant_rows.T @ (change * varying)
                  + varying_rows.T @ (change * fixed + change**2 * varying))
    return x + correction @ temporal


class SpatialEventOperator:
    """Reuse one spatial decomposition for joint or independent event training."""

    def __init__(self, models):
        started = time.perf_counter()
        self.models = list(models)
        section = self.models[0].section
        operator_config = {k: v for k, v in section.config.items() if k != "pressure_mode"}
        for model in self.models[1:]:
            if {k: v for k, v in model.section.config.items() if k != "pressure_mode"} != operator_config:
                raise ValueError("Shared pressure solves require identical spatial operators")
        A, target, self.labels = assemble_section_system(section)
        assembled = time.perf_counter()
        self.norm = np.linalg.norm(A, axis=0)
        A /= self.norm
        print(f"Reference SVD: {A.shape}", flush=True)
        left, singular, vt = svd(A, full_matrices=False, check_finite=False)
        keep = singular > singular[0] * section.config["svd_cutoff"]
        self.preconditioner = vt[keep].T / singular[keep]
        left = left[:, keep]
        self.targets = np.column_stack([target, *[
            section_target(model.section, self.labels) for model in self.models[1:]]])
        self.rhs = left.T @ self.targets
        decomposed = time.perf_counter()
        changing, start = [], 0
        for label, size in self.labels:
            if label.startswith("paste_constitutive_"):
                changing.extend(range(start, start + size))
            start += size
        self.changing = np.asarray(changing)
        columns = np.r_[np.arange(section.slices["paste", "ux"].start,
                                  section.slices["paste", "ux"].stop),
                        np.arange(section.slices["paste", "uy"].start,
                                  section.slices["paste", "uy"].stop)]
        self.constant_rows = left[self.changing].copy()
        self.varying_rows = A[np.ix_(self.changing, columns)] @ self.preconditioner[columns]
        self.matrix = A
        xy = sample_domains(section.config, section.config["seed"] + 11)["paste"]
        self.radius = np.hypot(xy[:, 0], xy[:, 1] - CENTER_Y)
        self.reference_modulus = section.modulus("paste", xy)
        self.timing_s = {"assembly": assembled - started,
                         "spatial_svd": decomposed - assembled,
                         "projection": time.perf_counter() - decomposed}

    def change(self, times):
        modulus = self.models[0].thermal.query(
            "paste", self.radius[:, None], np.asarray(times)[None, :])["modulus_pa"]
        return np.tile(modulus / self.reference_modulus[:, None] - 1, (3, 1))

    def independent(self, time_h, linear_rtol=1e-13, modes=None):
        """Train uncoupled readouts at one exact time with cached spatial rows."""
        started = time.perf_counter()
        change = self.change([time_h])
        fixed, varying = self.constant_rows, self.varying_rows
        diagonal = 1 + (2 * fixed * varying).T @ change[:, 0] + (varying**2).T @ change[:, 0]**2
        rank = len(diagonal)
        operator = LinearOperator((rank, rank), dtype=float,
            matvec=lambda v: whitened_normal_action(v[:, None], np.ones((1, 1)),
                                                    fixed, varying, change)[:, 0])
        inverse = LinearOperator(operator.shape, dtype=float, matvec=lambda v: v / diagonal)
        parts, records = {}, []
        for i, model in enumerate(self.models):
            name = model.section.config.get("pressure_mode", "uniform")
            if modes is not None and name not in modes:
                continue
            iterations = 0

            def progress(_):
                nonlocal iterations
                iterations += 1

            solution, info = cg(operator, self.rhs[:, i], x0=self.rhs[:, i], M=inverse,
                                rtol=linear_rtol, atol=0., callback=progress)
            if info != 0:
                raise RuntimeError(f"Independent event solve did not converge: {info}")
            part = copy(model.section)
            part.config = part.config | {"time_h": float(time_h)}
            part.weights = (self.preconditioner @ solution) / self.norm
            part.paste_modulus = lambda r: model.thermal.query(
                "paste", np.clip(r, 5.8, 6.02), time_h)["modulus_pa"]
            parts[name] = part
            records.append({"pressure_mode": name, "linear_iterations": iterations,
                "relative_normal_residual": float(np.linalg.norm(operator @ solution - self.rhs[:, i])
                                                   / np.linalg.norm(self.rhs[:, i]))})
        return parts, {"time_h": float(time_h), "elapsed_s": time.perf_counter() - started,
                       "right_hand_sides": records}


def train_event_family(model, breaks, quadrature_order=4, linear_rtol=1e-13,
                       additional_models=(), spatial_operator=None):
    """Joint physics least squares using the parameter-varying paste rows.

    One spatial SVD and one temporal QR precondition the coupled solve. The
    readout is trained at all quadrature times together; spatial field labels
    are not part of its objective.
    """
    started = time.perf_counter()
    section = model.section
    models = [model, *additional_models]
    operator_config = {k: v for k, v in section.config.items() if k != "pressure_mode"}
    for other in additional_models:
        if ({k: v for k, v in other.section.config.items() if k != "pressure_mode"} != operator_config
                or not np.array_equal(other.knots, model.knots) or other.degree != model.degree):
            raise ValueError("Shared pressure solves require identical spatial and time operators")
    times, qweights = time_quadrature(breaks, quadrature_order)
    design = model.time_basis(times)
    temporal, rtime = np.linalg.qr(np.sqrt(qweights)[:, None] * design,
                                   mode="reduced")
    spatial = SpatialEventOperator(models) if spatial_operator is None else spatial_operator
    if any(a is not b for a, b in zip(spatial.models, models, strict=True)):
        raise ValueError("The spatial operator belongs to different event models")
    A, labels, norm = spatial.matrix, spatial.labels, spatial.norm
    preconditioner, whitened_targets = spatial.preconditioner, spatial.rhs
    changing = spatial.changing
    constant_rows, varying_rows = spatial.constant_rows, spatial.varying_rows
    change = spatial.change(times)
    projected = time.perf_counter()

    shape = (preconditioner.shape[1], temporal.shape[1])
    shape_size = int(np.prod(shape))
    # Diagonal of the exact whitened normal operator.
    diagonal = (1 + ((2 * constant_rows * varying_rows).T @ change
                      + (varying_rows**2).T @ (change**2)) @ (temporal**2))
    operator = LinearOperator((shape_size, shape_size), dtype=float,
        matvec=lambda flat: whitened_normal_action(flat.reshape(shape), temporal,
                           constant_rows, varying_rows, change).ravel())
    inverse_diagonal = LinearOperator(operator.shape, dtype=float,
                                      matvec=lambda flat: flat / diagonal.ravel())
    records = []
    for index, current in enumerate(models):
        rhs = whitened_targets[:, index, None] * (np.sqrt(qweights) @ temporal)[None, :]
        iterations = 0

        def progress(_):
            nonlocal iterations
            iterations += 1
            if iterations % 50 == 0:
                print(f"Joint readout {index + 1}, iteration {iterations}", flush=True)

        solution, info = cg(operator, rhs.ravel(), x0=rhs.ravel(), M=inverse_diagonal,
                            rtol=linear_rtol, atol=0., callback=progress)
        if info != 0:
            raise RuntimeError(f"Joint readout solve did not converge: {info}")
        whitened = solution.reshape(shape)
        normal_residual = np.linalg.norm(operator @ solution - rhs.ravel()) / np.linalg.norm(rhs)
        normalized_coefficients = solve_triangular(rtime, (preconditioner @ whitened).T,
                                                   lower=False).T
        current.coefficients = normalized_coefficients / norm[:, None]
        at_time = whitened @ temporal.T
        residual = A @ (preconditioner @ at_time) - spatial.targets[:, index, None] * np.sqrt(qweights)
        residual[changing] += change * (varying_rows @ at_time)
        losses, start = {}, 0
        for label, size in labels:
            losses[label] = float(np.sum(residual[start:start + size]**2))
            start += size
        records.append({"pressure_mode": current.section.config.get("pressure_mode", "uniform"),
                        "linear_iterations": iterations,
                        "relative_normal_residual": float(normal_residual), "losses": losses})
    solved = time.perf_counter()
    return {"rank": shape[0], "spatial_weights": section.nweights,
            "time_features": shape[1], "joint_weights": int(model.coefficients.size),
            "spatial_rows": len(A), "time_h": times.tolist(),
            "time_weights": qweights.tolist(), "time_knots": model.knots.tolist(),
            "degree": model.degree, "quadrature_order": quadrature_order,
            "reference_time_h": section.config["time_h"],
            "linear_rtol": linear_rtol, **records[0], "right_hand_sides": records,
            "elapsed_s": time.perf_counter() - started,
            "timing_s": {**spatial.timing_s, "joint_solve": solved - projected},
            "changing_operator_bytes": int(constant_rows.nbytes + varying_rows.nbytes + change.nbytes),
            "dense_tensor_bytes": int(len(times) * len(A) * model.coefficients.size * 8)}
