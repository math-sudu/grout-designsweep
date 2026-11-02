"""Mixed physics-informed neural fields with a linear trainable readout.

Fixed tanh radial neurons and periodic angular features give a branch-free
neural representation. Output weights minimize first-order equilibrium,
constitutive, boundary and interface residuals. No displacement/stress labels
or other PDE solver enter training. Even and odd pressure branches use the
reflection-invariant section; fields are returned on either half in SI units.
"""

from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from scipy.linalg import lstsq
from scipy.stats import qmc


DOMAINS = ("soil", "paste", "lining")
COMPONENTS = ("ux", "uy", "sxx", "syy", "sxy")
RADIUS = {"soil": (6.02, 45.71), "paste": (5.80, 6.02), "lining": (5.25, 5.80)}
CENTER_Y = -25.52
LENGTH = 6.02
GROUND_E = 85e6
PRESSURE = 1e5
DISPLACEMENT = PRESSURE * LENGTH / GROUND_E
NU = 0.30


class FieldMatrixCache:
    """Bounded feature reuse across event times and pressure readouts."""

    def __init__(self, max_bytes=128 * 1024**2):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.peak_bytes = 0
        self.entries = OrderedDict()

    def bind(self, section):
        section._field_cache = self
        section._basis_keys = {domain: (basis.modes, basis.reflection,
            basis.slopes.tobytes(), basis.biases.tobytes(), basis.transform.tobytes())
            for domain, basis in section.bases.items()}

    def get(self, section, domain, xy):
        xy = np.asarray(xy, dtype=float)
        key = (domain, section._basis_keys[domain], xy.shape, xy.tobytes())
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][0]
        value = {comp: section.bases[domain].evaluate(xy, comp) for comp in COMPONENTS}
        size = sum(array.nbytes for matrices in value.values() for array in matrices)
        if size <= self.max_bytes:
            while self.bytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = value, size
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value


@dataclass
class NeuralBasis:
    domain: str
    slopes: np.ndarray
    biases: np.ndarray
    modes: int
    transform: np.ndarray
    reflection: str = "even"

    @classmethod
    def create(cls, domain, radial_neurons, modes, rng, reflection="even"):
        slopes = rng.uniform(0.6, 3.0, radial_neurons)
        biases = rng.uniform(-2.5, 2.5, radial_neurons)
        z = np.linspace(-1, 1, 401)
        h = np.c_[np.ones(len(z)), np.tanh(z[:,None]*slopes+biases)]
        _, singular, vt = np.linalg.svd(h, full_matrices=False)
        keep = singular > singular[0]*1e-9
        transform = vt[keep].T / singular[keep] * np.sqrt(len(z))
        if reflection not in ("even", "odd"):
            raise ValueError("Pressure reflection must be even or odd")
        return cls(domain, slopes, biases, modes, transform, reflection)

    def evaluate(self, xy, component):
        """Feature values and exact derivatives with respect to x and y."""
        x, y = np.asarray(xy).T
        dy = y - CENTER_Y
        r = np.hypot(x, dy)
        # theta measured from the vertical: x=r*sin(theta), dy=r*cos(theta).
        theta = np.arctan2(x, dy)
        st, ct = x / r, dy / r
        left, right = RADIUS[self.domain]
        if self.domain == "soil":
            z = 2 * np.log(r / left) / np.log(right / left) - 1
            dzdr = 2 / (r * np.log(right / left))
        else:
            z = 2 * (r - left) / (right - left) - 1
            dzdr = np.full_like(r, 2 / (right - left))
        h = np.tanh(z[:, None] * self.slopes + self.biases)
        hr = (1 - h * h) * self.slopes * dzdr[:, None]
        h = np.c_[np.ones(len(r)), h]
        hr = np.c_[np.zeros(len(r)), hr]
        h, hr = h @ self.transform, hr @ self.transform
        odd = (component in ("ux", "sxy")) != (self.reflection == "odd")
        n = np.arange(1, self.modes + 1) if odd else np.arange(self.modes + 1)
        a = np.sin(theta[:, None] * n) if odd else np.cos(theta[:, None] * n)
        at = n * np.cos(theta[:, None] * n) if odd else -n * np.sin(theta[:, None] * n)
        value = (h[:, :, None] * a[:, None, :]).reshape(len(r), -1)
        dr = (hr[:, :, None] * a[:, None, :]).reshape(len(r), -1)
        dt = (h[:, :, None] * at[:, None, :]).reshape(len(r), -1)
        dx = dr * st[:, None] + dt * (ct / r)[:, None]
        dyv = dr * ct[:, None] - dt * (st / r)[:, None]
        if self.domain == "soil" and component in ("ux", "uy"):
            factor = (y + 60) / 60
            fx, fy = np.zeros_like(x), np.full_like(y, 1 / 60)
            if component == "ux":
                fx = -2 * x / 900 * factor
                factor = factor * (1 - x * x / 900)
                fy = fy * (1 - x * x / 900)
            dx = dx * factor[:, None] + value * fx[:, None]
            dyv = dyv * factor[:, None] + value * fy[:, None]
            value = value * factor[:, None]
        return value, dx, dyv


class SectionPINN:
    def __init__(self, config, paste_modulus):
        self.config = config
        self.paste_modulus = paste_modulus
        rng = np.random.default_rng(config["seed"])
        self.bases = {d: NeuralBasis.create(d, config["radial_neurons"][d],
                                           config["angular_modes"], rng,
                                           config.get("reflection", "even")) for d in DOMAINS}
        self.slices = {}
        cursor = 0
        for d in DOMAINS:
            for comp in COMPONENTS:
                odd = (comp in ("ux", "sxy")) != (config.get("reflection", "even") == "odd")
                n = self.bases[d].transform.shape[1] * (config["angular_modes"] + (not odd))
                self.slices[d, comp] = slice(cursor, cursor + n)
                cursor += n
        self.nweights = cursor
        self.weights = np.zeros(cursor)

    def modulus(self, domain, xy):
        if domain == "soil":
            return np.full(len(xy), GROUND_E)
        if domain == "lining":
            return np.full(len(xy), 20e9)
        r = np.hypot(xy[:, 0], xy[:, 1] - CENTER_Y)
        return self.paste_modulus(r)

    def field_matrix(self, domain, xy):
        if hasattr(self, "_field_cache"):
            return self._field_cache.get(self, domain, xy)
        return {comp: self.bases[domain].evaluate(xy, comp) for comp in COMPONENTS}

    def query(self, domain, xy, derivatives=False):
        xy = np.asarray(xy,dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Section coordinates must be a finite N by 2 array")
        r = np.hypot(xy[:,0],xy[:,1]-CENTER_Y)
        left,right = RADIUS[domain]
        valid = (r >= left-1e-10)
        if domain == "soil":
            valid &= (np.abs(xy[:,0]) <= 30+1e-10) & (xy[:,1] >= -60-1e-10) & (xy[:,1] <= 1e-10)
        else:
            valid &= r <= right+1e-10
        if not valid.all():
            raise ValueError("Query outside the material subdomain")
        basis = self.field_matrix(domain, xy)
        result = {}
        for comp, mats in basis.items():
            scale = DISPLACEMENT if comp.startswith("u") else PRESSURE
            w = self.weights[self.slices[domain, comp]]
            result[comp] = mats[0] @ w * scale
            if derivatives:
                result[comp + "_x"] = mats[1] @ w * scale
                result[comp + "_y"] = mats[2] @ w * scale
        if derivatives:
            E = self.modulus(domain, np.asarray(xy))
            result["szz"] = E * NU / ((1 + NU) * (1 - 2 * NU)) * (
                result["ux_x"] + result["uy_y"])
        return result

    def arrays(self):
        import json
        payload = {"config_json": np.array(json.dumps(self.config)), "weights": self.weights}
        for domain, basis in self.bases.items():
            payload[domain + "_slopes"] = basis.slopes
            payload[domain + "_biases"] = basis.biases
            payload[domain + "_transform"] = basis.transform
        return payload

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path, paste_modulus):
        with np.load(path) as saved:
            return cls.from_arrays(saved, paste_modulus)

    @classmethod
    def from_arrays(cls, saved, paste_modulus):
        import json
        model = cls(json.loads(str(saved["config_json"])), paste_modulus)
        model.weights = saved["weights"].copy()
        for d in DOMAINS:
            model.bases[d].slopes = saved[d + "_slopes"].copy()
            model.bases[d].biases = saved[d + "_biases"].copy()
            model.bases[d].transform = saved[d + "_transform"].copy()
        return model


def annulus_points(radius, theta):
    radius, theta = np.broadcast_arrays(radius, theta)
    return np.column_stack([radius.ravel() * np.sin(theta.ravel()),
                            CENTER_Y + radius.ravel() * np.cos(theta.ravel())])


def sample_domains(config, seed, full_domain=False):
    sampler = qmc.Halton(2, scramble=True, seed=seed)
    result = {}
    for domain in DOMAINS:
        n = config["domain_points"][domain]
        uv = sampler.random(n)
        lo, hi = RADIUS[domain]
        if domain == "soil":
            # Polar sampling resolves both the loaded interface and far soil.
            angle = np.pi * uv[:, 1]
            outer = np.minimum(30 / np.maximum(np.sin(angle), 1e-15),
                               np.where(np.cos(angle) > 0,
                                        -CENTER_Y / np.maximum(np.cos(angle), 1e-15),
                                        (60 + CENTER_Y) / np.maximum(-np.cos(angle), 1e-15)))
            radius = lo * (outer / lo) ** uv[:, 0]
        else:
            angle = np.pi * uv[:, 1]
            radius = lo + (hi - lo) * uv[:, 0]
        result[domain] = annulus_points(radius, angle)
        if full_domain:
            result[domain][::2, 0] *= -1
    return result


def pressure_profile(config, theta):
    """Dimensionless pressure mode, with angle measured rightward from crown."""
    theta = (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi
    mode = config.get("pressure_mode", "uniform")
    if mode == "uniform":
        return np.ones_like(theta)
    half = .5 * (np.abs(theta) < np.deg2rad(config["sector_angle_deg"]))
    if mode == "sector_even":
        return half
    if mode == "sector_odd":
        return half * np.sign(theta)
    raise ValueError(f"Unknown pressure mode: {mode}")


def boundary_angles(config):
    """Midpoint quadrature on each continuous piece of the pressure boundary."""
    n = config["boundary_points"]
    cuts = [0., np.pi]
    if "sector_angle_deg" in config:
        angle = np.deg2rad(config["sector_angle_deg"])
        if not 0 < angle < np.pi:
            raise ValueError("Sector angle must lie between zero and 180 degrees")
        cuts.insert(1, angle)
    angles, weights = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        count = max(1, round(n * (b - a) / np.pi))
        angles.append(a + (np.arange(count) + .5) * (b - a) / count)
        weights.append(np.full(count, (b - a) / (np.pi * count)))
    return np.concatenate(angles), np.concatenate(weights)


def section_target(model, labels):
    """Pressure-dependent right-hand side on a shared spatial residual layout."""
    angle, weight = boundary_angles(model.config)
    pressure = pressure_profile(model.config, angle)
    values = []
    for label, size in labels:
        if label in ("paste_soil_tx", "paste_soil_ty"):
            normal = np.sin(angle) if label.endswith("tx") else np.cos(angle)
            values.append(-pressure * normal * model.config["interface_weight"] * np.sqrt(weight))
        else:
            values.append(np.zeros(size))
    return np.concatenate(values)


def assemble_section_system(model):
    """Assemble the weighted first-order mixed physics residuals."""
    c = model.config
    points = sample_domains(c, c["seed"] + 11)
    rows, targets, labels = [], [], []

    def add(label, terms, target=0, weight=1, quadrature=None):
        n = len(terms[0][2])
        row = np.zeros((n, model.nweights))
        for domain, comp, value in terms:
            row[:, model.slices[domain, comp]] += value
        if "equilibrium" in label or "constitutive" in label:
            weight *= c["domain_weight"][terms[0][0]]
        scale = np.full(n, weight / np.sqrt(n)) if quadrature is None else weight * np.sqrt(quadrature)
        rows.append(row * scale[:, None])
        targets.append(np.broadcast_to(target, (n,)) * scale)
        labels.append((label, n))

    for domain, xy in points.items():
        b = model.field_matrix(domain, xy)
        E = model.modulus(domain, xy) / GROUND_E
        mu = E / (2 * (1 + NU))
        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        add(domain + "_equilibrium_x", [(domain, "sxx", LENGTH*b["sxx"][1]),
                                         (domain, "sxy", LENGTH*b["sxy"][2])])
        add(domain + "_equilibrium_y", [(domain, "sxy", LENGTH*b["sxy"][1]),
                                         (domain, "syy", LENGTH*b["syy"][2])])
        add(domain + "_constitutive_xx", [(domain, "sxx", b["sxx"][0]),
                (domain, "ux", -LENGTH*(lam+2*mu)[:,None]*b["ux"][1]),
                (domain, "uy", -LENGTH*lam[:,None]*b["uy"][2])])
        add(domain + "_constitutive_yy", [(domain, "syy", b["syy"][0]),
                (domain, "uy", -LENGTH*(lam+2*mu)[:,None]*b["uy"][2]),
                (domain, "ux", -LENGTH*lam[:,None]*b["ux"][1])])
        add(domain + "_constitutive_xy", [(domain, "sxy", b["sxy"][0]),
                (domain, "ux", -LENGTH*mu[:,None]*b["ux"][2]),
                (domain, "uy", -LENGTH*mu[:,None]*b["uy"][1])])
    n = c["boundary_points"]
    angle, angle_weights = boundary_angles(c)
    nx, ny = np.sin(angle), np.cos(angle)
    for radius, inner, outer in [(5.8, "lining", "paste"), (6.02, "paste", "soil")]:
        xy = annulus_points(radius, angle)
        bi, bo = model.field_matrix(inner, xy), model.field_matrix(outer, xy)
        for comp in ("ux", "uy"):
            add(inner+"_"+outer+"_"+comp, [(inner, comp, bi[comp][0]),
                                           (outer, comp, -bo[comp][0])], weight=c["interface_weight"],
                quadrature=angle_weights)
        for name, ca, cb in [("tx", "sxx", "sxy"), ("ty", "sxy", "syy")]:
            target = -pressure_profile(c, angle) * (nx if name == "tx" else ny) if inner == "paste" else 0
            add(inner+"_"+outer+"_"+name,
                [(inner, ca, nx[:,None]*bi[ca][0]), (inner, cb, ny[:,None]*bi[cb][0]),
                 (outer, ca, -nx[:,None]*bo[ca][0]), (outer, cb, -ny[:,None]*bo[cb][0])],
                target, c["interface_weight"], angle_weights)
    b = model.field_matrix("lining", annulus_points(5.25, angle))
    add("intrados_tx", [("lining","sxx",nx[:,None]*b["sxx"][0]),
                         ("lining","sxy",ny[:,None]*b["sxy"][0])], weight=c["boundary_weight"], quadrature=angle_weights)
    add("intrados_ty", [("lining","sxy",nx[:,None]*b["sxy"][0]),
                         ("lining","syy",ny[:,None]*b["syy"][0])], weight=c["boundary_weight"], quadrature=angle_weights)
    x = 30 * (np.arange(n) + 0.5) / n
    top = model.field_matrix("soil", np.c_[x, np.zeros(n)])
    side = model.field_matrix("soil", np.c_[np.full(n,30), -2*x])
    for comp in ("sxy", "syy"):
        add("surface_"+comp, [("soil",comp,top[comp][0])], weight=c["boundary_weight"])
    add("side_sxy", [("soil","sxy",side["sxy"][0])], weight=c["boundary_weight"])
    A, b = np.concatenate(rows), np.concatenate(targets)
    return A, b, labels


def train_section(model):
    """Linear least-squares training of all mixed output layers together."""
    c = model.config
    A, b, labels = assemble_section_system(model)
    # Column normalization is a reparameterization, not an altered physics loss.
    norm = np.linalg.norm(A, axis=0)
    A /= norm
    print(f"Training mixed readout: {A.shape[0]} residuals, {A.shape[1]} weights", flush=True)
    weights, _, rank, singular = lstsq(A, b, cond=c["svd_cutoff"], lapack_driver="gelsd",
                                      overwrite_a=False, check_finite=False)
    model.weights = weights / norm
    residual = A @ weights - b
    stats = {"rank": int(rank), "weights": model.nweights, "rows": len(b), "losses": {}}
    start = 0
    for label, size in labels:
        stats["losses"][label] = float(np.sum(residual[start:start+size]**2))
        start += size
    return stats
