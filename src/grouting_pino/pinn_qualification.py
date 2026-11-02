"""Demand boundaries obtained from the retained uniform event fields."""

import numpy as np
from scipy.interpolate import BSpline, PPoly

from grouting_pino.pinn_mechanics import DISPLACEMENT, PRESSURE


def surface_time_spline(family, x, *, spatial_derivative=False):
    """Project the trained coefficients to surface response in micrometres/kPa."""
    x = np.asarray(x, dtype=float)
    section = family.section
    basis = section.bases["soil"].evaluate(np.c_[x, np.zeros_like(x)], "uy")
    coefficients = -(basis[int(spatial_derivative)]
                     @ family.coefficients[section.slices["soil", "uy"]])
    coefficients *= DISPLACEMENT * 1e9 / PRESSURE
    return BSpline(family.knots, coefficients.T, family.degree, extrapolate=False)


def spline_polynomial(spline):
    """Express each nonempty time interval as a polynomial, including C0 knots."""
    size = len(spline.t) - spline.k - 1
    basis = np.array([PPoly.from_spline((spline.t, row, spline.k)).c
                      for row in np.eye(size)])
    nonempty = np.diff(spline.t) > 0
    coefficients = np.einsum("bki,b...->ki...", basis[:, :, nonempty], spline.c[:size])
    return PPoly(coefficients, np.unique(spline.t), extrapolate=False)


def polynomial_range(polynomial):
    """Exact extrema of linear, quadratic or cubic time pieces at each point."""
    c = polynomial.c
    degree = len(c) - 1
    if degree > 3:
        raise ValueError("The event-time range requires degree at most three")
    width = np.diff(polynomial.x).reshape((-1,) + (1,) * (c.ndim - 2))
    offsets = [np.zeros_like(c[0]), np.broadcast_to(width, c[0].shape)]
    if degree == 2:
        offsets.append(np.divide(-c[1], 2*c[0], out=np.full_like(c[0], np.nan),
                                 where=c[0] != 0))
    elif degree == 3:
        a, b, d = 3*c[0], 2*c[1], c[2]
        discriminant = b*b - 4*a*d
        root = np.sqrt(np.maximum(discriminant, 0))
        for sign in (-1, 1):
            offsets.append(np.divide(-b + sign*root, 2*a,
                out=np.full_like(a, np.nan), where=(a != 0) & (discriminant >= 0)))
        offsets.append(np.divide(-d, b, out=np.full_like(b, np.nan),
                                 where=(a == 0) & (b != 0)))
    values = []
    for offset in offsets:
        value = np.zeros_like(c[0])
        for coefficient in c:
            value = value * offset + coefficient
        values.append(np.where((offset >= 0) & (offset <= width), value, np.nan))
    values = np.array(values)
    return np.nanmin(values, axis=(0, 1)), np.nanmax(values, axis=(0, 1))


def latest_required_increments(minimum_fractions):
    """Load only the cumulative requirement at each deadline; finish at one."""
    minima = np.asarray(minimum_fractions, dtype=float)
    if (minima.ndim != 1 or not len(minima) or not np.isfinite(minima).all()
            or np.any((minima < 0) | (minima > 1))):
        raise ValueError("Cumulative requirements must be finite fractions in [0, 1]")
    cumulative = np.maximum.accumulate(minima)
    cumulative[-1] = 1.
    return np.diff(np.r_[0., cumulative])


def polynomial_sublevel_intervals(polynomial, level):
    """Keep every feasible interval and isolated crossing of a scalar readout."""
    if polynomial.c.ndim != 2:
        raise ValueError("The inverse query requires a scalar response")
    start, end = polynomial.x[[0, -1]]
    roots = polynomial.solve(level, discontinuity=False, extrapolate=False)
    roots = roots[np.isfinite(roots) & (roots >= start) & (roots <= end)]
    nodes = np.unique(np.r_[polynomial.x, roots])
    pieces = [[float(a), float(b)] for a, b in zip(nodes[:-1], nodes[1:], strict=True)
              if polynomial((a+b)/2) <= level]
    pieces += [[float(t), float(t)] for t in roots]
    for t in (start, end):
        if polynomial(t) <= level:
            pieces.append([float(t), float(t)])
    merged = []
    for a, b in sorted(pieces):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def clip_demand_polygon(polygon, normal, bound):
    """Intersect a convex demand polygon with one closed displacement half-plane."""
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(polygon) == 0:
        return polygon
    result = []
    previous = polygon[-1]
    previous_gap = bound - np.dot(normal, previous)
    for current in polygon:
        gap = bound - np.dot(normal, current)
        if (gap >= 0) != (previous_gap >= 0):
            result.append(previous + previous_gap/(previous_gap-gap)*(current-previous))
        if gap >= 0:
            result.append(current)
        previous, previous_gap = current, gap
    return np.array(result).reshape(-1, 2)


def milestone_demand_polygon(fields, lower, upper):
    """All spatial/stage constraints for shares (a, b-a, 1-b), 0 <= a <= b <= 1."""
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 2 or fields.shape[0] != 3 or not np.isfinite(fields).all():
        raise ValueError("Three finite surface event fields are required")
    polygon = np.array([[0., 0.], [1., 1.], [0., 1.]])
    if not lower <= 0 <= upper:
        return np.empty((0, 2))
    first, second, final = fields
    normals = [np.c_[first, np.zeros_like(first)],
               np.c_[first-second, second], np.c_[first-second, second-final]]
    offsets = [np.zeros_like(first), np.zeros_like(first), final]
    for rows, values in zip(normals, offsets, strict=True):
        for normal, offset in zip(rows, values, strict=True):
            polygon = clip_demand_polygon(polygon, normal, upper-offset)
            polygon = clip_demand_polygon(polygon, -normal, offset-lower)
            if not len(polygon):
                return polygon
    return polygon
