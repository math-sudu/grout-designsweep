"""Load histories for the elastic limit of stress-free solidification.

For a fixed mesh, small strains and homogeneous displacement constraints,
sigma_dot = D(t) epsilon_dot implies K(t) u_dot = f_dot. Each unit response
therefore multiplies the load increment at its own application time. Responses
may contain every surface node and multiple prescribed thermal scenarios.
"""

import numpy as np


def cumulative_response(unit_responses, load_increments, *, initial_response=0.):
    """Accumulate nodal responses before applying a peak-response criterion.

    The first axis enumerates load events. Signed increments allow unloading;
    a zero increment represents holding the load while the material hardens.
    initial_response is the nodal displacement of an equilibrated starting
    state. The supplied increments are changes from that state's load.
    Units of the inputs must be consistent.
    """
    response = np.asarray(unit_responses, dtype=float)
    increments = np.asarray(load_increments, dtype=float)
    if increments.ndim != 1 or response.shape[0] != increments.size:
        raise ValueError("One load increment is required per response field")
    weights = increments.reshape((-1,) + (1,) * (response.ndim - 1))
    return np.asarray(initial_response) + np.cumsum(response * weights, axis=0)


def two_stage_fraction_interval(early_response, late_response, limit, *, initial_response=0.):
    """Return the admissible early fraction of a fixed total load, or None.

    Inputs are downward nodal responses to the full final load, and the allowed
    downward displacement. The initial_response consumes the same nodal budget;
    both stages and the initial state must satisfy it at every node/scenario.
    No ordering of nodal responses or fixed peak location is assumed.
    """
    early, late, limit, initial = np.broadcast_arrays(
        np.asarray(early_response, dtype=float),
        np.asarray(late_response, dtype=float), np.asarray(limit, dtype=float),
        np.asarray(initial_response, dtype=float))
    if np.any(initial > limit):
        return None
    limit = limit - initial
    slopes = np.concatenate((early.ravel(), (early - late).ravel()))
    bounds = np.concatenate((limit.ravel(), (limit - late).ravel()))
    if not (np.isfinite(slopes).all() and np.isfinite(bounds).all()):
        raise ValueError("Response constraints must be finite")
    if np.any((slopes == 0) & (bounds < 0)):
        return None
    positive, negative = slopes > 0, slopes < 0
    upper = min(1., np.min(bounds[positive] / slopes[positive], initial=np.inf))
    lower = max(0., np.max(bounds[negative] / slopes[negative], initial=-np.inf))
    return (float(lower), float(upper)) if lower <= upper else None
