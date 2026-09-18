"""Capacity-limited energy allocation, independent of tariffs and dispatch."""
import numpy as np


def allocate_pool(capacities, total, mode="proportional"):
    values = np.asarray(capacities, dtype=float)
    if mode not in ("equal", "proportional"):
        raise ValueError("Allocation mode must be equal or proportional")
    if values.ndim != 1 or not np.all(np.isfinite(values)) or not np.isfinite(total):
        raise ValueError("Expected finite one-dimensional capacities and total")
    values = np.maximum(values, 0.0)
    remaining = min(max(float(total), 0.0), float(values.sum()))
    out = np.zeros_like(values)
    if mode == "proportional":
        return values * (remaining / values.sum()) if values.sum() > 0 else out
    while remaining > 1e-12:
        room = values - out
        active = room > 1e-12
        if not np.any(active):
            break
        take = np.minimum(room[active], remaining / active.sum())
        out[active] += take
        remaining -= float(take.sum())
    return out


def match_energy(deficit, surplus, buyer_mode="proportional", seller_mode="proportional"):
    """Return balanced buyer/seller allocations without changing dispatch."""
    total = min(np.maximum(deficit, 0).sum(), np.maximum(surplus, 0).sum())
    return (allocate_pool(deficit, total, buyer_mode),
            allocate_pool(surplus, total, seller_mode))
