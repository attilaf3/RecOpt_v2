"""Shared greedy BESS energy limits. Inputs/outputs are kWh per step."""


def pv_charge(soc, surplus, upper, efficiency, limit, allowed=True):
    return max(0.0, min(surplus, limit, (upper-soc)/max(efficiency, 1e-12))) if allowed else 0.0


def load_discharge(soc, deficit, lower, efficiency, limit, allowed=True):
    return max(0.0, min(deficit, limit, (soc-lower)*efficiency)) if allowed else 0.0


def minimum_grid_charge(soc, lower, efficiency, limit, pv_charged=0.0, *, enabled=False, allowed=True):
    """After PV dispatch: refill only the minimum, using remaining charge capacity."""
    if not enabled or not allowed:
        return 0.0
    return max(0.0, min(max(limit-pv_charged, 0.0), (lower-soc)/max(efficiency, 1e-12)))
