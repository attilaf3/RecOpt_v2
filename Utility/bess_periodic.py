"""Periodic initialization for causal greedy battery simulations."""
import numpy as np
from Utility.result_schema import add_bess_states, add_bess_flows


def periodic_simulation(simulate, initial, *, dt=0.25, tolerance=1e-7, max_iterations=128):
    """Find a start state whose full-horizon simulation returns to that state.

    No artificial final charge/discharge is added. If the controller cannot
    reach a periodic trajectory, fail instead of reporting a cyclic result.
    """
    state = np.asarray(initial, dtype=float)
    for _ in range(max_iterations):
        result, end, lower, upper = simulate(state)
        end = np.asarray(end, dtype=float)
        if np.any(end < np.asarray(lower)-tolerance) or np.any(end > np.asarray(upper)+tolerance):
            raise ValueError("No valid cyclic greedy BESS state: insufficient energy to maintain SOC bounds")
        if np.max(np.abs(end-state), initial=0.0) <= tolerance:
            ts = result["timeseries"]
            history = np.asarray(ts["e_bess"])
            if np.any(history < np.asarray(lower)-tolerance) or np.any(history > np.asarray(upper)+tolerance):
                raise ValueError("Greedy BESS violates SOC bounds within the cycle")
            add_bess_states(ts, np.concatenate((state.reshape((1,)+state.shape), history), axis=0))
            add_bess_flows(ts, dt)
            result["bess_terminal"] = "cyclic"
            return result
        state = end
    raise ValueError("Cyclic greedy BESS initialization did not converge")
