"""Battery state and operating constraints for the common household model."""
from types import SimpleNamespace
import pulp


def build_bess(prob, T, *, dt, size, eta_in, eta_out, retention, soc_min,
               soc_max, soc_init, min_hours, run_lp, terminal="cyclic", grid_charge=None,
               min_mode_steps=4, allow_grid_charge_for_min_soc=False, prefix=""):
    """Cyclic SOC; two exclusive operating modes; optional minimum-SOC guard."""
    if terminal != "cyclic":
        raise ValueError("Only cyclic BESS SOC is supported")
    selected = "min_soc" if allow_grid_charge_for_min_soc else "pv_only"
    if grid_charge is not None and grid_charge != selected:
        raise ValueError("Use allow_grid_charge_for_min_soc; unrestricted grid charging is not supported")
    grid_charge = selected
    if not (size >= 0 and 0 <= soc_min <= soc_init <= soc_max <= 1):
        raise ValueError("Invalid BESS capacity or SOC bounds")
    if not (0 < eta_in <= 1 and 0 < eta_out <= 1 and 0 < retention <= 1 and min_hours > 0):
        raise ValueError("Invalid BESS efficiency or power parameter")
    power = size / min_hours
    def vector(name):
        return [pulp.LpVariable(f"{prefix}{name}_{t}", lowBound=0, upBound=power) for t in range(T)]
    b = SimpleNamespace(power=power, pv=vector("p_pv_bess"),
        grid=vector("p_grid_bess"), base=vector("p_bess_base"),
        boiler=vector("p_bess_boiler"), fixed=vector("p_bess_fixed"),
        charge=vector("p_bess_in"), discharge=vector("p_bess_out"),
        state=[pulp.LpVariable(f"{prefix}bess_state_{t}", lowBound=size*soc_min,
                              upBound=size*soc_max) for t in range(T+1)],
        charge_on=[pulp.LpVariable(f"{prefix}bess_charge_on_{t}", cat="Binary")
                   if not run_lp and size > 0 else 0 for t in range(T)],
        discharge_on=[pulp.LpVariable(f"{prefix}bess_discharge_on_{t}", cat="Binary")
                      if not run_lp and size > 0 else 0 for t in range(T)])
    prob += b.state[0] == size*soc_init, f"{prefix}bess_initial"
    if terminal == "cyclic":
        prob += b.state[T] == b.state[0], f"{prefix}bess_terminal"
    for t in range(T):
        prob += b.charge[t] == b.pv[t] + b.grid[t], f"{prefix}bess_charge_balance_{t}"
        prob += b.discharge[t] == b.base[t]+b.boiler[t]+b.fixed[t], f"{prefix}bess_discharge_balance_{t}"
        pre = b.state[t]*retention + dt*(b.pv[t]*eta_in-b.discharge[t]/eta_out)
        prob += b.state[t+1] == pre + dt*eta_in*b.grid[t], f"{prefix}bess_state_balance_{t}"
        if grid_charge == "pv_only":
            prob += b.grid[t] == 0, f"{prefix}bess_no_grid_charge_{t}"
        elif grid_charge == "min_soc":
            # Grid energy is allowed only to restore the minimum state.
            deficit = size*soc_min-pre
            added = dt*eta_in*b.grid[t]
            prob += added >= deficit, f"{prefix}bess_guard_lower_{t}"
            if size > 0:
                guard = pulp.LpVariable(f"{prefix}bess_guard_{t}", cat="Binary")
                big_m = size + dt*power*(eta_in+1/eta_out)
                prob += added <= deficit+big_m*(1-guard), f"{prefix}bess_guard_exact_{t}"
                prob += added <= big_m*guard, f"{prefix}bess_guard_upper_{t}"
        if not run_lp and size > 0:
            prob += b.charge_on[t]+b.discharge_on[t] <= 1, f"{prefix}bess_mode_{t}"
            prob += b.charge[t] <= power*b.charge_on[t], f"{prefix}bess_charge_gate_{t}"
            prob += b.discharge[t] <= power*b.discharge_on[t], f"{prefix}bess_discharge_gate_{t}"
    if not run_lp and size > 0:
        steps = min(T, max(1, int(min_mode_steps)))
        for name, mode in (("charge", b.charge_on), ("discharge", b.discharge_on)):
            for t in range(T):
                prob += pulp.lpSum(mode[(t+j) % T] for j in range(steps)) >= steps*(mode[t]-mode[(t-1) % T]), f"{prefix}bess_min_{name}_{t}"
    return b
