import numpy as np
import pulp
from Utility.configuration import config


def hp_profiles_from_temperature(
    T_out,
    H_tot_W_per_K,
    T_set=23.0,
    T_out_design=-18.0,
    T_out_mild=23.0,
    T_su_nom=45.0,
    T_su_min=30.0,
    cop_a0=5.06,
    cop_a1=-0.05,
    cop_min=0.5,
):
    """
    Linear pre-processing for an air-water heat pump.

    Returns time-series arrays that can be used as fixed coefficients in an LP/MILP:
      - q_hp_heat: thermal heating demand per timestep, in the same energy-step
        convention as the optimizer input profiles.
      - cop: COP(T_out), treated as a known coefficient.
      - p_hp_el: fixed electric energy per timestep if the heat pump must exactly
        cover q_hp_heat in the same timestep.

    Note: the optimizer files in this project use 15-minute profile values as
    per-step kWh-like quantities for annual billing. This helper follows that
    convention. If your T_out profile is paired with actual kW heat demand, pass
    q_hp_heat directly to individual_opt_hp instead.
    """
    T_out = np.asarray(T_out, dtype=float).ravel()

    # Heating demand: Q = H_tot * max(T_set - T_out, 0).
    # W -> kW-equivalent per profile step, consistent with the existing scripts.
    q_hp_heat = np.maximum(float(H_tot_W_per_K) * (float(T_set) - T_out), 0.0) / 1000.0

    # Heating curve, clipped between nominal and minimum supply temperature.
    m_su = (float(T_su_nom) - float(T_su_min)) / (float(T_out_design) - float(T_out_mild))
    T_su_raw = float(T_su_min) + m_su * (T_out - float(T_out_mild))
    T_su = np.clip(T_su_raw, float(T_su_min), float(T_su_nom))

    deltaT = T_su - T_out
    cop = np.maximum(float(cop_a0) + float(cop_a1) * deltaT, float(cop_min))
    p_hp_el = np.divide(q_hp_heat, cop, out=np.zeros_like(q_hp_heat), where=cop > 1e-9)

    return {
        "T_out": T_out,
        "q_hp_heat": q_hp_heat,
        "T_su": T_su,
        "deltaT": deltaT,
        "cop": cop,
        "p_hp_el": p_hp_el,
    }


def individual_opt_hp(
    p_pv,                   # (T,) kWh/step in existing project convention
    p_ue,                   # (T,) kWh/step - base electric consumption
    p_hp_el=None,           # (T,) kWh/step - fixed HP electric use, optional
    q_hp_heat=None,         # (T,) thermal kWh/step, optional alternative to p_hp_el
    cop=None,               # (T,) COP coefficients for q_hp_heat -> p_hp_el
    T_out=None,             # (T,) degC, optional alternative for deriving q_hp_heat/COP
    H_tot_W_per_K=None,     # W/K, required if T_out is used
    T_set=23.0,
    T_out_design=-18.0,
    T_out_mild=23.0,
    T_su_nom=45.0,
    T_su_min=30.0,
    cop_a0=5.06,
    cop_a1=-0.05,
    cop_min=0.5,
    dt=0.25,                # h, kept for metadata/API compatibility
    price_grid_low=config.getfloat("tariffs", "grid_a_low_ft_per_kwh"),
    price_grid_high=config.getfloat("tariffs", "grid_a_high_ft_per_kwh"),
    price_pv_grid=config.getfloat("tariffs", "pv_export_ft_per_kwh"),
    grid_low_cap_kwh=config.getfloat("tariffs", "grid_a_low_limit_kwh"),
    run_lp=False,
    msg=False,
    gapRel=None,
    timeLimit=None,
):
    """
    Individual economic optimization with PV + heat pump.

    Model idea:
      - Precompute COP(t) from outdoor temperature, or pass COP(t) directly.
      - The LP sees COP(t) as a fixed coefficient, so heat-pump electricity is
        linear: p_hp_el[t] = q_hp_heat[t] / COP[t].
      - PV can serve base load, HP load, or export to the grid.
      - Grid can serve base load and HP load.
      - No community sharing and no thermal storage are included here.

    Energy-flow logic:
      PV   -> own base load / own HP / external grid export
      Grid -> own base load / own HP

    If you later want flexible HP scheduling, add a thermal storage/building-state
    equation, e.g. e_th[t+1] = a*e_th[t] + COP[t]*p_hp_in[t] - q_heat[t].
    That remains linear as long as COP[t] is a known parameter.
    """
    p_pv = np.asarray(p_pv, dtype=float).ravel()
    p_ue = np.asarray(p_ue, dtype=float).ravel()
    assert p_pv.shape == p_ue.shape, "p_pv és p_ue hossza nem egyezik."

    T = len(p_pv)
    time_set = range(T)

    # Build HP electric profile from the most specific available input.
    hp_extra = {}
    if p_hp_el is not None:
        p_hp_el = np.asarray(p_hp_el, dtype=float).ravel()
        q_hp_heat_v = np.zeros_like(p_hp_el) if q_hp_heat is None else np.asarray(q_hp_heat, dtype=float).ravel()
        cop_v = np.ones_like(p_hp_el) if cop is None else np.asarray(cop, dtype=float).ravel()
    elif q_hp_heat is not None and cop is not None:
        q_hp_heat_v = np.asarray(q_hp_heat, dtype=float).ravel()
        cop_v = np.asarray(cop, dtype=float).ravel()
        p_hp_el = np.divide(q_hp_heat_v, cop_v, out=np.zeros_like(q_hp_heat_v), where=cop_v > 1e-9)
    elif T_out is not None and H_tot_W_per_K is not None:
        hp_extra = hp_profiles_from_temperature(
            T_out=T_out,
            H_tot_W_per_K=H_tot_W_per_K,
            T_set=T_set,
            T_out_design=T_out_design,
            T_out_mild=T_out_mild,
            T_su_nom=T_su_nom,
            T_su_min=T_su_min,
            cop_a0=cop_a0,
            cop_a1=cop_a1,
            cop_min=cop_min,
        )
        p_hp_el = hp_extra["p_hp_el"]
        q_hp_heat_v = hp_extra["q_hp_heat"]
        cop_v = hp_extra["cop"]
    else:
        p_hp_el = np.zeros_like(p_ue)
        q_hp_heat_v = np.zeros_like(p_ue)
        cop_v = np.ones_like(p_ue)

    p_hp_el = np.asarray(p_hp_el, dtype=float).ravel()
    q_hp_heat_v = np.asarray(q_hp_heat_v, dtype=float).ravel()
    cop_v = np.asarray(cop_v, dtype=float).ravel()

    assert p_pv.shape == p_ue.shape == p_hp_el.shape == q_hp_heat_v.shape == cop_v.shape, \
        "p_pv, p_ue, p_hp_el/q_hp_heat/cop hossza nem egyezik."

    total_load = p_ue + p_hp_el

    prob = pulp.LpProblem("individual_opt_hp", pulp.LpMinimize)

    # Big-M for import/export exclusivity.
    M_pv = float(np.max(p_pv)) + 1.0 if T > 0 else 1.0
    M_grid = float(np.max(total_load)) + 1.0 if T > 0 else 1.0

    # PV split.
    p_pv_load = [pulp.LpVariable(f"p_pv_load_{t}", lowBound=0) for t in time_set]
    p_pv_hp = [pulp.LpVariable(f"p_pv_hp_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    # Load supply split.
    p_grid_load = [pulp.LpVariable(f"p_grid_load_{t}", lowBound=0) for t in time_set]
    p_grid_hp = [pulp.LpVariable(f"p_grid_hp_{t}", lowBound=0) for t in time_set]

    # Grid import/export.
    p_grid_import = [pulp.LpVariable(f"p_grid_import_{t}", lowBound=0) for t in time_set]
    p_grid_export = [pulp.LpVariable(f"p_grid_export_{t}", lowBound=0) for t in time_set]
    d_grid = [pulp.LpVariable(f"d_grid_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Tariff blocks, matching individual_opt_bess.py.
    e_grid_low_step = [pulp.LpVariable(f"e_grid_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_high_step = [pulp.LpVariable(f"e_grid_high_step_{t}", lowBound=0) for t in time_set]
    rem_low = [pulp.LpVariable(f"rem_low_{t}", lowBound=0, upBound=grid_low_cap_kwh) for t in time_set]

    for t in time_set:
        # PV split.
        prob += p_pv_load[t] + p_pv_hp[t] + p_pv_grid[t] == p_pv[t], f"pv_split_{t}"

        # Separate base-load and HP-load balances. This makes HP contribution explicit.
        prob += p_pv_load[t] + p_grid_load[t] == p_ue[t], f"base_load_balance_{t}"
        prob += p_pv_hp[t] + p_grid_hp[t] == p_hp_el[t], f"hp_load_balance_{t}"

        # Grid meter definitions.
        prob += p_grid_import[t] == p_grid_load[t] + p_grid_hp[t], f"grid_import_def_{t}"
        prob += p_grid_export[t] == p_pv_grid[t], f"grid_export_def_{t}"

        # No simultaneous import/export in MIP mode.
        if not run_lp:
            prob += p_grid_import[t] <= M_grid * d_grid[t], f"grid_import_gate_{t}"
            prob += p_grid_export[t] <= M_pv * (1 - d_grid[t]), f"grid_export_gate_{t}"

        # Gross 15-minute import split into regulated/above-cap tariff blocks.
        prob += e_grid_low_step[t] + e_grid_high_step[t] == p_grid_import[t], f"grid_step_split_{t}"
        if t == 0:
            prob += rem_low[t] == grid_low_cap_kwh - e_grid_low_step[t], "rem_low_init"
            prob += e_grid_low_step[t] <= grid_low_cap_kwh, f"grid_low_step_cap_{t}"
        else:
            prob += rem_low[t] == rem_low[t - 1] - e_grid_low_step[t], f"rem_low_balance_{t}"
            prob += e_grid_low_step[t] <= rem_low[t - 1], f"grid_low_step_cap_{t}"

    cost_grid = pulp.lpSum(
        price_grid_low * e_grid_low_step[t] + price_grid_high * e_grid_high_step[t]
        for t in time_set
    )
    revenue_export = pulp.lpSum(price_pv_grid * p_pv_grid[t] for t in time_set)
    prob += cost_grid - revenue_export

    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)

    status_str = pulp.LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Not Solved", "Integer Feasible", "Undefined"}:
        raise RuntimeError(f"Hiba: {status_str}")

    def _val(x):
        v = pulp.value(x)
        return 0.0 if v is None else float(v)

    p_pv_load_v = np.array([_val(v) for v in p_pv_load], dtype=float)
    p_pv_hp_v = np.array([_val(v) for v in p_pv_hp], dtype=float)
    p_pv_grid_v = np.array([_val(v) for v in p_pv_grid], dtype=float)
    p_grid_load_v = np.array([_val(v) for v in p_grid_load], dtype=float)
    p_grid_hp_v = np.array([_val(v) for v in p_grid_hp], dtype=float)
    p_grid_import_v = np.array([_val(v) for v in p_grid_import], dtype=float)
    p_grid_export_v = np.array([_val(v) for v in p_grid_export], dtype=float)
    d_grid_v = np.array([_val(v) if not run_lp else 0.0 for v in d_grid], dtype=float)
    e_grid_low_step_v = np.array([_val(v) for v in e_grid_low_step], dtype=float)
    e_grid_high_step_v = np.array([_val(v) for v in e_grid_high_step], dtype=float)
    rem_low_v = np.array([_val(v) for v in rem_low], dtype=float)

    e_grid_low_v = float(np.sum(e_grid_low_step_v))
    e_grid_high_v = float(np.sum(e_grid_high_step_v))
    e_grid_total_v = e_grid_low_v + e_grid_high_v
    grid_cost = float(np.sum(price_grid_low * e_grid_low_step_v + price_grid_high * e_grid_high_step_v))
    export_revenue = float(np.sum(p_pv_grid_v) * price_pv_grid)
    net_cost = grid_cost - export_revenue

    results = {
        "p_pv": p_pv,
        "p_ue": p_ue,
        "p_hp_el": p_hp_el,
        "q_hp_heat": q_hp_heat_v,
        "cop": cop_v,
        "p_total_load": total_load,
        "dt": dt,
        "p_pv_load": p_pv_load_v,
        "p_pv_hp": p_pv_hp_v,
        "p_pv_grid": p_pv_grid_v,
        "p_grid_load": p_grid_load_v,
        "p_grid_hp": p_grid_hp_v,
        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "d_grid": d_grid_v,
        "e_grid_low": e_grid_low_v,
        "e_grid_high": e_grid_high_v,
        "e_grid_total": e_grid_total_v,
        "e_grid_export": float(np.sum(p_grid_export_v)),
        "hp_el_kwh": float(np.sum(p_hp_el)),
        "hp_heat_kwh": float(np.sum(q_hp_heat_v)),
        "pv_to_hp_kwh": float(np.sum(p_pv_hp_v)),
        "grid_to_hp_kwh": float(np.sum(p_grid_hp_v)),
        "grid_cost_Ft": float(grid_cost),
        "grid_export_revenue_Ft": float(export_revenue),
        "net_cost_Ft": float(net_cost),
        "objective_Ft": float(_val(prob.objective)),
        "status": status_str,
        "e_grid_low_step": e_grid_low_step_v,
        "e_grid_high_step": e_grid_high_step_v,
        "remaining_low_block_kwh": rem_low_v,
    }
    results.update({f"hp_{k}": v for k, v in hp_extra.items() if k not in results})
    return results
