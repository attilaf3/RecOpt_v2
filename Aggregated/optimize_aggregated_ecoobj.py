import time
from typing import Optional

import numpy as np
import pulp
from pandas import DataFrame
from pulp import LpStatusOptimal


def optimize(
    p_pv,
    p_ue,
    p_ut=None,
    dt=0.25,
    size_elh=None,
    size_bess=None,
    size_hss=None,
    vol_hss_water=None,
    run_lp=False,
    **kwargs
):
    """
    Aggregated single-site optimization:
      - own PV
      - own aggregated consumer
      - own community battery (but economically same meter / same owner)
      - external grid
      - optional optimized electric boiler + hot water storage

    Economic logic:
      - PV -> load: free
      - PV -> battery: free
      - PV -> grid export: 5 Ft/kWh revenue
      - Grid -> load: 36 / 71 Ft/kWh depending on common annual import tier
      - Grid -> battery: 36 / 71 Ft/kWh on the SAME annual import tier
      - Battery -> load: no internal transfer price, only optional degradation / penalty
      - No energy sharing / REC internal pricing

    Parameters
    ----------
    p_pv : array-like [kW]
        PV generation profile.
    p_ue : array-like [kW]
        Uncontrolled/general electric load.
    p_ut : array-like [kW_th] or None
        Thermal demand profile, only used if boiler is optimized.
        If boiler_mode='electric_load', this may be None.
    dt : float
        Time step in hours. For 15 min, dt=0.25.
    size_elh : float or None
        Electric heater max electrical power [kW].
    size_bess : float or None
        Battery energy capacity [kWh].
    size_hss : float or None
        Only acts as an enable flag for optimized thermal storage mode.
    vol_hss_water : float or None
        Boiler tank volume [liters]. If None, kwargs value/default is used.
    run_lp : bool
        If True, relax binaries.

    kwargs
    ------
    Boiler / thermal:
        boiler_mode: "electric_load" or "thermal_optimized"
        p_boiler_el: array-like [kW], required if boiler_mode="electric_load"
        eta_elh: float
        c_hss: float, default 0.001163  [kWh/kg/°C]
        a_hss: float, default 0.01275    [kW/°C]
        T_env: float, default 20
        T_max: float, default 55
        T_in: float, default 12
        T_init: float, default T_max

    Battery:
        eta_bess_in, eta_bess_out, eta_bess_stor
        t_bess_min
        soc_bess_min, soc_bess_max, soc_bess_init

    Economic:
        c_grid_cheap: 36
        c_grid_expensive: 71
        c_export: 5
        annual_cheap_limit_kwh: 2523
        c_grid_batt_penalty: 0         [Ft/kWh]
        c_batt_deg: 0                  [Ft/kWh throughput]
        allow_batt_export: False

    Solver:
        msg, gapRel, timeLimit
        solver: "GUROBI" or "CBC"

    Returns
    -------
    results, status, objective_detail, n_vars, n_cons, infeas_gap, grid_to_batt_energy
    """

    # ------------------------------------------------------------------
    # Input arrays
    # ------------------------------------------------------------------
    p_pv = np.asarray(p_pv, dtype=float).reshape(-1)
    p_ue = np.asarray(p_ue, dtype=float).reshape(-1)

    assert len(p_pv) == len(p_ue), "p_pv and p_ue must have the same length."
    n_timestep = len(p_pv)
    time_set = range(n_timestep)

    boiler_mode = kwargs.get("boiler_mode", "thermal_optimized")
    assert boiler_mode in ("electric_load", "thermal_optimized"), \
        "boiler_mode must be 'electric_load' or 'thermal_optimized'."

    p_boiler_el_input = kwargs.get("p_boiler_el", None)

    if boiler_mode == "electric_load":
        assert p_boiler_el_input is not None, \
            "For boiler_mode='electric_load', provide kwargs['p_boiler_el']."
        p_boiler_el_input = np.asarray(p_boiler_el_input, dtype=float).reshape(-1)
        assert len(p_boiler_el_input) == n_timestep, \
            "p_boiler_el must have same length as p_pv."
        p_ut_arr = np.zeros(n_timestep, dtype=float)
    else:
        assert p_ut is not None, \
            "For boiler_mode='thermal_optimized', p_ut thermal demand is required."
        p_ut_arr = np.asarray(p_ut, dtype=float).reshape(-1)
        assert len(p_ut_arr) == n_timestep, \
            "p_ut must have same length as p_pv."
        p_boiler_el_input = np.zeros(n_timestep, dtype=float)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    # Battery
    eta_bess_in = kwargs.get('eta_bess_in', 0.96)
    eta_bess_out = kwargs.get('eta_bess_out', 0.96)
    eta_bess_stor = kwargs.get('eta_bess_stor', 1.0)
    t_bess_min = kwargs.get('t_bess_min', 2.0)
    soc_bess_min = kwargs.get('soc_bess_min', 0.10)
    soc_bess_max = kwargs.get('soc_bess_max', 1.00)
    soc_bess_init = kwargs.get('soc_bess_init', soc_bess_min)

    # Boiler / HSS
    eta_elh = kwargs.get('eta_elh', 1.0)
    c_hss = kwargs.get('c_hss', 0.001163)   # kWh/kg/°C
    a_hss = kwargs.get('a_hss', 0.01275)    # kW/°C
    T_env = kwargs.get('T_env', 20.0)
    T_max = kwargs.get('T_max', 55.0)
    T_in = kwargs.get('T_in', 12.0)
    T_min = T_in
    T_init = kwargs.get('T_init', T_max)

    if vol_hss_water is None:
        vol_hss_water = kwargs.get('vol_hss_water', 120.0)

    # Economic [Ft/kWh]
    c_grid_cheap = kwargs.get('c_grid_cheap', 36.0)
    c_grid_expensive = kwargs.get('c_grid_expensive', 71.0)
    c_export = kwargs.get('c_export', 5.0)
    annual_cheap_limit_kwh = kwargs.get('annual_cheap_limit_kwh', 2523.0)
    c_grid_batt_penalty = kwargs.get('c_grid_batt_penalty', 0.0)
    c_batt_deg = kwargs.get('c_batt_deg', 0.0)

    # Optional policy / technical switches
    allow_batt_export = kwargs.get('allow_batt_export', False)
    p_grid_bess_max = kwargs.get('p_grid_bess_max', None)  # if None, defaults to battery charge power
    objective = kwargs.get('objective', "economic")
    assert objective in ("economic", "environmental"), \
        "objective must be 'economic' or 'environmental'."

    # Solver params
    msg = kwargs.get('msg', True)
    gapRel = kwargs.get('gapRel', None)
    timeLimit = kwargs.get('timeLimit', None)
    solver_name = kwargs.get('solver', 'GUROBI').upper()

    # ------------------------------------------------------------------
    # Flags and derived values
    # ------------------------------------------------------------------
    elh_flag = boiler_mode == "thermal_optimized" and size_elh is not None and size_elh > 0
    hss_flag = boiler_mode == "thermal_optimized" and size_hss is not None
    bess_flag = size_bess is not None and size_bess > 0

    if boiler_mode == "thermal_optimized":
        assert elh_flag, "For thermal_optimized mode, size_elh must be > 0."
        assert hss_flag, "For thermal_optimized mode, size_hss must be provided as enable flag."

    battery_power = (size_bess / t_bess_min) if bess_flag else 0.0
    p_grid_bess_max = battery_power if p_grid_bess_max is None else float(p_grid_bess_max)

    # Some robust big-M values
    base_max_load = float(np.max(p_ue)) if n_timestep else 0.0
    max_boiler_el_input = float(np.max(p_boiler_el_input)) if boiler_mode == "electric_load" else (size_elh or 0.0)
    M_grid_import = max(1.0, base_max_load + max_boiler_el_input + battery_power)
    M_grid_export = max(1.0, float(np.max(p_pv)) + battery_power)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    prob = pulp.LpProblem("AggregatedSingleSiteOpt", pulp.LpMinimize)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    # Energy flow variables [kW]
    p_pv_load = [pulp.LpVariable(f'Ppv_load_{t}', lowBound=0) for t in time_set]
    p_pv_bess = [pulp.LpVariable(f'Ppv_bess_{t}', lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f'Ppv_grid_{t}', lowBound=0) for t in time_set]

    p_grid_load = [pulp.LpVariable(f'Pgrid_load_{t}', lowBound=0) for t in time_set]
    p_grid_bess = [pulp.LpVariable(f'Pgrid_bess_{t}', lowBound=0) for t in time_set]

    p_bess_load = [pulp.LpVariable(f'Pbess_load_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    p_bess_grid = [pulp.LpVariable(f'Pbess_grid_{t}', lowBound=0) if bess_flag else 0 for t in time_set]

    p_bess_in = [pulp.LpVariable(f'Pbess_in_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    p_bess_out = [pulp.LpVariable(f'Pbess_out_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    e_bess_stor = [pulp.LpVariable(f'Ebess_stor_{t}', lowBound=0) if bess_flag else 0 for t in time_set]

    d_bess = [pulp.LpVariable(f'Dbess_{t}', cat=pulp.LpBinary) if bess_flag and not run_lp else 0 for t in time_set]
    d_grid = [pulp.LpVariable(f'Dgrid_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    p_grid_in = [pulp.LpVariable(f'Pgrid_in_{t}', lowBound=0) for t in time_set]   # export to external grid
    p_grid_out = [pulp.LpVariable(f'Pgrid_out_{t}', lowBound=0) for t in time_set] # import from external grid

    # Boiler electric-load mode
    p_elh_in = [0 for _ in time_set]
    p_elh_out = [0 for _ in time_set]
    p_hss_in = [0 for _ in time_set]
    p_hss_out = [0 for _ in time_set]
    e_hss_stor = [0 for _ in time_set]
    t_hss = [0 for _ in time_set]

    if boiler_mode == "thermal_optimized":
        p_elh_in = [pulp.LpVariable(f'Pelh_in_{t}', lowBound=0) for t in time_set]
        p_elh_out = [pulp.LpVariable(f'Pelh_out_{t}', lowBound=0) for t in time_set]
        p_hss_in = [pulp.LpVariable(f'Phss_in_{t}', lowBound=0) for t in time_set]
        p_hss_out = [pulp.LpVariable(f'Phss_out_{t}', lowBound=0) for t in time_set]
        # Keep this only as derived reporting variable? No, here we tie it to T_hss.
        e_hss_stor = [pulp.LpVariable(f'Ehss_stor_{t}', lowBound=0) for t in time_set]
        t_hss = [pulp.LpVariable(f'Thss_{t}', lowBound=T_min, upBound=T_max) for t in time_set]

    # Annual tariff split variables [kWh]
    e_grid_import_cheap = pulp.LpVariable('Egrid_import_cheap', lowBound=0)
    e_grid_import_expensive = pulp.LpVariable('Egrid_import_expensive', lowBound=0)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    for t in time_set:
        k = (t + 1) % n_timestep

        # PV allocation
        prob += p_pv_load[t] + p_pv_bess[t] + p_pv_grid[t] == p_pv[t], f"{t}_PV_balance"

        # Battery flow decomposition
        if bess_flag:
            prob += p_bess_in[t] == p_pv_bess[t] + p_grid_bess[t], f"{t}_BESS_charge_split"
            prob += p_bess_out[t] == p_bess_load[t] + p_bess_grid[t], f"{t}_BESS_discharge_split"
        else:
            prob += p_pv_bess[t] == 0, f"{t}_NoBESS_pv_bess"
            prob += p_grid_bess[t] == 0, f"{t}_NoBESS_grid_bess"

        # Grid flow decomposition
        prob += p_grid_out[t] == p_grid_load[t] + p_grid_bess[t], f"{t}_Grid_import_split"
        prob += p_grid_in[t] == p_pv_grid[t] + (p_bess_grid[t] if bess_flag else 0), f"{t}_Grid_export_split"

        # Total electric load
        if boiler_mode == "electric_load":
            total_load_expr = float(p_ue[t]) + float(p_boiler_el_input[t])
        else:
            total_load_expr = float(p_ue[t]) + p_elh_in[t]

        # Load supply balance
        prob += p_pv_load[t] + p_grid_load[t] + (p_bess_load[t] if bess_flag else 0) == total_load_expr, \
            f"{t}_Load_balance"

        # Grid mutual exclusivity
        if not run_lp:
            prob += p_grid_out[t] <= M_grid_import * d_grid[t], f"{t}_Grid_import_limit"
            prob += p_grid_in[t] <= M_grid_export * (1 - d_grid[t]), f"{t}_Grid_export_limit"

        # Battery constraints
        if bess_flag:
            # Power limits
            if run_lp:
                prob += p_bess_in[t] <= battery_power, f"{t}_BESS_max_input"
                prob += p_bess_out[t] <= battery_power, f"{t}_BESS_max_output"
            else:
                prob += p_bess_in[t] <= battery_power * d_bess[t], f"{t}_BESS_max_input"
                prob += p_bess_out[t] <= battery_power * (1 - d_bess[t]), f"{t}_BESS_max_output"

            # Optional separate grid->battery limit
            prob += p_grid_bess[t] <= p_grid_bess_max, f"{t}_Grid_to_BESS_limit"

            # If battery export not allowed
            if not allow_batt_export:
                prob += p_bess_grid[t] == 0, f"{t}_No_BESS_export"

            # SOC bounds
            prob += e_bess_stor[t] <= size_bess * soc_bess_max, f"{t}_BESS_SOC_max"
            prob += e_bess_stor[t] >= size_bess * soc_bess_min, f"{t}_BESS_SOC_min"

            # Battery dynamics
            prob += e_bess_stor[k] == (
                e_bess_stor[t] * eta_bess_stor
                + (p_bess_in[t] * eta_bess_in - p_bess_out[t] / eta_bess_out) * dt
            ), f"{t}_BESS_balance"

        # Boiler / HSS optimized mode
        if boiler_mode == "thermal_optimized":
            # Heater constitutive law
            prob += p_elh_out[t] == eta_elh * p_elh_in[t], f"{t}_ELH_constitutive"
            prob += p_elh_in[t] <= size_elh, f"{t}_ELH_power_max"

            # Heat flow links
            prob += p_hss_in[t] == p_elh_out[t], f"{t}_ELH_to_HSS"
            prob += p_hss_out[t] == float(p_ut_arr[t]), f"{t}_Thermal_demand_supply"

            # Temperature-energy relation
            prob += e_hss_stor[t] == vol_hss_water * c_hss * (t_hss[t] - T_in), f"{t}_HSS_energy_temp"

            # Tank dynamics
            prob += (
                vol_hss_water * c_hss * (t_hss[k] - t_hss[t])
                == (p_hss_in[t] - p_hss_out[t]) * dt
                - a_hss * (t_hss[t] - T_env) * dt
            ), f"{t}_HSS_balance"

            # Initial state can be controlled through T_init by fixing t=0 if not cyclic desired
            # Here we keep cyclic dynamics, but force the current state to be feasible around T_init if user wants:
            # optional softer approach: fix first state
            # We do it only at t=0 for determinacy.
            if t == 0:
                prob += t_hss[t] == T_init, f"{t}_HSS_initial_temp"

    # Annual import split for tariff tiers
    total_import_energy_expr = pulp.lpSum([p_grid_out[t] * dt for t in time_set])
    prob += e_grid_import_cheap + e_grid_import_expensive == total_import_energy_expr, "Annual_import_split"
    prob += e_grid_import_cheap <= annual_cheap_limit_kwh, "Annual_cheap_limit"

    # Fix initial battery SOC at t=0 for determinacy
    if bess_flag:
        prob += e_bess_stor[0] == size_bess * soc_bess_init, "BESS_initial_SOC"

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    if objective == 'economic':
        # Consumer bill minimization for one aggregated site
        prob += (
            c_grid_cheap * e_grid_import_cheap
            + c_grid_expensive * e_grid_import_expensive
            - c_export * pulp.lpSum([p_grid_in[t] * dt for t in time_set])
            + c_grid_batt_penalty * pulp.lpSum([p_grid_bess[t] * dt for t in time_set])
            + c_batt_deg * pulp.lpSum([
                ((p_bess_in[t] + p_bess_out[t]) * dt) if bess_flag else 0
                for t in time_set
            ])
        )
    else:
        # Environmental / grid interaction minimization
        prob += pulp.lpSum([
            (p_grid_out[t] + p_grid_in[t]) * dt
            for t in time_set
        ])

    # Debug LP export
    if run_lp:
        prob.writeLP("debug_agg_single_site.lp")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    t0 = time.time()

    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    else:
        solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)

    status = prob.solve(solver)
    if status != LpStatusOptimal:
        raise RuntimeError(f"Unable to solve the problem! Solver status: {pulp.LpStatus[status]}")

    solve_time = time.time() - t0
    objective_value = float(pulp.value(prob.objective))

    # ------------------------------------------------------------------
    # Extract values
    # ------------------------------------------------------------------
    def _val(x):
        v = pulp.value(x)
        return 0.0 if v is None else float(v)

    for t in time_set:
        p_pv_load[t] = _val(p_pv_load[t])
        p_pv_bess[t] = _val(p_pv_bess[t])
        p_pv_grid[t] = _val(p_pv_grid[t])

        p_grid_load[t] = _val(p_grid_load[t])
        p_grid_bess[t] = _val(p_grid_bess[t])

        p_grid_in[t] = _val(p_grid_in[t])
        p_grid_out[t] = _val(p_grid_out[t])

        if bess_flag:
            p_bess_load[t] = _val(p_bess_load[t])
            p_bess_grid[t] = _val(p_bess_grid[t])
            p_bess_in[t] = _val(p_bess_in[t])
            p_bess_out[t] = _val(p_bess_out[t])
            e_bess_stor[t] = _val(e_bess_stor[t])
            d_bess[t] = _val(d_bess[t])

        d_grid[t] = _val(d_grid[t])

        if boiler_mode == "thermal_optimized":
            p_elh_in[t] = _val(p_elh_in[t])
            p_elh_out[t] = _val(p_elh_out[t])
            p_hss_in[t] = _val(p_hss_in[t])
            p_hss_out[t] = _val(p_hss_out[t])
            e_hss_stor[t] = _val(e_hss_stor[t])
            t_hss[t] = _val(t_hss[t])

    e_grid_import_cheap_val = _val(e_grid_import_cheap)
    e_grid_import_expensive_val = _val(e_grid_import_expensive)

    if not bess_flag:
        p_bess_load = np.zeros(n_timestep)
        p_bess_grid = np.zeros(n_timestep)
        p_bess_in = np.zeros(n_timestep)
        p_bess_out = np.zeros(n_timestep)
        e_bess_stor = np.zeros(n_timestep)
        d_bess = np.zeros(n_timestep)

    if boiler_mode == "electric_load":
        p_elh_in = np.array(p_boiler_el_input, dtype=float)
        p_elh_out = np.array(p_boiler_el_input, dtype=float) * eta_elh
        p_hss_in = np.zeros(n_timestep)
        p_hss_out = np.zeros(n_timestep)
        e_hss_stor = np.zeros(n_timestep)
        t_hss = np.zeros(n_timestep)
    else:
        p_elh_in = np.array(p_elh_in, dtype=float)
        p_elh_out = np.array(p_elh_out, dtype=float)
        p_hss_in = np.array(p_hss_in, dtype=float)
        p_hss_out = np.array(p_hss_out, dtype=float)
        e_hss_stor = np.array(e_hss_stor, dtype=float)
        t_hss = np.array(t_hss, dtype=float)

    # Derived KPI arrays
    p_total_load = p_ue + (p_boiler_el_input if boiler_mode == "electric_load" else p_elh_in)
    p_self_consumed_pv = np.array(p_pv_load) + np.array(p_pv_bess)
    p_direct_self_consumed_pv = np.array(p_pv_load)

    results = dict(
        p_pv=np.array(p_pv, dtype=float),
        p_ue=np.array(p_ue, dtype=float),
        p_ut=np.array(p_ut_arr, dtype=float),

        p_total_load=np.array(p_total_load, dtype=float),

        p_pv_load=np.array(p_pv_load, dtype=float),
        p_pv_bess=np.array(p_pv_bess, dtype=float),
        p_pv_grid=np.array(p_pv_grid, dtype=float),

        p_grid_load=np.array(p_grid_load, dtype=float),
        p_grid_bess=np.array(p_grid_bess, dtype=float),

        p_bess_load=np.array(p_bess_load, dtype=float),
        p_bess_grid=np.array(p_bess_grid, dtype=float),
        p_bess_in=np.array(p_bess_in, dtype=float),
        p_bess_out=np.array(p_bess_out, dtype=float),
        e_bess_stor=np.array(e_bess_stor, dtype=float),

        p_grid_in=np.array(p_grid_in, dtype=float),    # export
        p_grid_out=np.array(p_grid_out, dtype=float),  # import

        p_elh_in=np.array(p_elh_in, dtype=float),
        p_elh_out=np.array(p_elh_out, dtype=float),
        p_hss_in=np.array(p_hss_in, dtype=float),
        p_hss_out=np.array(p_hss_out, dtype=float),
        e_hss_stor=np.array(e_hss_stor, dtype=float),
        t_hss=np.array(t_hss, dtype=float),

        p_self_consumed_pv=np.array(p_self_consumed_pv, dtype=float),
        p_direct_self_consumed_pv=np.array(p_direct_self_consumed_pv, dtype=float),

        d_bess=np.array(d_bess, dtype=float),
        d_grid=np.array(d_grid, dtype=float),
    )

    # Financial KPIs [Ft]
    annual_export_energy = float(np.sum(results["p_grid_in"]) * dt)
    annual_import_energy = float(np.sum(results["p_grid_out"]) * dt)
    annual_grid_to_batt_energy = float(np.sum(results["p_grid_bess"]) * dt)
    annual_batt_throughput = float(np.sum(results["p_bess_in"] + results["p_bess_out"]) * dt)

    grid_energy_cost = c_grid_cheap * e_grid_import_cheap_val + c_grid_expensive * e_grid_import_expensive_val
    export_revenue = c_export * annual_export_energy
    grid_batt_penalty_cost = c_grid_batt_penalty * annual_grid_to_batt_energy
    batt_deg_cost = c_batt_deg * annual_batt_throughput

    if objective == "economic":
        objective_detail = {
            "economic_total_bill_ft": objective_value,
            "grid_energy_cost_ft": float(grid_energy_cost),
            "export_revenue_ft": float(export_revenue),
            "grid_batt_penalty_ft": float(grid_batt_penalty_cost),
            "battery_degradation_ft": float(batt_deg_cost),
            "cheap_import_kwh": float(e_grid_import_cheap_val),
            "expensive_import_kwh": float(e_grid_import_expensive_val),
        }
    else:
        objective_detail = {
            "environmental_grid_interaction_kwh": objective_value
        }

    # Some convenience KPIs
    sum_batt_to_grid = float(np.sum(results["p_bess_grid"]) * dt)
    infeas_gap = None
    n_vars = prob.numVariables()
    n_cons = prob.numConstraints()

    # Extra helpful KPIs
    results["kpi"] = {
        "solve_time_sec": solve_time,
        "annual_import_kwh": annual_import_energy,
        "annual_export_kwh": annual_export_energy,
        "annual_grid_to_batt_kwh": annual_grid_to_batt_energy,
        "annual_batt_throughput_kwh": annual_batt_throughput,
        "annual_self_consumed_pv_kwh": float(np.sum(results["p_self_consumed_pv"]) * dt),
        "annual_direct_self_consumed_pv_kwh": float(np.sum(results["p_direct_self_consumed_pv"]) * dt),
        "annual_total_load_kwh": float(np.sum(results["p_total_load"]) * dt),
        "annual_pv_kwh": float(np.sum(results["p_pv"]) * dt),
        "self_consumption_ratio": float(
            0.0 if np.sum(results["p_pv"]) <= 0 else np.sum(results["p_self_consumed_pv"]) / np.sum(results["p_pv"])
        ),
        "self_sufficiency_ratio": float(
            0.0 if np.sum(results["p_total_load"]) <= 0
            else 1.0 - np.sum(results["p_grid_out"]) / np.sum(results["p_total_load"])
        ),
    }

    return results, status, objective_detail, n_vars, n_cons, infeas_gap, annual_grid_to_batt_energy


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('input.csv', sep=';', index_col=0, parse_dates=True)

    # 15 min annual data assumed; if your file is already 15 min, do not resample.
    # Example keeps native resolution:
    df_filtered = df.copy()

    na_p_consumed = df_filtered["consumer1"].to_numpy(dtype=float)
    na_p_pv = df_filtered["pv1"].to_numpy(dtype=float)

    # If optimized boiler mode is used:
    na_p_ut = df_filtered["thermal_user1"].to_numpy(dtype=float)

    results, status, objective, n_vars, n_cons, infeas_gap, annual_grid_to_batt = optimize(
        p_pv=na_p_pv,
        p_ue=na_p_consumed,
        p_ut=na_p_ut,
        dt=0.25,
        size_elh=2.0,
        size_hss=1.0,          # enable flag for thermal optimized boiler mode
        vol_hss_water=120.0,
        size_bess=12.0,

        boiler_mode="thermal_optimized",   # or "electric_load"
        # p_boiler_el=df_filtered["boiler_el"].to_numpy(dtype=float),  # needed only if electric_load

        objective="economic",

        # Battery
        eta_bess_in=0.96,
        eta_bess_out=0.96,
        eta_bess_stor=1.0,
        t_bess_min=2.0,
        soc_bess_min=0.10,
        soc_bess_max=1.00,
        soc_bess_init=0.20,

        # Boiler
        eta_elh=1.0,
        T_env=20.0,
        T_in=12.0,
        T_max=55.0,
        T_init=45.0,
        a_hss=0.01275,

        # Economics [Ft/kWh]
        c_grid_cheap=36.0,
        c_grid_expensive=71.0,
        c_export=5.0,
        annual_cheap_limit_kwh=2523.0,
        c_grid_batt_penalty=0.0,   # put >0 if you want to discourage grid charging
        c_batt_deg=0.0,

        allow_batt_export=False,
        p_grid_bess_max=6.0,

        solver="GUROBI",   # or "CBC"
        gapRel=0.0005,
        msg=True,
    )

    print("Status:", pulp.LpStatus[status])
    print("Objective:", objective)
    print("n_vars:", n_vars, "n_cons:", n_cons)
    print("Annual grid->battery [kWh]:", annual_grid_to_batt)
    print("KPIs:", results["kpi"])