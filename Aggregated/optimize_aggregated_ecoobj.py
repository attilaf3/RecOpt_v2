import time
import numpy as np
import pulp


def optimize_aggregated(
    p_pv,
    p_ue,
    p_el_heater=None,
    p_dhw=None,
    dt=0.25,
    size_elh=None,
    size_bess=None,
    vol_hss_water=None,
    run_lp=False,
    **kwargs
):
    """
    Aggregált gazdasági optimalizálás:
      - egy aggregált PV
      - egy aggregált fogyasztó
      - egy aggregált BESS
      - opcionális aggregált bojler/HSS

    Két bojler mód:
      1) boiler_mode="electric_load"
         A bojler fix villamos terhelésként szerepel (p_el_heater).

      2) boiler_mode="thermal_optimized"
         Az ELH + HSS optimalizált, a hőigény p_dhw.
    """

    # ------------------------------------------------------------------
    # Bemenetek
    # ------------------------------------------------------------------
    p_pv = np.asarray(p_pv, dtype=float).reshape(-1)
    p_ue = np.asarray(p_ue, dtype=float).reshape(-1)

    assert len(p_pv) == len(p_ue), "p_pv és p_ue hossza egyezzen."
    n_timestep = len(p_pv)
    time_set = range(n_timestep)


    boiler_mode = kwargs.get("boiler_mode", "thermal_optimized")
    assert boiler_mode in ("electric_load", "thermal_optimized")

    if boiler_mode == "electric_load":
        if p_el_heater is None:
            raise ValueError("electric_load módban p_el_heater kötelező.")
        p_el_heater = np.asarray(p_el_heater, dtype=float).reshape(-1)
        if len(p_el_heater) != len(p_pv):
            raise ValueError("p_el_heater hossza nem egyezik p_pv-vel.")
        p_dhw = np.zeros_like(p_pv)
    else:
        if p_dhw is None:
            raise ValueError("thermal_optimized módban p_dhw kötelező.")
        p_dhw = np.asarray(p_dhw, dtype=float).reshape(-1)
        if len(p_dhw) != len(p_pv):
            raise ValueError("p_dhw hossza nem egyezik p_pv-vel.")
        p_el_heater = np.zeros_like(p_pv)

    # ------------------------------------------------------------------
    # Paraméterek
    # ------------------------------------------------------------------
    n_household = int(kwargs.get("n_household", 1))
    assert n_household >= 1, "n_household legalább 1 legyen."

    # Battery
    eta_bess_in = kwargs.get("eta_bess_in", 0.98)
    eta_bess_out = kwargs.get("eta_bess_out", 0.96)
    eta_bess_stor = kwargs.get("eta_bess_stor", 0.995)
    t_bess_min = kwargs.get("t_bess_min", 2.0)
    soc_bess_min = kwargs.get("soc_bess_min", 0.2)
    soc_bess_max = kwargs.get("soc_bess_max", 1.0)
    soc_bess_init = kwargs.get("soc_bess_init", soc_bess_min)

    # HSS / bojler
    eta_elh = kwargs.get("eta_elh", 1.0)
    c_hss = kwargs.get("c_hss", 0.00116667)    # kWh/kg/°C
    a_hss = kwargs.get("a_hss", 0.00125)       # kW/°C
    T_env = kwargs.get("T_env", 20.0)
    T_max = kwargs.get("T_max", 65.0)
    T_in = kwargs.get("T_in", 10.0)
    T_min = kwargs.get("T_min", T_in)
    T_init = kwargs.get("T_init", T_max)
    vol_hss_water = kwargs.get("vol_hss_water", 120.0) if vol_hss_water is None else vol_hss_water

    # Gazdasági paraméterek [Ft/kWh]
    c_grid_cheap = kwargs.get("c_grid_cheap", 36.0)
    c_grid_expensive = kwargs.get("c_grid_expensive", 71.0)
    c_export = kwargs.get("c_export", 5.0)
    annual_cheap_limit_kwh = kwargs.get("annual_cheap_limit_kwh", 2523.0 * n_household)

    # Solver
    msg = kwargs.get("msg", True)
    gapRel = kwargs.get("gapRel", None)
    timeLimit = kwargs.get("timeLimit", None)
    objective = kwargs.get("objective", "economic")
    assert objective in ("economic", "environmental"), \
        "objective csak 'economic' vagy 'environmental' lehet."

    # ------------------------------------------------------------------
    # Flag-ek
    # ------------------------------------------------------------------
    bess_flag = size_bess is not None and size_bess > 0
    hss_flag = (boiler_mode == "thermal_optimized") and (size_elh is not None) and (size_elh > 0) and (vol_hss_water is not None) and (vol_hss_water > 0)

    battery_power = (size_bess / t_bess_min) if bess_flag else 0.0

    max_base_load = float(np.max(p_ue)) if n_timestep else 0.0
    max_boiler_load = float(np.max(p_el_heater)) if boiler_mode == "electric_load" else (size_elh or 0.0)

    M_grid_import = max(1.0, max_base_load + max_boiler_load + battery_power)
    M_grid_export = max(1.0, float(np.max(p_pv)) + battery_power)

    # ------------------------------------------------------------------
    # Modell
    # ------------------------------------------------------------------
    prob = pulp.LpProblem("AggregatedEconomicOpt", pulp.LpMinimize)

    # ------------------------------------------------------------------
    # Döntési változók
    # ------------------------------------------------------------------
    # PV felosztás
    p_pv_load = [pulp.LpVariable(f"Ppv_load_{t}", lowBound=0) for t in time_set]
    p_pv_bess = [pulp.LpVariable(f"Ppv_bess_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"Ppv_grid_{t}", lowBound=0) for t in time_set]

    # Grid felosztás
    p_grid_load = [pulp.LpVariable(f"Pgrid_load_{t}", lowBound=0) for t in time_set]
    p_grid_bess = [pulp.LpVariable(f"Pgrid_bess_{t}", lowBound=0) for t in time_set]

    p_grid_out = [pulp.LpVariable(f"Pgrid_out_{t}", lowBound=0) for t in time_set]  # import
    p_grid_in = [pulp.LpVariable(f"Pgrid_in_{t}", lowBound=0) for t in time_set]    # export

    d_grid = [pulp.LpVariable(f"Dgrid_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # BESS
    p_bess_load = [pulp.LpVariable(f"Pbess_load_{t}", lowBound=0) if bess_flag else 0 for t in time_set]
    p_bess_grid = [pulp.LpVariable(f"Pbess_grid_{t}", lowBound=0) if bess_flag else 0 for t in time_set]
    p_bess_in = [pulp.LpVariable(f"Pbess_in_{t}", lowBound=0) if bess_flag else 0 for t in time_set]
    p_bess_out = [pulp.LpVariable(f"Pbess_out_{t}", lowBound=0) if bess_flag else 0 for t in time_set]
    e_bess_stor = [pulp.LpVariable(f"Ebess_stor_{t}", lowBound=0) if bess_flag else 0 for t in time_set]
    d_bess = [pulp.LpVariable(f"Dbess_{t}", cat=pulp.LpBinary) if bess_flag and not run_lp else 0 for t in time_set]

    # ELH + HSS
    p_elh_in = [0 for _ in time_set]
    p_elh_out = [0 for _ in time_set]
    p_hss_in = [0 for _ in time_set]
    p_hss_out = [0 for _ in time_set]
    e_hss_stor = [0 for _ in time_set]
    t_hss = [0 for _ in time_set]

    if hss_flag:
        p_elh_in = [pulp.LpVariable(f"Pelh_in_{t}", lowBound=0) for t in time_set]
        p_elh_out = [pulp.LpVariable(f"Pelh_out_{t}", lowBound=0) for t in time_set]
        p_hss_in = [pulp.LpVariable(f"Phss_in_{t}", lowBound=0) for t in time_set]
        p_hss_out = [pulp.LpVariable(f"Phss_out_{t}", lowBound=0) for t in time_set]
        e_hss_stor = [pulp.LpVariable(f"Ehss_stor_{t}", lowBound=0) for t in time_set]
        t_hss = [pulp.LpVariable(f"Thss_{t}", lowBound=T_min, upBound=T_max) for t in time_set]

    # Éves tarifa bontás
    e_grid_import_cheap = pulp.LpVariable("Egrid_import_cheap", lowBound=0)
    e_grid_import_expensive = pulp.LpVariable("Egrid_import_expensive", lowBound=0)

    # ------------------------------------------------------------------
    # Korlátok
    # ------------------------------------------------------------------
    for t in time_set:
        k = (t + 1) % n_timestep

        # PV balance
        prob += p_pv_load[t] + p_pv_bess[t] + p_pv_grid[t] == p_pv[t], f"{t}_PV_balance"

        # BESS split
        if bess_flag:
            prob += p_bess_in[t] == p_pv_bess[t] + p_grid_bess[t], f"{t}_BESS_charge_split"
            prob += p_bess_out[t] == p_bess_load[t] + p_bess_grid[t], f"{t}_BESS_discharge_split"
        else:
            prob += p_pv_bess[t] == 0, f"{t}_NoBESS_pv_bess"
            prob += p_grid_bess[t] == 0, f"{t}_NoBESS_grid_bess"

        # Grid split
        prob += p_grid_out[t] == p_grid_load[t] + p_grid_bess[t], f"{t}_Grid_import_split"
        prob += p_grid_in[t] == p_pv_grid[t] + (p_bess_grid[t] if bess_flag else 0), f"{t}_Grid_export_split"

        # Összes villamos terhelés
        if boiler_mode == "electric_load":
            total_load = float(p_ue[t] + p_el_heater[t])
        else:
            total_load = float(p_ue[t]) + p_elh_in[t]

        prob += p_pv_load[t] + p_grid_load[t] + (p_bess_load[t] if bess_flag else 0) == total_load, \
            f"{t}_Load_balance"

        # Grid import/export kizárás
        if not run_lp:
            prob += p_grid_out[t] <= M_grid_import * d_grid[t], f"{t}_Grid_import_limit"
            prob += p_grid_in[t] <= M_grid_export * (1 - d_grid[t]), f"{t}_Grid_export_limit"

        # BESS
        if bess_flag:
            if run_lp:
                prob += p_bess_in[t] <= battery_power, f"{t}_BESS_max_in"
                prob += p_bess_out[t] <= battery_power, f"{t}_BESS_max_out"
            else:
                prob += p_bess_in[t] <= battery_power * d_bess[t], f"{t}_BESS_max_in"
                prob += p_bess_out[t] <= battery_power * (1 - d_bess[t]), f"{t}_BESS_max_out"

            prob += p_grid_bess[t] <= battery_power, f"{t}_Grid_to_BESS_limit"

            prob += e_bess_stor[t] <= size_bess * soc_bess_max, f"{t}_BESS_SOC_max"
            prob += e_bess_stor[t] >= size_bess * soc_bess_min, f"{t}_BESS_SOC_min"

            prob += e_bess_stor[k] == (
                e_bess_stor[t] * eta_bess_stor
                + (p_bess_in[t] * eta_bess_in - p_bess_out[t] / eta_bess_out) * dt
            ), f"{t}_BESS_balance"

        # HSS / ELH
        if hss_flag:
            prob += p_elh_out[t] == eta_elh * p_elh_in[t], f"{t}_ELH_constitutive"
            prob += p_elh_in[t] <= size_elh, f"{t}_ELH_max"

            prob += p_hss_in[t] == p_elh_out[t], f"{t}_ELH_to_HSS"
            prob += p_hss_out[t] == float(p_dhw[t]), f"{t}_DHW_supply"

            prob += e_hss_stor[t] == vol_hss_water * c_hss * (t_hss[t] - T_in), f"{t}_HSS_energy_temp"

            prob += (
                    vol_hss_water * c_hss * (t_hss[k] - t_hss[t])
                    == (p_hss_in[t] - p_hss_out[t]) * dt
                    - a_hss * (t_hss[t] - T_env) * dt
            ), f"{t}_HSS_balance"

            prob += p_hss_out[k] <= vol_hss_water * c_hss * (t_hss[t] - T_in) / dt, f"{t}_HSS_max_out"
            prob += p_hss_in[t] <= vol_hss_water * c_hss * (T_max - t_hss[t]) / dt, f"{t}_HSS_max_in"

            if t == 0:
                prob += t_hss[t] == T_init, f"{t}_HSS_initial_temp"

    # Éves olcsó/drága tarifa felosztás
    total_import_energy = pulp.lpSum([p_grid_out[t] * dt for t in time_set])
    prob += e_grid_import_cheap + e_grid_import_expensive == total_import_energy, "Annual_import_split"
    prob += e_grid_import_cheap <= annual_cheap_limit_kwh, "Annual_cheap_limit"

    if bess_flag:
        prob += e_bess_stor[0] == size_bess * soc_bess_init, "BESS_initial_SOC"

    # ------------------------------------------------------------------
    # Célfüggvény
    # ------------------------------------------------------------------
    if objective == "economic":
        prob += (
            c_grid_cheap * e_grid_import_cheap
            + c_grid_expensive * e_grid_import_expensive
            - c_export * pulp.lpSum([p_grid_in[t] * dt for t in time_set])
        )
    else:
        prob += pulp.lpSum([(p_grid_out[t] + p_grid_in[t]) * dt for t in time_set])

    if run_lp:
        prob.writeLP("debug_aggregated.lp")

    # ------------------------------------------------------------------
    # Megoldás
    # ------------------------------------------------------------------
    t0 = time.time()
    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Nem sikerült megoldani. Solver státusz: {pulp.LpStatus[status]}")

    solve_time = time.time() - t0
    objective_value = float(pulp.value(prob.objective))

    # ------------------------------------------------------------------
    # Eredmények kiolvasása
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
        p_grid_out[t] = _val(p_grid_out[t])
        p_grid_in[t] = _val(p_grid_in[t])

        d_grid[t] = _val(d_grid[t])

        if bess_flag:
            p_bess_load[t] = _val(p_bess_load[t])
            p_bess_grid[t] = _val(p_bess_grid[t])
            p_bess_in[t] = _val(p_bess_in[t])
            p_bess_out[t] = _val(p_bess_out[t])
            e_bess_stor[t] = _val(e_bess_stor[t])
            d_bess[t] = _val(d_bess[t])

        if hss_flag:
            p_elh_in[t] = _val(p_elh_in[t])
            p_elh_out[t] = _val(p_elh_out[t])
            p_hss_in[t] = _val(p_hss_in[t])
            p_hss_out[t] = _val(p_hss_out[t])
            e_hss_stor[t] = _val(e_hss_stor[t])
            t_hss[t] = _val(t_hss[t])

    cheap_import_kwh = _val(e_grid_import_cheap)
    expensive_import_kwh = _val(e_grid_import_expensive)

    if not bess_flag:
        p_bess_load = np.zeros(n_timestep)
        p_bess_grid = np.zeros(n_timestep)
        p_bess_in = np.zeros(n_timestep)
        p_bess_out = np.zeros(n_timestep)
        e_bess_stor = np.zeros(n_timestep)
        d_bess = np.zeros(n_timestep)

    if not hss_flag:
        p_elh_in = np.array(p_el_heater, dtype=float) if boiler_mode == "electric_load" else np.zeros(n_timestep)
        p_elh_out = np.array(p_elh_in, dtype=float) * eta_elh
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

    p_total_load = p_ue + (p_el_heater if boiler_mode == "electric_load" else p_elh_in)
    p_self_consumed_pv = np.array(p_pv_load) + np.array(p_pv_bess)
    p_direct_self_consumed_pv = np.array(p_pv_load)

    results = dict(
        p_pv=np.array(p_pv, dtype=float),
        p_ue=np.array(p_ue, dtype=float),
        p_dhw=np.array(p_dhw, dtype=float),
        p_el_heater=np.array(p_el_heater, dtype=float),

        p_total_load=np.array(p_total_load, dtype=float),

        p_pv_load=np.array(p_pv_load, dtype=float),
        p_pv_bess=np.array(p_pv_bess, dtype=float),
        p_pv_grid=np.array(p_pv_grid, dtype=float),

        p_grid_load=np.array(p_grid_load, dtype=float),
        p_grid_bess=np.array(p_grid_bess, dtype=float),
        p_grid_out=np.array(p_grid_out, dtype=float),
        p_grid_in=np.array(p_grid_in, dtype=float),

        p_bess_load=np.array(p_bess_load, dtype=float),
        p_bess_grid=np.array(p_bess_grid, dtype=float),
        p_bess_in=np.array(p_bess_in, dtype=float),
        p_bess_out=np.array(p_bess_out, dtype=float),
        e_bess_stor=np.array(e_bess_stor, dtype=float),

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

    annual_export_energy = float(np.sum(results["p_grid_in"]) * dt)
    annual_import_energy = float(np.sum(results["p_grid_out"]) * dt)
    annual_grid_to_batt_energy = float(np.sum(results["p_grid_bess"]) * dt)

    grid_energy_cost = c_grid_cheap * cheap_import_kwh + c_grid_expensive * expensive_import_kwh
    export_revenue = c_export * annual_export_energy

    if objective == "economic":
        objective_detail = {
            "economic_total_bill_ft": objective_value,
            "grid_energy_cost_ft": float(grid_energy_cost),
            "export_revenue_ft": float(export_revenue),
            "cheap_import_kwh": float(cheap_import_kwh),
            "expensive_import_kwh": float(expensive_import_kwh),
            "annual_import_kwh": float(annual_import_energy),
            "annual_export_kwh": float(annual_export_energy),
            "solve_time_sec": float(solve_time),
        }
    else:
        objective_detail = {
            "environmental_grid_interaction_kwh": objective_value,
            "annual_import_kwh": float(annual_import_energy),
            "annual_export_kwh": float(annual_export_energy),
            "solve_time_sec": float(solve_time),
        }

    n_vars = prob.numVariables()
    n_cons = prob.numConstraints()
    infeas_gap = None

    return results, status, objective_detail, n_vars, n_cons, infeas_gap, annual_grid_to_batt_energy
