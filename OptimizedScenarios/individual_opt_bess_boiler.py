import numpy as np
import pulp


def individual_opt_bess_boiler(
    p_pv,                   # (T,) kW
    p_ue,                   # (T,) kW
    p_dhw,                  # (T,) kW_th  (hőigény teljesítményben)
    dt=0.25,                # óra
    # --- Boiler / HSS ---
    size_elh=0.0,           # kW (elektromos fűtőszál max teljesítmény)
    vol_hss_water=0.0,      # liter ~ kg
    T_env=20.0,             # °C
    T_max=65.0,             # °C
    T_min=10.0,             # °C
    T_in=10.0,              # °C
    a_hss=0.01275,          # kW/°C (hőveszteség tényező)
    eta_elh=0.95,           # -
    enforce_cl_rules=True,
    cl_max_on_hours_per_day=8.0,
    cl_min_midday_hours_per_day=4.0,
    # --- BESS ---
    size_bess=0.0,          # kWh
    eta_bess_in=0.98,
    eta_bess_out=0.96,
    eta_bess_stor=0.995,
    soc_bess_min=0.10,
    soc_bess_max=0.90,
    soc_bess_init=None,
    t_bess_min=2.0,         # h  -> max P = size_bess / t_bess_min
    # --- Tariffs (bruttó, 15 perces elszámolás, de low/high éves blokk) ---
    p_el_heater_fixed=None,

    price_grid_a_low=36.0,
    price_grid_a_high=71.0,
    price_grid_b_low=23.0,
    price_grid_b_high=61.0,
    price_pv_grid=5.0,
    grid_a_low_cap_kwh=2523.0,
    grid_b_low_cap_kwh=2523.0,
    objective="grid",
    # --- Solve ---
    run_lp=False,           # ha True: LP relax (binárisok kikapcsolva)
    msg=False,
    gapRel=None,
    timeLimit=None,
):
    """
    PV + optimalizált bojler (HSS) + BESS (opcionális), megosztás nélkül.
    p_* mindenhol kW, energia (kWh) mindig sum(p)*dt.

    Bruttó elszámolás:
      - import: kWh-ként fizetjük (low/high blokk kimerítéssel, 15 percenként bookolva)
      - export: külön bevétel (Ft/kWh)
    """
    p_pv = np.asarray(p_pv, float).ravel()
    p_ue = np.asarray(p_ue, float).ravel()
    p_dhw = np.asarray(p_dhw, float).ravel()
    if p_el_heater_fixed is None:
        p_el_heater_fixed = np.zeros_like(p_ue)
    else:
        p_el_heater_fixed = np.asarray(p_el_heater_fixed, dtype=float).ravel()

    assert p_el_heater_fixed.shape == p_ue.shape, "p_el_heater_fixed hossza nem egyezik."
    assert p_pv.shape == p_ue.shape == p_dhw.shape, "p_pv, p_ue, p_dhw hossza nem egyezik"

    T = len(p_pv)
    time_set = range(T)

    has_pv = np.sum(p_pv) > 1e-9
    hss_active = has_pv and (float(size_elh) > 1e-9) and (float(vol_hss_water) > 1e-9)
    fixed_boiler_active = (not hss_active) and (np.sum(p_el_heater_fixed) > 1e-9)
    bess_active = float(size_bess) > 1e-9
    if soc_bess_init is None:
        soc_bess_init = soc_bess_min

    # konstans: kWh/(liter*°C)
    c_hss = 0.00116667

    # BESS max teljesítmény (kW)
    battery_power = float(size_bess) / max(float(t_bess_min), 1e-9) if bess_active else 0.0

    prob = pulp.LpProblem("ind_opt_boiler_bess_kw", pulp.LpMinimize)

    # Big-M
    M_pv = float(np.max(p_pv)) + 1.0 if T > 0 else 1.0
    # grid import max: load + boiler + bess charge
    M_grid = float(np.max(p_ue) + max(float(size_elh), 0.0) + max(battery_power, 0.0)) + 1.0 if T > 0 else 1.0

    # -----------------------------
    # Variables
    # -----------------------------
    # PV split
    p_pv_load = [pulp.LpVariable(f"p_pv_load_{t}", lowBound=0) for t in time_set]
    p_pv_elh  = [pulp.LpVariable(f"p_pv_elh_{t}", lowBound=0) for t in time_set]
    p_pv_bess = [pulp.LpVariable(f"p_pv_bess_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    # Grid splits
    p_grid_load = [pulp.LpVariable(f"p_grid_load_{t}", lowBound=0) for t in time_set]
    p_grid_elh  = [pulp.LpVariable(f"p_grid_elh_{t}", lowBound=0) for t in time_set]
    p_grid_bess = [pulp.LpVariable(f"p_grid_bess_{t}", lowBound=0) for t in time_set]
    p_grid_elh_fixed = [pulp.LpVariable(f"p_grid_elh_fixed_{t}", lowBound=0) for t in time_set]

    # BESS to load
    p_bess_load = [pulp.LpVariable(f"p_bess_load_{t}", lowBound=0) for t in time_set]

    # Battery internal
    p_bess_in  = [pulp.LpVariable(f"p_bess_in_{t}", lowBound=0) for t in time_set]
    p_bess_out = [pulp.LpVariable(f"p_bess_out_{t}", lowBound=0) for t in time_set]

    if bess_active:
        e_bess = [
            pulp.LpVariable(
                f"e_bess_{t}",
                lowBound=float(size_bess) * float(soc_bess_min),
                upBound=float(size_bess) * float(soc_bess_max),
            )
            for t in time_set
        ]
        d_bess = [pulp.LpVariable(f"d_bess_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    else:
        e_bess = [0.0] * T
        d_bess = [0.0] * T

    # HSS / ELH
    if hss_active:
        p_elh_in  = [pulp.LpVariable(f"p_elh_in_{t}", lowBound=0, upBound=float(size_elh)) for t in time_set]  # kW
        p_hss_in  = [pulp.LpVariable(f"p_hss_in_{t}", lowBound=0) for t in time_set]                          # kW_th
        p_hss_out = [pulp.LpVariable(f"p_hss_out_{t}", lowBound=0) for t in time_set]                         # kW_th
        t_hss     = [pulp.LpVariable(f"t_hss_{t}", lowBound=float(T_min), upBound=float(T_max)) for t in time_set]
        d_cl      = [pulp.LpVariable(f"d_cl_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    else:
        p_elh_in  = [0.0] * T
        p_hss_in  = [0.0] * T
        p_hss_out = [0.0] * T
        t_hss     = [0.0] * T
        d_cl      = [0.0] * T

    # Grid import/export (kW)
    p_grid_import = [pulp.LpVariable(f"p_grid_import_{t}", lowBound=0) for t in time_set]
    p_grid_export = [pulp.LpVariable(f"p_grid_export_{t}", lowBound=0) for t in time_set]
    d_grid = [pulp.LpVariable(f"d_grid_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]  # 1=import, 0=export

    # Tariff blocks per step (kWh/step)
    e_grid_a_low_step = [pulp.LpVariable(f"e_grid_a_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_a_high_step = [pulp.LpVariable(f"e_grid_a_high_step_{t}", lowBound=0) for t in time_set]
    rem_a_low = [pulp.LpVariable(f"rem_a_low_{t}", lowBound=0, upBound=float(grid_a_low_cap_kwh)) for t in time_set]

    e_grid_b_low_step = [pulp.LpVariable(f"e_grid_b_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_b_high_step = [pulp.LpVariable(f"e_grid_b_high_step_{t}", lowBound=0) for t in time_set]
    rem_b_low = [pulp.LpVariable(f"rem_b_low_{t}", lowBound=0, upBound=float(grid_b_low_cap_kwh)) for t in time_set]

    # -----------------------------
    # Constraints
    # -----------------------------
    for t in time_set:
        k = (t + 1) % T

        # PV split
        prob += (p_pv_load[t] + p_pv_elh[t] + p_pv_bess[t] + p_pv_grid[t] == p_pv[t]), f"pv_split_{t}"

        # UE load balance
        prob += (p_pv_load[t] + p_bess_load[t] + p_grid_load[t] == p_ue[t]), f"ue_balance_{t}"

        if fixed_boiler_active:
            prob += p_grid_elh_fixed[t] == p_el_heater_fixed[t], f"fixed_boiler_grid_{t}"
        else:
            prob += p_grid_elh_fixed[t] == 0, f"no_fixed_boiler_grid_{t}"

        # Boiler electrical supply
        if hss_active:
            prob += (p_elh_in[t] == p_pv_elh[t] + p_grid_elh[t]), f"elh_supply_{t}"
            prob += (p_hss_in[t] == eta_elh * p_elh_in[t]), f"hss_in_def_{t}"
            prob += (p_hss_out[t] == p_dhw[t]), f"dhw_balance_{t}"

            # HSS dynamics (kW -> kWh with *dt)
            prob += (
                float(vol_hss_water) * c_hss * (t_hss[k] - t_hss[t])
                == (p_hss_in[t] - p_hss_out[t] - float(a_hss) * (t_hss[t] - float(T_env))) * float(dt)
            ), f"hss_balance_{t}"

            # Physical limits (kW_th)
            # max discharge next step based on temperature at t
            prob += (
                p_hss_out[k] <= float(vol_hss_water) * c_hss * (t_hss[t] - float(T_in)) / float(dt)
            ), f"hss_max_out_{t}"

            # max charge based on remaining capacity
            prob += (
                p_hss_in[t] <= float(vol_hss_water) * c_hss * (float(T_max) - t_hss[t]) / float(dt)
            ), f"hss_charge_space_{t}"

            # On/off control
            if not run_lp:
                prob += p_elh_in[t] <= float(size_elh) * d_cl[t], f"elh_onoff_{t}"
                if t != 0 and t != T - 1:
                    prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1], f"cl_min_on_time_{t}"
        else:
            # no HSS -> no boiler power
            prob += p_pv_elh[t] == 0, f"no_hss_pv_elh_{t}"
            prob += p_grid_elh[t] == 0, f"no_hss_grid_elh_{t}"


        # Battery relations
        if not bess_active:
            prob += p_pv_bess[t] == 0, f"no_bess_pv_bess_{t}"
            prob += p_grid_bess[t] == 0, f"no_bess_grid_bess_{t}"
            prob += p_bess_load[t] == 0, f"no_bess_load_{t}"
            prob += p_bess_in[t] == 0, f"no_bess_in_{t}"
            prob += p_bess_out[t] == 0, f"no_bess_out_{t}"
        else:
            # charge
            prob += (p_bess_in[t] == p_pv_bess[t] + p_grid_bess[t]), f"bess_in_def_{t}"
            # discharge only to load
            prob += (p_bess_out[t] == p_bess_load[t]), f"bess_out_def_{t}"
            # dynamics
            prob += (
                e_bess[k]
                == e_bess[t] * float(eta_bess_stor)
                + (p_bess_in[t] * float(eta_bess_in) - p_bess_out[t] / max(float(eta_bess_out), 1e-9)) * float(dt)
            ), f"bess_balance_{t}"

            # power caps + mutual exclusivity
            if run_lp:
                prob += p_bess_in[t] <= battery_power, f"bess_in_cap_{t}"
                prob += p_bess_out[t] <= battery_power, f"bess_out_cap_{t}"
            else:
                prob += p_bess_in[t] <= d_bess[t] * battery_power, f"bess_in_gate_{t}"
                prob += p_bess_out[t] <= (1 - d_bess[t]) * battery_power, f"bess_out_gate_{t}"

        # Grid import/export definitions
        prob += (
                p_grid_import[t]
                == p_grid_load[t] + p_grid_elh[t] + p_grid_elh_fixed[t] + p_grid_bess[t]
        ), f"grid_import_def_{t}"
        prob += (p_grid_export[t] == p_pv_grid[t]), f"grid_export_def_{t}"

        # No simultaneous import and export
        if not run_lp:
            prob += p_grid_import[t] <= M_grid * d_grid[t], f"grid_import_gate_{t}"
            prob += p_grid_export[t] <= M_pv * (1 - d_grid[t]), f"grid_export_gate_{t}"

        # 15-min (step) import energy split into low/high blocks
        e_imp_a_t = float(dt) * (p_grid_load[t] + p_grid_bess[t])
        e_imp_b_t = float(dt) * (p_grid_elh[t] + p_grid_elh_fixed[t])

        prob += e_grid_a_low_step[t] + e_grid_a_high_step[t] == e_imp_a_t, f"grid_a_step_split_{t}"
        prob += e_grid_b_low_step[t] + e_grid_b_high_step[t] == e_imp_b_t, f"grid_b_step_split_{t}"

        if t == 0:
            prob += rem_a_low[t] == float(grid_a_low_cap_kwh) - e_grid_a_low_step[t], "rem_a_low_init"
            prob += e_grid_a_low_step[t] <= float(grid_a_low_cap_kwh), f"grid_a_low_step_cap_{t}"

            prob += rem_b_low[t] == float(grid_b_low_cap_kwh) - e_grid_b_low_step[t], "rem_b_low_init"
            prob += e_grid_b_low_step[t] <= float(grid_b_low_cap_kwh), f"grid_b_low_step_cap_{t}"
        else:
            prob += rem_a_low[t] == rem_a_low[t - 1] - e_grid_a_low_step[t], f"rem_a_low_balance_{t}"
            prob += e_grid_a_low_step[t] <= rem_a_low[t - 1], f"grid_a_low_step_cap_{t}"

            prob += rem_b_low[t] == rem_b_low[t - 1] - e_grid_b_low_step[t], f"rem_b_low_balance_{t}"
            prob += e_grid_b_low_step[t] <= rem_b_low[t - 1], f"grid_b_low_step_cap_{t}"

    # initial SOC
    if bess_active:
        prob += e_bess[0] == float(size_bess) * float(soc_bess_init), "bess_init"

    # daily CL rules
    if hss_active and enforce_cl_rules and (not run_lp):
        n_steps_day = int(round(24.0 / float(dt)))
        if abs(n_steps_day * float(dt) - 24.0) > 1e-9:
            raise ValueError("dt must divide 24h exactly (e.g., 0.25).")

        start_mid = int(round(10.0 / float(dt)))
        mid_len = int(round(6.0 / float(dt)))

        y_middle = [0] * n_steps_day
        for i in range(start_mid, min(start_mid + mid_len, n_steps_day)):
            y_middle[i] = 1

        max_on_steps = int(round(float(cl_max_on_hours_per_day) / float(dt)))
        min_mid_steps = int(round(float(cl_min_midday_hours_per_day) / float(dt)))

        for j in range(0, T, n_steps_day):
            day_idx = range(j, min(j + n_steps_day, T))
            if len(list(day_idx)) < n_steps_day:
                continue

            prob += pulp.lpSum(d_cl[t] for t in day_idx) <= max_on_steps, f"day_{j}_cl_maxon"
            prob += pulp.lpSum(d_cl[t] * y_middle[t - j] for t in day_idx) >= min_mid_steps, f"day_{j}_cl_midmin"

    if bess_active and not run_lp:
        min_on_steps = 4
        for t in range(1, T - min_on_steps + 1):
            start_up = d_bess[t] - d_bess[t - 1]
            prob += pulp.lpSum(d_bess[tau] for tau in range(t, t + min_on_steps)) >= min_on_steps * start_up

    # -----------------------------
    # Objective: economic (Ft)
    # -----------------------------
    if objective == "grid":
        prob += pulp.lpSum(
            float(dt) * (p_grid_import[t] + p_grid_export[t])
            for t in time_set
        )
    else:
        raise ValueError("objective must be 'grid'")

    # -----------------------------
    # Solve
    # -----------------------------
    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)
    status_str = pulp.LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Not Solved", "Integer Feasible", "Undefined"}:
        raise RuntimeError(f"Hiba: {status_str}")

    def _val(x):
        v = pulp.value(x)
        return 0.0 if v is None else float(v)

    p_grid_elh_fixed_v = np.array([_val(v) for v in p_grid_elh_fixed], dtype=float)

    e_grid_a_low_step_v = np.array([_val(v) for v in e_grid_a_low_step], dtype=float)
    e_grid_a_high_step_v = np.array([_val(v) for v in e_grid_a_high_step], dtype=float)
    e_grid_b_low_step_v = np.array([_val(v) for v in e_grid_b_low_step], dtype=float)
    e_grid_b_high_step_v = np.array([_val(v) for v in e_grid_b_high_step], dtype=float)

    e_grid_a_low_v = float(np.sum(e_grid_a_low_step_v))
    e_grid_a_high_v = float(np.sum(e_grid_a_high_step_v))
    e_grid_b_low_v = float(np.sum(e_grid_b_low_step_v))
    e_grid_b_high_v = float(np.sum(e_grid_b_high_step_v))

    import_cost_a_ft = (
            float(price_grid_a_low) * e_grid_a_low_v
            + float(price_grid_a_high) * e_grid_a_high_v
    )

    import_cost_b_ft = (
            float(price_grid_b_low) * e_grid_b_low_v
            + float(price_grid_b_high) * e_grid_b_high_v
    )

    e_export = float(np.sum([_val(v) for v in p_pv_grid]) * float(dt))
    export_rev = float(price_pv_grid) * e_export
    brt_bill = import_cost_a_ft + import_cost_b_ft - export_rev

    # outputs
    res = {
        "status": status_str,
        "objective_Ft": float(_val(prob.objective)),
        "dt": float(dt),

        # raw inputs
        "p_pv": p_pv,
        "p_ue": p_ue,
        "p_dhw": p_dhw,

        # PV split
        "p_pv_load": np.array([_val(v) for v in p_pv_load]),
        "p_pv_elh":  np.array([_val(v) for v in p_pv_elh]),
        "p_pv_bess": np.array([_val(v) for v in p_pv_bess]),
        "p_pv_grid": np.array([_val(v) for v in p_pv_grid]),

        # grid
        "p_grid_elh_fixed": p_grid_elh_fixed_v,
        "p_grid_to_base": np.array([_val(v) for v in p_grid_load]),
        "p_grid_to_boiler": np.array([_val(v) for v in p_grid_elh]) + p_grid_elh_fixed_v,
        "p_grid_bess": np.array([_val(v) for v in p_grid_bess]),
        "p_grid_import": np.array([_val(v) for v in p_grid_import]),
        "p_grid_export": np.array([_val(v) for v in p_grid_export]),

        # BESS
        "bess_active": int(bess_active),
        "size_bess": float(size_bess),
        "battery_power_kw": float(battery_power),
        "p_bess_load": np.array([_val(v) for v in p_bess_load]),
        "p_bess_in": np.array([_val(v) for v in p_bess_in]),
        "p_bess_out": np.array([_val(v) for v in p_bess_out]),
        "e_bess": np.array([_val(v) for v in e_bess]) if bess_active else np.zeros(T),
        "d_bess": np.array([_val(v) for v in d_bess]) if (bess_active and not run_lp) else np.zeros(T),

        # Boiler / HSS
        "hss_active": int(hss_active),
        "size_elh_kw": float(size_elh),
        "p_elh_in": np.array([_val(v) for v in p_elh_in]) if hss_active else np.zeros(T),
        "p_hss_in": np.array([_val(v) for v in p_hss_in]) if hss_active else np.zeros(T),
        "p_hss_out": np.array([_val(v) for v in p_hss_out]) if hss_active else np.zeros(T),
        "t_hss": np.array([_val(v) for v in t_hss]) if hss_active else np.zeros(T),
        "d_cl": np.array([_val(v) for v in d_cl]) if (hss_active and not run_lp) else np.zeros(T),

        # tariff bookkeeping
        "grid_import_a_kwh": float(np.sum(res["p_grid_load"] + res["p_grid_bess"]) * float(dt)),
        "grid_import_b_kwh": float(np.sum(res["p_grid_elh"] + p_grid_elh_fixed_v) * float(dt)),

        "grid_import_low_kwh": e_grid_a_low_v + e_grid_b_low_v,
        "grid_import_high_kwh": e_grid_a_high_v + e_grid_b_high_v,

        "grid_import_a_low_kwh": e_grid_a_low_v,
        "grid_import_a_high_kwh": e_grid_a_high_v,
        "grid_import_b_low_kwh": e_grid_b_low_v,
        "grid_import_b_high_kwh": e_grid_b_high_v,

        "grid_import_total_kwh": e_grid_a_low_v + e_grid_a_high_v + e_grid_b_low_v + e_grid_b_high_v,
        "grid_export_kwh": e_export,

        "import_cost_a_ft": import_cost_a_ft,
        "import_cost_b_ft": import_cost_b_ft,
        "import_cost_ft": import_cost_a_ft + import_cost_b_ft,
        "export_revenue_ft": export_rev,
        "brt_bill_ft": brt_bill,
    }


    # stored thermal energy proxy at end (kWh)
    if hss_active:
        e_hss = float(vol_hss_water) * c_hss * (res["t_hss"] - float(T_in))
        res["e_hss_stor"] = e_hss
        res["final_hss_energy_kwh"] = float(e_hss[-1]) if len(e_hss) else 0.0
    else:
        res["e_hss_stor"] = np.zeros(T)
        res["final_hss_energy_kwh"] = 0.0

    return res