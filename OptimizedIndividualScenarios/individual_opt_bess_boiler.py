"""Egy háztartás egyéni PV–bojler–BESS optimalizálása."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pulp


def individual_opt_bess_boiler(
    p_pv,
    p_ue,
    p_dhw,
    *,
    dt=0.25,
    p_el_heater_fixed=None,
    size_elh=0.0,
    vol_hss_water=0.0,
    T_env=20.0,
    T_max=65.0,
    T_min=35.0,
    T_in=10.0,
    T_set=50.0,
    a_hss=0.01275,
    eta_elh=0.95,
    enforce_cl_rules=True,
    cl_min_activation_hours=2.0,
    cl_max_on_hours_per_day=12.0,
    cl_min_midday_hours_per_day=4.0,
    size_bess=0.0,
    eta_bess_in=0.98,
    eta_bess_out=0.96,
    eta_bess_stor=0.995,
    soc_bess_min=0.10,
    soc_bess_max=0.90,
    soc_bess_init=0.50,
    t_bess_min=2.0,
    bess_min_activation_hours=1.0,
    price_grid_a_low=36.0,
    price_grid_a_high=71.0,
    price_grid_b_low=23.0,
    price_grid_b_high=61.0,
    price_pv_grid=5.0,
    grid_a_low_cap_kwh=2523.0,
    grid_b_low_cap_kwh=2523.0,
    run_lp=False,
    msg=False,
    gapRel=0.005,
    timeLimit=None,
    solver="gurobi",
    debug_lp_path=None,
):
    """Minimalizálja az éves villanyszámlát bruttó elszámolás mellett.

    A dinamikusan optimalizált HSS A tarifás, ezért saját PV-ből és
    BESS-ből is ellátható. A változatlan mért bojlerprofil automatikusan
    B tarifás, és csak a külön B hálózati körről kap energiát.
    """
    p_pv = np.maximum(np.asarray(p_pv, dtype=float).ravel(), 0.0)
    p_ue = np.maximum(np.asarray(p_ue, dtype=float).ravel(), 0.0)
    p_dhw = np.maximum(np.asarray(p_dhw, dtype=float).ravel(), 0.0)
    if p_el_heater_fixed is None:
        p_el_heater_fixed = np.zeros_like(p_ue)
    p_el_heater_fixed = np.maximum(
        np.asarray(p_el_heater_fixed, dtype=float).ravel(), 0.0
    )
    if not (p_pv.shape == p_ue.shape == p_dhw.shape == p_el_heater_fixed.shape):
        raise ValueError("Minden bemeneti idősornak azonos hosszúságúnak kell lennie.")
    if dt <= 0 or len(p_pv) == 0:
        raise ValueError("A dt legyen pozitív, az idősor pedig nem lehet üres.")

    T = len(p_pv)
    steps = range(T)
    c_hss = 0.00116667  # kWh/(liter·°C)
    hss_active = bool(size_elh > 1e-9 and vol_hss_water > 1e-9)
    fixed_boiler_active = bool(not hss_active and p_el_heater_fixed.sum() > 1e-9)
    bess_active = bool(size_bess > 1e-9)

    if hss_active and not (T_min <= T_set <= T_max):
        raise ValueError("A HSS-nél T_min <= T_set <= T_max szükséges.")
    if bess_active:
        if not (0 <= soc_bess_min <= soc_bess_init <= soc_bess_max <= 1):
            raise ValueError("Hibás BESS SOC-határok vagy kezdeti SOC.")
        if min(eta_bess_in, eta_bess_out, eta_bess_stor, t_bess_min) <= 0:
            raise ValueError("A BESS hatásfokai és t_bess_min legyenek pozitívak.")

    model = pulp.LpProblem("individual_opt_bess_boiler", pulp.LpMinimize)
    binary = pulp.LpContinuous if run_lp else pulp.LpBinary
    battery_power = size_bess / t_bess_min if bess_active else 0.0
    max_boiler = max(float(size_elh), float(np.max(p_el_heater_fixed)), 0.0)
    m_grid = float(np.max(p_ue) + max_boiler + battery_power + 1.0)
    m_pv = float(np.max(p_pv) + 1.0)

    def vars_(prefix, upper=None, count=T):
        return [pulp.LpVariable(f"{prefix}_{t}", lowBound=0, upBound=upper) for t in range(count)]

    pv_base, pv_boiler, pv_bess, pv_export = (
        vars_("pv_base"), vars_("pv_boiler"), vars_("pv_bess"), vars_("pv_export")
    )
    grid_base, grid_boiler, grid_bess = vars_("grid_base"), vars_("grid_boiler"), vars_("grid_bess")
    bess_base, bess_boiler = vars_("bess_base"), vars_("bess_boiler")
    grid_import = vars_("grid_import")
    d_export = [pulp.LpVariable(f"d_export_{t}", 0, 1, cat=binary) for t in steps]

    if hss_active:
        p_elh = vars_("p_elh", float(size_elh))
        p_hss_in = vars_("p_hss_in")
        p_hss_out = vars_("p_hss_out")
        t_hss = [pulp.LpVariable(f"t_hss_{t}", T_min, T_max) for t in range(T + 1)]
        d_cl = [pulp.LpVariable(f"d_cl_{t}", 0, 1, cat=binary) for t in steps]
        d_cl_start = [pulp.LpVariable(f"d_cl_start_{t}", 0, 1, cat=binary) for t in steps]
    else:
        p_elh = p_hss_in = p_hss_out = [0.0] * T
        t_hss = [0.0] * (T + 1)
        d_cl = d_cl_start = [0.0] * T

    if bess_active:
        e_min, e_max = size_bess * soc_bess_min, size_bess * soc_bess_max
        e_bess = [pulp.LpVariable(f"e_bess_{t}", e_min, e_max) for t in range(T + 1)]
        e_pre = [pulp.LpVariable(f"e_bess_pre_{t}", lowBound=0, upBound=e_max) for t in steps]
        d_charge = [pulp.LpVariable(f"d_charge_{t}", 0, 1, cat=binary) for t in steps]
        d_discharge = [pulp.LpVariable(f"d_discharge_{t}", 0, 1, cat=binary) for t in steps]
        d_charge_start = [pulp.LpVariable(f"d_charge_start_{t}", 0, 1, cat=binary) for t in steps]
        d_discharge_start = [pulp.LpVariable(f"d_discharge_start_{t}", 0, 1, cat=binary) for t in steps]
        d_grid_guard = [pulp.LpVariable(f"d_grid_guard_{t}", 0, 1, cat=binary) for t in steps]
    else:
        e_bess = [0.0] * (T + 1)
        e_pre = d_charge = d_discharge = [0.0] * T
        d_charge_start = d_discharge_start = d_grid_guard = [0.0] * T

    for t in steps:
        model += pv_base[t] + pv_boiler[t] + pv_bess[t] + pv_export[t] == p_pv[t]
        model += pv_base[t] + bess_base[t] + grid_base[t] == p_ue[t]

        if hss_active:
            model += pv_boiler[t] + bess_boiler[t] + grid_boiler[t] == p_elh[t]
            model += p_hss_in[t] == eta_elh * p_elh[t]
            model += p_hss_out[t] == p_dhw[t]
            model += vol_hss_water * c_hss * (t_hss[t + 1] - t_hss[t]) == dt * (
                p_hss_in[t] - p_hss_out[t] - a_hss * (t_hss[t] - T_env)
            )
            model += p_hss_out[t] <= vol_hss_water * c_hss * (t_hss[t] - T_in) / dt
            model += p_hss_in[t] <= vol_hss_water * c_hss * (T_max - t_hss[t]) / dt
            model += p_elh[t] <= size_elh * d_cl[t]
        elif fixed_boiler_active:
            # A mért profil változatlanul a külön B tarifás körön marad.
            model += pv_boiler[t] == 0
            model += bess_boiler[t] == 0
            model += grid_boiler[t] == p_el_heater_fixed[t]
        else:
            model += pv_boiler[t] == 0
            model += bess_boiler[t] == 0
            model += grid_boiler[t] == 0

        if bess_active:
            charge = pv_bess[t] + grid_bess[t]
            discharge = bess_base[t] + bess_boiler[t]
            model += charge <= battery_power * d_charge[t]
            model += discharge <= battery_power * d_discharge[t]
            model += d_charge[t] + d_discharge[t] <= 1
            model += e_pre[t] == e_bess[t] * eta_bess_stor + dt * (
                pv_bess[t] * eta_bess_in - discharge / eta_bess_out
            )
            model += e_bess[t + 1] == e_pre[t] + dt * eta_bess_in * grid_bess[t]
            # Hálózati töltés csak az önkisülés miatti SOC-minimum pótlása lehet.
            guard_m = max(float(size_bess), 1.0)
            model += e_pre[t] >= e_min - guard_m * d_grid_guard[t]
            model += e_pre[t] <= e_min + guard_m * (1 - d_grid_guard[t])
            replenishment = dt * eta_bess_in * grid_bess[t]
            model += replenishment >= e_min - e_pre[t]
            model += replenishment <= e_min - e_pre[t] + guard_m * (1 - d_grid_guard[t])
            model += grid_bess[t] <= battery_power * d_grid_guard[t]
        else:
            model += pv_bess[t] == 0
            model += grid_bess[t] == 0
            model += bess_base[t] == 0
            model += bess_boiler[t] == 0

        model += grid_import[t] == grid_base[t] + grid_boiler[t] + grid_bess[t]
        # A PV-export csak az A tarifás iránnyal kizáró. A külön mért
        # B tarifás bojler ugyanabban a lépésben is fogyaszthat.
        grid_import_a_t = (
            grid_base[t] + grid_bess[t] + (grid_boiler[t] if hss_active else 0)
        )
        model += grid_import_a_t <= m_grid * (1 - d_export[t])
        model += pv_export[t] <= m_pv * d_export[t]

    if hss_active:
        model += t_hss[0] == T_set
        model += t_hss[T] >= T_set
    if bess_active:
        model += e_bess[0] == size_bess * soc_bess_init

    def add_start_and_minimum(signal, starts, hours, prefix):
        min_steps = max(1, int(round(hours / dt)))
        model.addConstraint(starts[0] == signal[0], f"{prefix}_start_0")
        for t in range(1, T):
            model.addConstraint(starts[t] >= signal[t] - signal[t - 1], f"{prefix}_start_lo_{t}")
            model.addConstraint(starts[t] <= signal[t], f"{prefix}_start_on_{t}")
            model.addConstraint(starts[t] <= 1 - signal[t - 1], f"{prefix}_start_prev_{t}")
        for t in range(T):
            if t + min_steps <= T:
                model.addConstraint(
                    pulp.lpSum(signal[k] for k in range(t, t + min_steps)) >= min_steps * starts[t],
                    f"{prefix}_minimum_{t}",
                )
            else:
                model.addConstraint(starts[t] == 0, f"{prefix}_no_late_start_{t}")

    if hss_active and enforce_cl_rules:
        add_start_and_minimum(d_cl, d_cl_start, cl_min_activation_hours, "cl")
        day_steps = int(round(24 / dt))
        if abs(day_steps * dt - 24) > 1e-9:
            raise ValueError("A dt-nek maradék nélkül kell osztania a 24 órát.")
        max_steps = int(round(cl_max_on_hours_per_day / dt))
        midday_min = int(round(cl_min_midday_hours_per_day / dt))
        midday_start, midday_stop = int(round(10 / dt)), int(round(16 / dt))
        for start in range(0, T, day_steps):
            if start + day_steps > T:
                continue
            model += pulp.lpSum(d_cl[t] for t in range(start, start + day_steps)) <= max_steps
            model += pulp.lpSum(d_cl[t] for t in range(start + midday_start, start + midday_stop)) >= midday_min

    if bess_active:
        add_start_and_minimum(d_charge, d_charge_start, bess_min_activation_hours, "charge")
        add_start_and_minimum(d_discharge, d_discharge_start, bess_min_activation_hours, "discharge")

    annual_import_a = dt * pulp.lpSum(
        grid_base[t] + grid_bess[t] + (grid_boiler[t] if hss_active else 0)
        for t in steps
    )
    annual_import_b = dt * pulp.lpSum(
        grid_boiler[t] if fixed_boiler_active else 0 for t in steps
    )
    import_a_low = pulp.LpVariable("grid_import_a_low_kwh", 0, grid_a_low_cap_kwh)
    import_a_high = pulp.LpVariable("grid_import_a_high_kwh", 0)
    import_b_low = pulp.LpVariable("grid_import_b_low_kwh", 0, grid_b_low_cap_kwh)
    import_b_high = pulp.LpVariable("grid_import_b_high_kwh", 0)
    model += import_a_low + import_a_high == annual_import_a
    model += import_b_low + import_b_high == annual_import_b
    model += (
        price_grid_a_low * import_a_low
        + price_grid_a_high * import_a_high
        + price_grid_b_low * import_b_low
        + price_grid_b_high * import_b_high
        - price_pv_grid * dt * pulp.lpSum(pv_export)
    )

    if debug_lp_path is not None:
        model.writeLP(str(Path(debug_lp_path)))
    if solver.lower() == "gurobi":
        solver_cmd = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    elif solver.lower() == "cbc":
        solver_cmd = pulp.PULP_CBC_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    else:
        raise ValueError("A solver 'gurobi' vagy 'cbc' lehet.")
    status = model.solve(solver_cmd)
    status_text = pulp.LpStatus.get(status, str(status))
    if status_text not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"Az optimalizálás nem adott érvényes megoldást: {status_text}")

    def values(items):
        return np.asarray([float(pulp.value(v) or 0.0) for v in items], dtype=float)

    result = {
        "status": status_text,
        "objective_value": float(pulp.value(model.objective)),
        "p_pv_base": values(pv_base),
        "p_pv_boiler": values(pv_boiler),
        "p_pv_bess": values(pv_bess),
        "p_pv_export": values(pv_export),
        "p_grid_base": values(grid_base),
        "p_grid_boiler": values(grid_boiler),
        "p_grid_bess": values(grid_bess),
        "p_grid_import": values(grid_import),
        "p_bess_base": values(bess_base),
        "p_bess_boiler": values(bess_boiler),
        "p_bess_charge": values(pv_bess) + values(grid_bess),
        "p_bess_discharge": values(bess_base) + values(bess_boiler),
        "e_bess": values(e_bess),
        "d_bess_charge": values(d_charge),
        "d_bess_discharge": values(d_discharge),
        "p_elh": values(p_elh),
        "p_hss_in": values(p_hss_in),
        "p_hss_out": values(p_hss_out),
        "t_hss": values(t_hss),
        "d_cl": values(d_cl),
        "d_cl_start": values(d_cl_start),
        "hss_active": int(hss_active),
        "bess_active": int(bess_active),
        "battery_power_kw": float(battery_power),
        "boiler_tariff": "A" if hss_active else "B" if fixed_boiler_active else "",
        "grid_import_a_low_kwh": float(pulp.value(import_a_low) or 0.0),
        "grid_import_a_high_kwh": float(pulp.value(import_a_high) or 0.0),
        "grid_import_b_low_kwh": float(pulp.value(import_b_low) or 0.0),
        "grid_import_b_high_kwh": float(pulp.value(import_b_high) or 0.0),
    }
    result["grid_import_a_kwh"] = result["grid_import_a_low_kwh"] + result["grid_import_a_high_kwh"]
    result["grid_import_b_kwh"] = result["grid_import_b_low_kwh"] + result["grid_import_b_high_kwh"]
    result["grid_import_total_kwh"] = result["grid_import_a_kwh"] + result["grid_import_b_kwh"]
    result["grid_export_kwh"] = float(result["p_pv_export"].sum() * dt)
    result["import_cost_a_ft"] = price_grid_a_low * result["grid_import_a_low_kwh"] + price_grid_a_high * result["grid_import_a_high_kwh"]
    result["import_cost_b_ft"] = price_grid_b_low * result["grid_import_b_low_kwh"] + price_grid_b_high * result["grid_import_b_high_kwh"]
    result["import_cost_ft"] = result["import_cost_a_ft"] + result["import_cost_b_ft"]
    result["export_revenue_ft"] = price_pv_grid * result["grid_export_kwh"]
    result["bill_ft"] = result["import_cost_ft"] - result["export_revenue_ft"]
    return result
