"""Egy háztartás egyéni PV–bojler–BESS optimalizálása."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    T_abs_min=10.0,
    T_in=10.0,
    T_comfort=40.0,
    T_setpoint_min=50.0,
    T_initial=50.0,
    morning_block=(6.0, 9.0),
    evening_block=(17.0, 22.0),
    a_hss=0.01275,
    eta_elh=0.95,
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

    A HSS állapota eltárolt hőenergia. A 40 °C-os használati meleg víz
    igénye energiaelvételként jelenik meg, ezért nincs 40 °C-os hard
    átlaghőmérséklet-korlát. A reggeli és esti tiltott sávban a bojler
    sem hálózatból, sem PV-ből, sem BESS-ből nem működhet.
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

    if hss_active and not (
        T_in <= T_abs_min <= T_comfort <= T_setpoint_min <= T_max
        and T_abs_min <= T_initial <= T_max
    ):
        raise ValueError(
            "A HSS-nél T_in <= T_abs_min <= T_comfort <= "
            "T_setpoint_min <= T_max és T_abs_min <= T_initial <= T_max szükséges."
        )
    if bess_active:
        if not (0 <= soc_bess_min <= soc_bess_init <= soc_bess_max <= 1):
            raise ValueError("Hibás BESS SOC-határok vagy kezdeti SOC.")
        if min(eta_bess_in, eta_bess_out, eta_bess_stor, t_bess_min) <= 0:
            raise ValueError("A BESS hatásfokai és t_bess_min legyenek pozitívak.")

    if not hss_active and not bess_active:
        # Nincs döntési változó: saját PV az alapfogyasztást fedezi,
        # a mért bojler továbbra is kizárólag a külön B-körön marad.
        result = {key: np.zeros(T) for key in (
            "p_pv_boiler", "p_pv_bess", "p_grid_bess", "p_bess_base",
            "p_bess_boiler", "p_bess_charge", "p_bess_discharge",
            "d_bess_charge", "d_bess_discharge", "p_elh", "p_hss_in",
            "p_hss_out", "d_cl", "d_cl_start", "d_heat", "e_hss",
            "d_below_comfort", "below_comfort", "heating_blocked",
            "boiler_target_C",
        )}
        result.update({
            "status": "Direct",
            "calculation_mode": "direct",
            "p_pv_base": np.minimum(p_pv, p_ue),
            "p_pv_export": np.maximum(p_pv - p_ue, 0.0),
            "p_grid_base": np.maximum(p_ue - p_pv, 0.0),
            "p_grid_boiler": p_el_heater_fixed.copy(),
            "e_bess": np.zeros(T + 1), "e_hss": np.zeros(T + 1),
            "t_hss": np.zeros(T + 1),
            "hss_active": 0, "bess_active": 0, "battery_power_kw": 0.0,
            "boiler_tariff": "B" if fixed_boiler_active else "",
            "longest_continuous_below_40_h": 0.0,
            "dhw_while_below_40_steps": 0,
            "minimum_dhw_energy_margin_kwhth": 0.0,
            "dhw_energy_shortfall_steps": 0,
            "optimized_setpoint_min_C": 0.0,
            "optimized_setpoint_max_C": 0.0,
            "boiler_control_status": "NOT_APPLICABLE",
        })
        result["p_grid_import"] = result["p_grid_base"] + result["p_grid_boiler"]
        for tariff, flow, cap in (
            ("a", result["p_grid_base"], grid_a_low_cap_kwh),
            ("b", result["p_grid_boiler"], grid_b_low_cap_kwh),
        ):
            energy = float(flow.sum() * dt)
            result[f"grid_import_{tariff}_low_kwh"] = min(energy, cap)
            result[f"grid_import_{tariff}_high_kwh"] = max(energy - cap, 0.0)
        result = _finish_bill(result, dt, price_grid_a_low, price_grid_a_high,
                              price_grid_b_low, price_grid_b_high, price_pv_grid)
        result["objective_value"] = result["bill_ft"]
        print("[INFO] Közvetlen számítás: nincs aktív BESS vagy dinamikus bojler.")
        return result

    import pulp  # Csak az aktív eszközök optimalizálásához szükséges.

    model = pulp.LpProblem("individual_opt_bess_boiler", pulp.LpMinimize)
    binary = pulp.LpContinuous if run_lp else pulp.LpBinary
    battery_power = size_bess / t_bess_min if bess_active else 0.0
    max_boiler = max(float(size_elh), float(np.max(p_el_heater_fixed)), 0.0)
    m_grid = float(np.max(p_ue) + max_boiler + battery_power + 1.0)
    m_pv = float(np.max(p_pv) + 1.0)

    def in_window(hour, window):
        """Félig nyílt napi időablak: [kezdés, befejezés)."""
        start, stop = map(float, window)
        return start <= hour < stop if start <= stop else hour >= start or hour < stop

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
        thermal_capacity = vol_hss_water * c_hss
        e_abs_min = thermal_capacity * (T_abs_min - T_in)
        e_comfort = thermal_capacity * (T_comfort - T_in)
        e_setpoint_min = thermal_capacity * (T_setpoint_min - T_in)
        e_hss_max = thermal_capacity * (T_max - T_in)
        e_initial = thermal_capacity * (T_initial - T_in)
        p_elh = vars_("p_elh", float(size_elh))
        p_hss_in = vars_("p_hss_in")
        p_hss_out = vars_("p_hss_out")
        e_hss = [
            pulp.LpVariable(f"e_hss_{t}", e_abs_min, e_hss_max)
            for t in range(T + 1)
        ]

        day_steps = int(round(24.0 / dt))
        if abs(day_steps * dt - 24.0) > 1e-9:
            raise ValueError("A dt-nek maradék nélkül kell osztania a 24 órát.")
        setpoint_at_step = {}
        setpoint_windows = []
        for day_start in range(0, T, day_steps):
            for label, window in (("morning", morning_block), ("evening", evening_block)):
                start_hour = float(window[0])
                start_offset = int(round(start_hour / dt))
                if abs(start_offset * dt - start_hour) > 1e-9:
                    raise ValueError("A tiltott sávok kezdetének a dt időrácsára kell esnie.")
                start_step = day_start + start_offset
                if start_step < T:
                    setpoint = pulp.LpVariable(
                        f"boiler_setpoint_{label}_{day_start // day_steps}",
                        T_setpoint_min,
                        T_max,
                    )
                    setpoint_at_step[start_step] = setpoint
                    setpoint_windows.append((start_step, window, setpoint))
    else:
        p_elh = p_hss_in = p_hss_out = [0.0] * T
        e_hss = [0.0] * (T + 1)
        setpoint_at_step = {}
        setpoint_windows = []

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
            equivalent_temperature = T_in + e_hss[t] / thermal_capacity
            model += e_hss[t + 1] == e_hss[t] + dt * (
                p_hss_in[t]
                - p_hss_out[t]
                - a_hss * (equivalent_temperature - T_env)
            )
            # A vételezéshez szükséges hőenergiának már a lépés elején a
            # tárolóban kell lennie. Nincs külön 40 °C-os átlaghőmérséklet-korlát.
            model += dt * p_hss_out[t] <= e_hss[t] - e_abs_min
            model += dt * p_hss_in[t] <= e_hss_max - e_hss[t]

            hour = (t * dt) % 24.0
            blocked = in_window(hour, morning_block) or in_window(hour, evening_block)
            if blocked:
                # A bojler a tiltott sávban sem hálózatból, sem saját PV-ből,
                # sem BESS-ből nem kaphat energiát.
                model += p_elh[t] == 0
                model += grid_boiler[t] == 0
                model += pv_boiler[t] == 0
                model += bess_boiler[t] == 0

            if t in setpoint_at_step:
                # Lineáris kapcsolat: E = V*c*(T_setpoint-T_in).
                model += e_hss[t] == thermal_capacity * (
                    setpoint_at_step[t] - T_in
                )
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
        model += e_hss[0] == e_initial
        model += e_hss[T] >= e_setpoint_min
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

    e_hss_values = values(e_hss)
    if hss_active:
        t_hss_values = T_in + e_hss_values / thermal_capacity
        d_cl_values = (values(p_elh) > 1e-6).astype(float)
        d_cl_start_values = np.zeros(T)
        if T:
            d_cl_start_values[0] = d_cl_values[0]
            d_cl_start_values[1:] = np.maximum(
                d_cl_values[1:] - d_cl_values[:-1], 0.0
            )
    else:
        t_hss_values = np.zeros(T + 1)
        d_cl_values = np.zeros(T)
        d_cl_start_values = np.zeros(T)

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
        "e_hss": e_hss_values,
        # Diagnosztikai ekvivalens átlaghőmérséklet; nem vezérlési korlát.
        "t_hss": t_hss_values,
        "d_cl": d_cl_values,
        "d_cl_start": d_cl_start_values,
        "d_below_comfort": np.zeros(T),
        "hss_active": int(hss_active),
        "bess_active": int(bess_active),
        "battery_power_kw": float(battery_power),
        "boiler_tariff": "A" if hss_active else "B" if fixed_boiler_active else "",
        "grid_import_a_low_kwh": float(pulp.value(import_a_low) or 0.0),
        "grid_import_a_high_kwh": float(pulp.value(import_a_high) or 0.0),
        "grid_import_b_low_kwh": float(pulp.value(import_b_low) or 0.0),
        "grid_import_b_high_kwh": float(pulp.value(import_b_high) or 0.0),
    }
    if hss_active:
        temperatures = result["t_hss"][:T]
        below = temperatures < T_comfort - 1e-6
        blocked_mask = np.asarray([
            in_window((t * dt) % 24.0, morning_block)
            or in_window((t * dt) % 24.0, evening_block)
            for t in steps
        ], dtype=float)
        target = np.full(T, T_setpoint_min, dtype=float)
        optimized_setpoints = []
        for start_step, window, setpoint_var in setpoint_windows:
            setpoint_value = float(pulp.value(setpoint_var) or T_setpoint_min)
            optimized_setpoints.append(setpoint_value)
            day_start = (start_step // day_steps) * day_steps
            for t in range(day_start, min(day_start + day_steps, T)):
                if in_window((t * dt) % 24.0, window):
                    target[t] = setpoint_value
        longest_steps = current_steps = 0
        for is_below in below:
            current_steps = current_steps + 1 if is_below else 0
            longest_steps = max(longest_steps, current_steps)
        result["below_comfort"] = below.astype(float)
        result["d_below_comfort"] = below.astype(float)
        result["d_heat"] = (result["p_elh"] > 1e-6).astype(float)
        result["heating_blocked"] = blocked_mask
        result["boiler_target_C"] = target
        result["longest_continuous_below_40_h"] = float(longest_steps * dt)
        below_with_dhw = below & (p_dhw > 1e-9)
        result["dhw_while_below_40_steps"] = int(np.count_nonzero(below_with_dhw))
        dhw_energy_margin = result["e_hss"][:T] - e_abs_min - dt * p_dhw
        result["minimum_dhw_energy_margin_kwhth"] = float(np.min(dhw_energy_margin))
        result["dhw_energy_shortfall_steps"] = int(
            np.count_nonzero(dhw_energy_margin < -1e-6)
        )
        result["optimized_setpoint_min_C"] = (
            float(np.min(optimized_setpoints)) if optimized_setpoints else T_setpoint_min
        )
        result["optimized_setpoint_max_C"] = (
            float(np.max(optimized_setpoints)) if optimized_setpoints else T_setpoint_min
        )
        blocked_boiler_power = (
            result["p_grid_boiler"]
            + result["p_pv_boiler"]
            + result["p_bess_boiler"]
        )
        if float(np.min(result["e_hss"])) < e_abs_min - 1e-5:
            control_status = "FAIL_ABSOLUTE_MIN"
        elif np.any((blocked_boiler_power > 1e-6) & (blocked_mask > 0.5)):
            control_status = "FAIL_BOILER_ENERGY_IN_BLOCKED_WINDOW"
        elif result["dhw_energy_shortfall_steps"]:
            control_status = "FAIL_DHW_ENERGY_SHORTFALL"
        else:
            control_status = "PASS"
        result["boiler_control_status"] = control_status
    else:
        result.update({
            "below_comfort": np.zeros(T),
            "d_heat": np.zeros(T),
            "heating_blocked": np.zeros(T),
            "boiler_target_C": np.zeros(T),
            "longest_continuous_below_40_h": 0.0,
            "dhw_while_below_40_steps": 0,
            "minimum_dhw_energy_margin_kwhth": 0.0,
            "dhw_energy_shortfall_steps": 0,
            "optimized_setpoint_min_C": 0.0,
            "optimized_setpoint_max_C": 0.0,
            "boiler_control_status": "NOT_APPLICABLE",
        })
    result["calculation_mode"] = "optimization"
    return _finish_bill(result, dt, price_grid_a_low, price_grid_a_high,
                        price_grid_b_low, price_grid_b_high, price_pv_grid)


def _finish_bill(result, dt, price_grid_a_low, price_grid_a_high,
                 price_grid_b_low, price_grid_b_high, price_pv_grid):
    """Azonos pénzügyi összesítés a közvetlen és optimalizált ágon."""
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