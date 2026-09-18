import numpy as np
import pulp
from Optimization import extract_vector, solve_problem, variable_value
from Utility.configuration import config
from OptimizedIndividualScenarios.boiler_constraints import (
    add_boiler_step, add_boiler_schedule, WATER_HEAT_CAPACITY_KWH_PER_L_K,
)
from Optimization.bess_constraints import build_bess
from OptimizedIndividualScenarios.optimization_constraints import (
    add_import_export_exclusivity,
    add_two_tier_step_constraints,
    create_two_tier_tariff_block,
)


def optimize_household(
    p_pv,                   # (T,)
    p_ue,                   # (T,)
    p_dhw,                  # (T,) hőigény kW
    dt=config.getfloat("simulation", "dt_hours"),
    size_elh=0.0,           # kW
    vol_hss_water=0.0,      # liter
    T_env=20.0,
    T_max=65.0,
    T_min=38.0,
    T_in=10.0,
    a_hss=0.01275,
    eta_elh=0.95,
    p_el_heater_fixed=None,
    price_grid_a_low=config.getfloat("tariffs", "grid_a_low_ft_per_kwh"),
    price_grid_a_high=config.getfloat("tariffs", "grid_a_high_ft_per_kwh"),
    price_grid_b_low=config.getfloat("tariffs", "grid_b_low_ft_per_kwh"),
    price_grid_b_high=config.getfloat("tariffs", "grid_b_high_ft_per_kwh"),
    price_pv_grid=config.getfloat("tariffs", "pv_export_ft_per_kwh"),
    grid_a_low_cap_kwh=config.getfloat("tariffs", "grid_a_low_limit_kwh"),
    grid_b_low_cap_kwh=config.getfloat("tariffs", "grid_b_low_limit_kwh"),
    run_lp=False,
    msg=True,
    enforce_cl_rules=True,
    cl_max_on_hours_per_day=8.0,
    cl_min_midday_hours_per_day=4.0,
    gapRel=None,
    timeLimit=None,
    boiler_tariff="B",
    objective="bill",
    include_boiler=True, include_bess=False, include_heat_pump=False,
    size_bess=0.0, eta_bess_in=0.98, eta_bess_out=0.96,
    eta_bess_stor=0.995, soc_bess_min=0.1, soc_bess_max=0.9,
    soc_bess_init=0.5, t_bess_min=2.0,
    bess_terminal="cyclic", bess_grid_charge=None,
    allow_grid_charge_for_min_soc=False,
    bess_min_mode_steps=4, fixed_pv_priority=False,
):
    """Optimize one household in kW, with energy results in kWh.
    
    Boiler and BESS components share one electrical balance. HP flexibility is
    deliberately rejected until a thermal HP component is available.
    """
    if include_heat_pump:
        raise NotImplementedError("Flexible heat-pump optimization is not implemented")
    if objective not in {"bill", "grid"}:
        raise ValueError("objective must be 'bill' or 'grid'")
    if dt <= 0:
        raise ValueError("dt must be positive")


    p_pv = np.asarray(p_pv, dtype=float).ravel()
    p_ue = np.asarray(p_ue, dtype=float).ravel()
    p_dhw = np.asarray(p_dhw, dtype=float).ravel()

    if not (p_pv.shape == p_ue.shape == p_dhw.shape):
        raise ValueError("p_pv, p_ue and p_dhw must have matching lengths")
    if not len(p_pv) or any(not np.all(np.isfinite(x)) or np.any(x < 0)
                               for x in (p_pv, p_ue, p_dhw)):
        raise ValueError("Profiles must be nonempty, finite and nonnegative")
    T = len(p_pv)
    time_set = range(T)

    if p_el_heater_fixed is None:
        p_el_heater_fixed = np.zeros_like(p_ue)
    else:
        p_el_heater_fixed = np.asarray(p_el_heater_fixed, dtype=float).ravel()

    if (p_el_heater_fixed.shape != p_ue.shape
            or not np.all(np.isfinite(p_el_heater_fixed))
            or np.any(p_el_heater_fixed < 0)):
        raise ValueError("Fixed boiler profile must match the horizon and be finite and nonnegative")

    if not include_boiler:
        size_elh = vol_hss_water = 0.0
        p_dhw = np.zeros(T)
    has_pv = np.sum(p_pv) > 1e-9
    hss_active = has_pv and (size_elh > 1e-9) and (vol_hss_water > 1e-9)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    c_hss = WATER_HEAT_CAPACITY_KWH_PER_L_K

    prob = pulp.LpProblem("individual_boiler_grid_min", pulp.LpMinimize)

    bess = build_bess(prob, T, dt=dt, size=size_bess if include_bess else 0.0,
        eta_in=eta_bess_in, eta_out=eta_bess_out, retention=eta_bess_stor,
        soc_min=soc_bess_min, soc_max=soc_bess_max,
        soc_init=soc_bess_min if soc_bess_init is None else soc_bess_init,
        min_hours=t_bess_min, run_lp=run_lp, terminal=bess_terminal,
        grid_charge=bess_grid_charge, min_mode_steps=bess_min_mode_steps,
        allow_grid_charge_for_min_soc=allow_grid_charge_for_min_soc)

    # Villamos teljesítményáramok
    p_pv_load = [pulp.LpVariable(f"p_pv_load_{t}", lowBound=0) for t in time_set]
    p_pv_elh  = [pulp.LpVariable(f"p_pv_elh_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    p_pv_elh_fixed = [pulp.LpVariable(f"p_pv_elh_fixed_{t}", lowBound=0) for t in time_set]
    p_grid_elh_fixed = [pulp.LpVariable(f"p_grid_elh_fixed_{t}", lowBound=0) for t in time_set]

    p_grid_load = [pulp.LpVariable(f"p_grid_load_{t}", lowBound=0) for t in time_set]
    p_grid_elh  = [pulp.LpVariable(f"p_grid_elh_{t}", lowBound=0) for t in time_set]

    d_export = [pulp.LpVariable(f"d_export_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    M_grid = float(np.max(p_ue + p_el_heater_fixed)) + size_elh + bess.power + 1.0
    M_pv = float(np.max(p_pv)) + 1.0

    # HSS / ELH
    if hss_active:
        p_elh_in = [pulp.LpVariable(f"p_elh_in_{t}", lowBound=0, upBound=size_elh) for t in time_set]
        p_hss_in = [pulp.LpVariable(f"p_hss_in_{t}", lowBound=0) for t in time_set]
        p_hss_out = [pulp.LpVariable(f"p_hss_out_{t}", lowBound=0) for t in time_set]
        t_hss = [pulp.LpVariable(f"t_hss_{t}", lowBound=T_min, upBound=T_max) for t in time_set]
        d_cl = [pulp.LpVariable(f"d_cl_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    else:
        p_elh_in = [0.0] * T
        p_hss_in = [0.0] * T
        p_hss_out = [0.0] * T
        t_hss = [0.0] * T
        d_cl = [0.0] * T

    # Tarifa blokk
    # 15 perces bruttó elszámolás
    tariff_a = create_two_tier_tariff_block(T, grid_a_low_cap_kwh, "e_grid_a")
    tariff_b = create_two_tier_tariff_block(T, grid_b_low_cap_kwh, "e_grid_b")
    e_grid_a_low_step, e_grid_a_high_step, rem_a_low = (
        tariff_a.low_step, tariff_a.high_step, tariff_a.remaining_low
    )
    e_grid_b_low_step, e_grid_b_high_step, rem_b_low = (
        tariff_b.low_step, tariff_b.high_step, tariff_b.remaining_low
    )

    for t in time_set:
        # PV szétosztás
        prob += (
                p_pv_load[t]
                + p_pv_elh[t]
                + p_pv_elh_fixed[t]
                + p_pv_grid[t] + bess.pv[t]
                == p_pv[t]
        ), f"pv_split_{t}"

        if boiler_tariff == "B":
            prob += bess.boiler[t] == 0, f"no_bess_boiler_B_{t}"
            prob += bess.fixed[t] == 0, f"no_bess_fixed_B_{t}"
            # B tarifás bojler: külön mérő/kör, nem kaphat PV-ből.
            prob += p_pv_elh[t] == 0, f"no_pv_to_hss_B_{t}"
            prob += p_pv_elh_fixed[t] == 0, f"no_pv_to_fixed_boiler_B_{t}"

        prob += (
                p_pv_elh_fixed[t] + p_grid_elh_fixed[t] + bess.fixed[t] == p_el_heater_fixed[t]
        ), f"fixed_boiler_balance_{t}"

        # Hagyományos villamos fogyasztás
        prob += p_pv_load[t] + p_grid_load[t] + bess.base[t] == p_ue[t], f"ue_balance_{t}"

        if fixed_pv_priority:
            base_pv = min(p_pv[t], p_ue[t])
            prob += p_pv_load[t] == base_pv, f"fixed_pv_base_{t}"
            if boiler_tariff == "A":
                prob += p_pv_elh_fixed[t] == min(max(p_pv[t]-base_pv, 0), p_el_heater_fixed[t]), f"fixed_pv_boiler_{t}"
        if not hss_active:
            prob += bess.boiler[t] == 0, f"no_bess_hss_{t}"
        if not run_lp:
            if boiler_tariff == "A":
                p_grid_import_a_t = p_grid_load[t] + p_grid_elh[t] + p_grid_elh_fixed[t] + bess.grid[t]
            else:
                p_grid_import_a_t = p_grid_load[t] + bess.grid[t]

            add_import_export_exclusivity(
                prob,
                p_grid_import_a_t,
                p_pv_grid[t],
                d_export[t],
                M_grid,
                M_pv,
                t,
                selector_is_import=False,
            )


        add_boiler_step(prob, t=t, T=T, dt=dt, hss_active=hss_active,
            p_elh_in=p_elh_in, p_hss_in=p_hss_in, p_hss_out=p_hss_out,
            t_hss=t_hss, d_cl=d_cl, p_pv_elh=p_pv_elh, p_grid_elh=p_grid_elh,
            p_bess_elh=bess.boiler, p_dhw=p_dhw, size_elh=size_elh,
            vol_hss_water=vol_hss_water, T_env=T_env, T_max=T_max, T_in=T_in,
            a_hss=a_hss, eta_elh=eta_elh, run_lp=run_lp)

        # 15 perces import felosztása kedvezményes és piaci részre
        if boiler_tariff == "A":
            # A tarifás bojler: UE + bojler hálózati része is A tarifán van.
            e_imp_a_t = dt * (p_grid_load[t] + p_grid_elh[t] + p_grid_elh_fixed[t] + bess.grid[t])
            e_imp_b_t = 0.0
        else:
            # B tarifás bojler: bojler csak hálózatból, B tarifán.
            e_imp_a_t = dt * (p_grid_load[t] + bess.grid[t])
            e_imp_b_t = dt * (p_grid_elh[t] + p_grid_elh_fixed[t])

        add_two_tier_step_constraints(
            prob, tariff_a, t, e_imp_a_t, grid_a_low_cap_kwh, "grid_a"
        )
        add_two_tier_step_constraints(
            prob, tariff_b, t, e_imp_b_t, grid_b_low_cap_kwh, "grid_b"
        )

    add_boiler_schedule(prob, d_cl=d_cl, T=T, dt=dt, hss_active=hss_active,
        enforce_cl_rules=enforce_cl_rules, run_lp=run_lp,
        cl_max_on_hours_per_day=cl_max_on_hours_per_day,
        cl_min_midday_hours_per_day=cl_min_midday_hours_per_day)

    if objective == "bill":
        prob += pulp.lpSum(
            price_grid_a_low * e_grid_a_low_step[t]
            + price_grid_a_high * e_grid_a_high_step[t]
            + price_grid_b_low * e_grid_b_low_step[t]
            + price_grid_b_high * e_grid_b_high_step[t]
            - price_pv_grid * dt * p_pv_grid[t]
            for t in time_set
        )
    else:
        prob += dt*pulp.lpSum(p_grid_load[t]+p_grid_elh[t]+p_grid_elh_fixed[t]
                              +bess.grid[t]+p_pv_grid[t] for t in time_set)

    solve_result = solve_problem(
        prob, msg=msg, gap_rel=gapRel, time_limit=timeLimit
    )
    status_str = solve_result.status

    p_pv_load_v = extract_vector(p_pv_load)
    p_pv_elh_v = extract_vector(p_pv_elh)
    p_pv_grid_v = extract_vector(p_pv_grid)
    p_grid_load_v = extract_vector(p_grid_load)
    p_grid_elh_v = extract_vector(p_grid_elh)
    p_pv_elh_fixed_v = extract_vector(p_pv_elh_fixed)
    p_grid_elh_fixed_v = extract_vector(p_grid_elh_fixed)

    # Explicit grid import/export idősorok, hogy ugyanúgy legyen, mint a BESS modellben
    p_grid_import_v = p_grid_load_v + p_grid_elh_v + p_grid_elh_fixed_v + extract_vector(bess.grid)
    p_grid_export_v = p_pv_grid_v

    if hss_active:
        p_elh_in_v = extract_vector(p_elh_in)
        p_hss_in_v = extract_vector(p_hss_in)
        p_hss_out_v = extract_vector(p_hss_out)
        t_hss_v = extract_vector(t_hss)
        e_hss_stor_v = vol_hss_water * c_hss * (t_hss_v - T_in)
    else:
        p_elh_in_v = np.zeros(T)
        p_hss_in_v = np.zeros(T)
        p_hss_out_v = np.zeros(T)
        t_hss_v = np.zeros(T)
        e_hss_stor_v = np.zeros(T)

    if hss_active and not run_lp:
        d_cl_v = extract_vector(d_cl)
    else:
        d_cl_v = np.zeros(T)

    if not run_lp:
        d_export_v = extract_vector(d_export)
    else:
        d_export_v = np.zeros(T)

    e_grid_a_low_step_v = extract_vector(e_grid_a_low_step)
    e_grid_a_high_step_v = extract_vector(e_grid_a_high_step)
    e_grid_b_low_step_v = extract_vector(e_grid_b_low_step)
    e_grid_b_high_step_v = extract_vector(e_grid_b_high_step)

    rem_a_low_v = extract_vector(rem_a_low)
    rem_b_low_v = extract_vector(rem_b_low)

    if objective == "grid":
        # Tariff variables are not minimized by an energy objective. Recompute
        # the bill chronologically from solved imports instead of arbitrary
        # low/high assignments returned by the solver.
        a_energy = dt*(p_grid_load_v+extract_vector(bess.grid))
        b_energy = dt*(p_grid_elh_v+p_grid_elh_fixed_v)
        if boiler_tariff == "A":
            a_energy = a_energy+b_energy
            b_energy = np.zeros(T)
        def tariff_steps(energy, cap):
            cumulative = np.cumsum(energy)
            low = np.minimum(energy, np.maximum(cap-(cumulative-energy), 0))
            return low, energy-low, np.maximum(cap-np.cumsum(low), 0)
        e_grid_a_low_step_v, e_grid_a_high_step_v, rem_a_low_v = tariff_steps(a_energy, grid_a_low_cap_kwh)
        e_grid_b_low_step_v, e_grid_b_high_step_v, rem_b_low_v = tariff_steps(b_energy, grid_b_low_cap_kwh)

    e_grid_a_low_v = float(np.sum(e_grid_a_low_step_v))
    e_grid_a_high_v = float(np.sum(e_grid_a_high_step_v))
    e_grid_b_low_v = float(np.sum(e_grid_b_low_step_v))
    e_grid_b_high_v = float(np.sum(e_grid_b_high_step_v))

    import_cost_a_ft = float(
        price_grid_a_low * e_grid_a_low_v
        + price_grid_a_high * e_grid_a_high_v
    )

    import_cost_b_ft = float(
        price_grid_b_low * e_grid_b_low_v
        + price_grid_b_high * e_grid_b_high_v
    )

    e_grid_total_v = e_grid_a_low_v + e_grid_a_high_v + e_grid_b_low_v + e_grid_b_high_v
    e_grid_export_v = float(np.sum(p_pv_grid_v) * dt)

    export_revenue = float(price_pv_grid * e_grid_export_v)
    net_cost = import_cost_a_ft + import_cost_b_ft - export_revenue

    if boiler_tariff == "A":
        p_grid_to_base = p_grid_load_v + p_grid_elh_v + p_grid_elh_fixed_v
        p_grid_to_boiler = np.zeros(T, dtype=float)
    else:
        p_grid_to_base = p_grid_load_v
        p_grid_to_boiler = p_grid_elh_v + p_grid_elh_fixed_v

    p_el_heater_total = p_elh_in_v + p_el_heater_fixed

    results = {
        "p_pv_load": p_pv_load_v,
        "p_pv_elh": p_pv_elh_v,
        "p_pv_grid": p_pv_grid_v,
        "p_grid_load": p_grid_load_v,
        "p_grid_elh": p_grid_elh_v,
        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "p_elh_in": p_elh_in_v,
        "p_hss_in": p_hss_in_v,
        "p_hss_out": p_hss_out_v,
        "t_hss": t_hss_v,
        "e_hss_stor": e_hss_stor_v,
        "grid_export_revenue_Ft": export_revenue,
        "net_cost_Ft": net_cost,
        "objective_value": variable_value(prob.objective),
        "objective_type": objective,
        "status": status_str,
        "solver": solve_result.solver,
        "hss_active": int(hss_active),
        "d_cl": d_cl_v,
        "d_export": d_export_v,
        "boiler_tariff": boiler_tariff,

        "e_grid_total": e_grid_total_v,
        "e_grid_export": e_grid_export_v,

        "grid_import_a_kwh": e_grid_a_low_v + e_grid_a_high_v,
        "grid_import_b_kwh": float(np.sum(p_grid_to_boiler) * dt),

        "grid_import_low_kwh": e_grid_a_low_v + e_grid_b_low_v,
        "grid_import_high_kwh": e_grid_a_high_v + e_grid_b_high_v,

        "grid_import_a_low_kwh": e_grid_a_low_v,
        "grid_import_a_high_kwh": e_grid_a_high_v,
        "grid_import_b_low_kwh": e_grid_b_low_v,
        "grid_import_b_high_kwh": e_grid_b_high_v,

        "import_cost_a_ft": import_cost_a_ft,
        "import_cost_b_ft": import_cost_b_ft,
        "import_cost_ft": import_cost_a_ft + import_cost_b_ft,
        "export_revenue_ft": export_revenue,
        "brt_bill_ft": net_cost,

        "e_grid_a_low_step": e_grid_a_low_step_v,
        "e_grid_a_high_step": e_grid_a_high_step_v,
        "e_grid_b_low_step": e_grid_b_low_step_v,
        "e_grid_b_high_step": e_grid_b_high_step_v,
        "remaining_a_low_block_kwh": rem_a_low_v,
        "remaining_b_low_block_kwh": rem_b_low_v,

        "p_pv_elh_fixed": p_pv_elh_fixed_v,
        "p_grid_elh_fixed": p_grid_elh_fixed_v,
        "p_grid_to_base": p_grid_load_v,
        "p_grid_to_boiler": p_grid_elh_v + p_grid_elh_fixed_v,
        "p_el_heater_total": p_elh_in_v + p_el_heater_fixed,
        "p_pv_to_boiler": p_pv_elh_v + p_pv_elh_fixed_v,
    }

    results.update({
        "p_pv": p_pv, "p_ue": p_ue, "p_dhw": p_dhw, "dt": dt,
        "p_total_load": p_ue + p_el_heater_total,
        "p_el_heater": p_el_heater_fixed,
        "p_pv_bess": extract_vector(bess.pv),
        "p_grid_bess": extract_vector(bess.grid),
        "p_bess_load": extract_vector(bess.base),
        "p_bess_ue": extract_vector(bess.base),
        "p_bess_elh": extract_vector(bess.boiler)+extract_vector(bess.fixed),
        "p_bess_to_boiler": extract_vector(bess.boiler)+extract_vector(bess.fixed),
        "p_bess_in": extract_vector(bess.charge),
        "p_bess_out": extract_vector(bess.discharge),
        "e_bess": extract_vector(bess.state[:-1]),
        "e_bess_boundary": extract_vector(bess.state),
        "d_bess_ch": extract_vector(bess.charge_on),
        "d_bess_dis": extract_vector(bess.discharge_on),
        "d_bess": extract_vector(bess.charge_on),
        "d_grid": 1-d_export_v if not run_lp else np.zeros(T),
        "bess_active": int(include_bess and size_bess > 0),
        "size_bess": size_bess if include_bess else 0,
        "battery_power": bess.power, "battery_power_kw": bess.power,
        "soc_bess_min": soc_bess_min, "soc_bess_max": soc_bess_max,
        "soc_bess_init": soc_bess_min if soc_bess_init is None else soc_bess_init,
        "bess_terminal": "cyclic",
        "bess_grid_charge": "min_soc" if allow_grid_charge_for_min_soc else "pv_only",
        "allow_grid_charge_for_min_soc": allow_grid_charge_for_min_soc,
        "grid_import_total_kwh": e_grid_total_v, "grid_export_kwh": e_grid_export_v,
        "p_pv_ue": p_pv_load_v, "p_grid_ue": p_grid_load_v,
        "objective_Ft": net_cost if objective == "bill" else None,
        "objective_unit": "Ft" if objective == "bill" else "kWh",
        "final_hss_energy_kwh": float(e_hss_stor_v[-1]),
        "size_elh_kw": size_elh,
    })
    from Utility.result_schema import add_bess_states, add_bess_flows
    add_bess_states(results, results["e_bess_boundary"])
    add_bess_flows(results, dt)
    return results
