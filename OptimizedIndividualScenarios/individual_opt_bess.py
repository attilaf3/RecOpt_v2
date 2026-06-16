import numpy as np
import pulp


# Notes: ez van készen
def individual_opt_bess(
    p_pv,                   # (T,) kW
    p_ue,                   # (T,) kW - alap villamos fogyasztás
    p_el_heater=None,       # (T,) kW - FIX bojler villamos fogyasztás (nem optimalizált)
    dt=0.25,                # óra
    size_bess=0.0,          # kWh
    eta_bess_in=0.98,
    eta_bess_out=0.96,
    eta_bess_stor=0.995,
    soc_bess_min=0.10,
    soc_bess_max=0.90,
    soc_bess_init=0.5,
    t_bess_min=2.0,         # h -> max teljesítmény = size_bess / t_bess_min
    price_grid_a_low=36.0,
    price_grid_a_high=71.0,
    price_grid_b_low=23.0,
    price_grid_b_high=61.0,
    price_pv_grid=5.0,
    grid_a_low_cap_kwh=2523.0,
    grid_b_low_cap_kwh=2523.0,
    boiler_tariff="B",
    objective="bill",
    run_lp=False,
    msg=False,
    gapRel=None,
    timeLimit=None,
):
    """
    Egy háztartás egyéni gazdasági optimalizálása, ahol:
      - PV van
      - BESS opcionális
      - közösségi megosztás nincs
      - bojler NEM optimalizált, hanem fix villamos terhelésként szerepel
      - cél: éves nettó villanyszámla minimalizálása

    Energiaáramlási logika:
      PV -> saját load / saját BESS / külső hálózati export
      BESS -> saját load
      Grid -> saját load és saját BESS
    """

    p_pv = np.asarray(p_pv, dtype=float).ravel()
    p_ue = np.asarray(p_ue, dtype=float).ravel()

    if p_el_heater is None:
        p_el_heater = np.zeros_like(p_ue)
    else:
        p_el_heater = np.asarray(p_el_heater, dtype=float).ravel()

    assert p_pv.shape == p_ue.shape == p_el_heater.shape, \
        "p_pv, p_ue, p_el_heater hossza nem egyezik."

    T = len(p_pv)
    time_set = range(T)

    total_load = p_ue + p_el_heater
    boiler_tariff = str(boiler_tariff).upper().strip()

    # ------------------------------------------------------------------
    # Nonopt-tal ekvivalens fix PV -> load sorrend
    # ------------------------------------------------------------------
    if boiler_tariff == "A":
        # A tarifás bojler: a bojler a normál fogyasztási kör része.
        p_dispatch_load = total_load.copy()
    else:
        # B tarifás bojler: a PV/BESS csak az alapfogyasztást látja.
        p_dispatch_load = p_ue.copy()

    # A nonopt logika szerint:
    # PV először a dispatch loadra megy.
    p_pv_to_load_fixed = np.minimum(p_pv, p_dispatch_load)

    # PV többlet, amit vagy BESS-be töltünk, vagy exportálunk.
    p_surplus = np.maximum(p_pv - p_pv_to_load_fixed, 0.0)

    # Fogyasztási hiány, amit vagy BESS-ből, vagy hálózatból látunk el.
    p_deficit = np.maximum(p_dispatch_load - p_pv_to_load_fixed, 0.0)

    # A lokális ellátás riportolásához ugyanazt az elvet használjuk:
    # először alapfogyasztás, utána bojler.
    p_pv_ue_fixed = np.minimum(p_ue, p_pv_to_load_fixed)

    if boiler_tariff == "A":
        p_pv_elh_fixed = np.maximum(p_pv_to_load_fixed - p_pv_ue_fixed, 0.0)
        p_deficit_ue = np.maximum(p_ue - p_pv_ue_fixed, 0.0)
        p_deficit_elh = np.maximum(p_el_heater - p_pv_elh_fixed, 0.0)
    else:
        # B tarifás bojler nem kaphat PV-t.
        p_pv_elh_fixed = np.zeros(T, dtype=float)
        p_deficit_ue = p_deficit.copy()
        p_deficit_elh = p_el_heater.copy()

    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    bess_active = float(size_bess) > 1e-9
    if soc_bess_init is None:
        soc_bess_init = soc_bess_min

    battery_power = float(size_bess) / max(float(t_bess_min), 1e-9) if bess_active else 0.0


    prob = pulp.LpProblem("individual_opt_bess_fixed_boiler", pulp.LpMinimize)

    # Big-M
    M_pv = float(np.max(p_pv)) + 1.0 if T > 0 else 1.0
    M_grid = float(np.max(total_load) + battery_power) + 1.0 if T > 0 else 1.0

    # ------------------------------------------------------------------
    # Változók
    # ------------------------------------------------------------------
    # PV split
    p_pv_ue = [pulp.LpVariable(f"p_pv_ue_{t}", lowBound=0) for t in time_set]
    p_pv_elh = [pulp.LpVariable(f"p_pv_elh_{t}", lowBound=0) for t in time_set]
    p_pv_bess = [pulp.LpVariable(f"p_pv_bess_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    p_bess_ue = [pulp.LpVariable(f"p_bess_ue_{t}", lowBound=0) for t in time_set]
    p_bess_elh = [pulp.LpVariable(f"p_bess_elh_{t}", lowBound=0) for t in time_set]

    p_grid_ue = [pulp.LpVariable(f"p_grid_ue_{t}", lowBound=0) for t in time_set]
    p_grid_elh = [pulp.LpVariable(f"p_grid_elh_{t}", lowBound=0) for t in time_set]

    # Battery
    p_bess_in = [pulp.LpVariable(f"p_bess_in_{t}", lowBound=0) for t in time_set]
    p_bess_out = [pulp.LpVariable(f"p_bess_out_{t}", lowBound=0) for t in time_set]
    p_grid_bess = [pulp.LpVariable(f"p_grid_bess_{t}", lowBound=0) for t in time_set]

    if bess_active:
        e_bess = [
            pulp.LpVariable(
                f"e_bess_{t}",
                lowBound=float(size_bess) * float(soc_bess_min),
                upBound=float(size_bess) * float(soc_bess_max),
            )
            for t in time_set
        ]

        e_bess_pre = [
            pulp.LpVariable(
                f"e_bess_pre_{t}",
                lowBound=0,
                upBound=float(size_bess) * float(soc_bess_max),
            )
            for t in time_set
        ]

        d_grid_bess_guard = [
            pulp.LpVariable(f"d_grid_bess_guard_{t}", cat=pulp.LpBinary) if not run_lp else 0
            for t in time_set
        ]


        d_bess_ch = [
            pulp.LpVariable(f"d_bess_ch_{t}", cat=pulp.LpBinary) if not run_lp else 0
            for t in time_set
        ]

        d_bess_dis = [
            pulp.LpVariable(f"d_bess_dis_{t}", cat=pulp.LpBinary) if not run_lp else 0
            for t in time_set
        ]
    else:
        e_bess = [0.0] * T
        d_bess_ch = [0.0] * T
        d_bess_dis = [0.0] * T
        e_bess_pre = [0.0] * T
        d_grid_bess_guard = [0.0] * T

    # Grid import/export
    p_grid_import = [pulp.LpVariable(f"p_grid_import_{t}", lowBound=0) for t in time_set]
    p_grid_export = [pulp.LpVariable(f"p_grid_export_{t}", lowBound=0) for t in time_set]
    d_grid = [pulp.LpVariable(f"d_grid_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Tariff blocks
    # 15 perces bruttó elszámolás
    e_grid_a_low_step = [pulp.LpVariable(f"e_grid_a_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_a_high_step = [pulp.LpVariable(f"e_grid_a_high_step_{t}", lowBound=0) for t in time_set]
    rem_a_low = [pulp.LpVariable(f"rem_a_low_{t}", lowBound=0, upBound=grid_a_low_cap_kwh) for t in time_set]

    e_grid_b_low_step = [pulp.LpVariable(f"e_grid_b_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_b_high_step = [pulp.LpVariable(f"e_grid_b_high_step_{t}", lowBound=0) for t in time_set]
    rem_b_low = [pulp.LpVariable(f"rem_b_low_{t}", lowBound=0, upBound=grid_b_low_cap_kwh) for t in time_set]

    # ------------------------------------------------------------------
    # Korlátok
    # ------------------------------------------------------------------
    for t in time_set:
        # k = (t + 1) % T

        # ------------------------------------------------------------------
        # Nonopt-tal ekvivalens villamos sorrend
        # ------------------------------------------------------------------

        # PV -> load fixen, nem döntési változóként.
        prob += p_pv_ue[t] == float(p_pv_ue_fixed[t]), f"pv_ue_fixed_{t}"
        prob += p_pv_elh[t] == float(p_pv_elh_fixed[t]), f"pv_elh_fixed_{t}"

        # A maradék PV vagy BESS-be megy, vagy export.
        prob += (
                p_pv_bess[t] + p_pv_grid[t] == float(p_surplus[t])
        ), f"surplus_split_{t}"

        # A PV után fennmaradó alapfogyasztási hiány:
        # vagy BESS-ből, vagy hálózatból.
        prob += (
                p_bess_ue[t] + p_grid_ue[t] == float(p_deficit_ue[t])
        ), f"ue_deficit_split_{t}"

        # A PV után fennmaradó bojlerhiány:
        # A tarifán: BESS vagy grid is mehet rá.
        # B tarifán: BESS nem mehet rá, csak grid.
        prob += (
                p_bess_elh[t] + p_grid_elh[t] == float(p_deficit_elh[t])
        ), f"boiler_deficit_split_{t}"

        if boiler_tariff == "B":
            prob += p_bess_elh[t] == 0, f"no_bess_to_boiler_B_{t}"

        prob += (
            p_grid_export[t] == p_pv_grid[t]
        ), f"grid_export_def_{t}"

        prob += (
                p_grid_import[t] == p_grid_ue[t] + p_grid_elh[t] + p_grid_bess[t]
        ), f"grid_import_def_{t}"

        # Ne legyen egyszerre A-körös import és PV export.
        # B tarifán a bojler külön körön importálhat, de az A kör
        # akkor sem importálhat és exportálhat egyszerre.
        if not run_lp:
            if boiler_tariff == "A":
                p_grid_import_a_t = p_grid_ue[t] + p_grid_elh[t] + p_grid_bess[t]
            else:
                p_grid_import_a_t = p_grid_ue[t] + p_grid_bess[t]

            prob += p_grid_import_a_t <= M_grid * d_grid[t], f"grid_import_a_gate_{t}"
            prob += p_grid_export[t] <= M_pv * (1 - d_grid[t]), f"grid_export_gate_{t}"

        # Akku logika
        if not bess_active:
            prob += p_pv_bess[t] == 0, f"no_bess_pv_bess_{t}"
            prob += p_bess_ue[t] == 0, f"no_bess_ue_{t}"
            prob += p_bess_elh[t] == 0, f"no_bess_elh_{t}"
            prob += p_bess_in[t] == 0, f"no_bess_in_{t}"
            prob += p_bess_out[t] == 0, f"no_bess_out_{t}"
            prob += p_grid_bess[t] == 0, f"no_bess_grid_bess_{t}"
        else:
            # BESS tölthet PV-ből és hálózatból is.
            # A hálózati BESS töltés A tarifás import.
            prob += (
                    p_bess_in[t] == p_pv_bess[t] + p_grid_bess[t]
            ), f"bess_in_def_{t}"

            # kisütés csak a loadra

            prob += (
                    p_bess_out[t] == p_bess_ue[t] + p_bess_elh[t]
            ), f"bess_out_def_{t}"


            # dinamika
            # prob += e_bess[k] == (
            #         e_bess[t] * eta_bess_stor
            #         + dt * (p_bess_in[t] * eta_bess_in - p_bess_out[t] / max(eta_bess_out, 1e-9))
            # )

            if t < T - 1:
                soc_min_abs = float(size_bess) * float(soc_bess_min)
                M_soc = float(size_bess)

                # SOC grid->BESS pótlás nélkül.
                # Ez ugyanaz, mint a nonopt-ban: először önkisülés,
                # PV-töltés és kisütés hatása, de még hálózati pótlás nélkül.
                prob += e_bess_pre[t + 1] == (
                        e_bess[t] * eta_bess_stor
                        + dt * (
                                p_pv_bess[t] * eta_bess_in
                                - p_bess_out[t] / max(eta_bess_out, 1e-9)
                        )
                ), f"bess_pre_dyn_{t}"

                # A tényleges SOC már tartalmazhat grid->BESS minimumszint-pótlást.
                prob += e_bess[t + 1] == (
                        e_bess_pre[t + 1]
                        + dt * eta_bess_in * p_grid_bess[t]
                ), f"bess_dyn_with_grid_guard_{t}"

                if not run_lp:
                    grid_added = dt * eta_bess_in * p_grid_bess[t]
                    deficit_to_min = soc_min_abs - e_bess_pre[t + 1]

                    # Ha e_bess_pre < SOC_min, akkor grid_added pontosan a hiány.
                    # Ha e_bess_pre >= SOC_min, akkor grid_added = 0.
                    prob += grid_added >= deficit_to_min, f"grid_bess_guard_lb_{t}"
                    prob += (
                            grid_added <= deficit_to_min + M_soc * (1 - d_grid_bess_guard[t])
                    ), f"grid_bess_guard_exact_{t}"
                    prob += grid_added <= M_soc * d_grid_bess_guard[t], f"grid_bess_guard_ub_{t}"

                else:
                    # LP-relaxáció esetén kevésbé szigorú, de MIP futásnál a fenti pontos.
                    prob += (
                            dt * eta_bess_in * p_grid_bess[t]
                            >= soc_min_abs - e_bess_pre[t + 1]
                    ), f"grid_bess_guard_lp_{t}"

            else:
                # Utolsó időlépésben nincs következő SOC, ezért itt ne töltsön hálózatból.
                prob += p_grid_bess[t] == 0, f"no_grid_bess_last_step_{t}"

            # teljesítménykorlát
            if run_lp:
                prob += p_bess_in[t] <= battery_power, f"bess_in_cap_{t}"
                prob += p_bess_out[t] <= battery_power, f"bess_out_cap_{t}"
            else:
                prob += d_bess_ch[t] + d_bess_dis[t] <= 1, f"bess_no_simultaneous_{t}"

                prob += p_bess_in[t] <= d_bess_ch[t] * battery_power, f"bess_in_gate_{t}"
                prob += p_bess_out[t] <= d_bess_dis[t] * battery_power, f"bess_out_gate_{t}"

            # SOC korlát külön is
            prob += e_bess[t] >= float(size_bess) * float(soc_bess_min), f"soc_min_{t}"
            prob += e_bess[t] <= float(size_bess) * float(soc_bess_max), f"soc_max_{t}"

        # Bojler tarifalogika
        if boiler_tariff == "A":
            # A tarifás bojler:
            # - kaphat PV-ből
            # - kaphat BESS-ből
            # - hálózatból A tarifán vételez
            e_imp_a_t = dt * (p_grid_ue[t] + p_grid_elh[t] + p_grid_bess[t])
            e_imp_b_t = 0.0


        else:
            # B tarifás bojler:
            # - külön mérő / külön áramkör
            # - nem kaphat PV-ből
            # - nem kaphat BESS-ből
            # - teljes bojlerigény B tarifás hálózati import

            e_imp_a_t = dt * (p_grid_ue[t] + p_grid_bess[t])
            e_imp_b_t = dt * p_grid_elh[t]

        prob += e_grid_a_low_step[t] + e_grid_a_high_step[t] == e_imp_a_t
        prob += e_grid_b_low_step[t] + e_grid_b_high_step[t] == e_imp_b_t

        if t == 0:
            prob += rem_a_low[t] == grid_a_low_cap_kwh - e_grid_a_low_step[t]
            prob += e_grid_a_low_step[t] <= grid_a_low_cap_kwh

            prob += rem_b_low[t] == grid_b_low_cap_kwh - e_grid_b_low_step[t]
            prob += e_grid_b_low_step[t] <= grid_b_low_cap_kwh
        else:
            prob += rem_a_low[t] == rem_a_low[t - 1] - e_grid_a_low_step[t]
            prob += e_grid_a_low_step[t] <= rem_a_low[t - 1]

            prob += rem_b_low[t] == rem_b_low[t - 1] - e_grid_b_low_step[t]
            prob += e_grid_b_low_step[t] <= rem_b_low[t - 1]

    # kezdeti SOC
    if bess_active:
        prob += e_bess[0] == float(size_bess) * float(soc_bess_init), "bess_init"

    if bess_active and not run_lp:
        min_mode_steps = 4  # 4 * 15 perc = 1 óra

        for t in range(1, T - min_mode_steps + 1):
            start_ch = d_bess_ch[t] - d_bess_ch[t - 1]
            start_dis = d_bess_dis[t] - d_bess_dis[t - 1]

            prob += (
                    pulp.lpSum(d_bess_ch[tau] for tau in range(t, t + min_mode_steps))
                    >= min_mode_steps * start_ch
            ), f"bess_min_charge_{t}"

            prob += (
                    pulp.lpSum(d_bess_dis[tau] for tau in range(t, t + min_mode_steps))
                    >= min_mode_steps * start_dis
            ), f"bess_min_discharge_{t}"

    # ------------------------------------------------------------------
    # Célfüggvény: hálózati interakció [Ft]
    if objective == "grid":
        prob += pulp.lpSum(
            dt * (p_grid_import[t] + p_grid_export[t])
            for t in time_set
        )
    # Célfüggvény: éves bruttó villanyszámla [Ft]
    elif objective == "bill":
        prob += pulp.lpSum(
            price_grid_a_low * e_grid_a_low_step[t]
            + price_grid_a_high * e_grid_a_high_step[t]
            + price_grid_b_low * e_grid_b_low_step[t]
            + price_grid_b_high * e_grid_b_high_step[t]
            - price_pv_grid * dt * p_grid_export[t]
            for t in time_set
        )


    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)

    status_str = pulp.LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"Hiba: {status_str}")

    def _val(x):
        v = pulp.value(x)
        return 0.0 if v is None else float(v)

    p_pv_ue_v = np.array([_val(v) for v in p_pv_ue], dtype=float)
    p_pv_elh_v = np.array([_val(v) for v in p_pv_elh], dtype=float)
    p_pv_bess_v = np.array([_val(v) for v in p_pv_bess], dtype=float)
    p_pv_grid_v = np.array([_val(v) for v in p_pv_grid], dtype=float)

    p_bess_ue_v = np.array([_val(v) for v in p_bess_ue], dtype=float)
    p_bess_elh_v = np.array([_val(v) for v in p_bess_elh], dtype=float)
    p_grid_ue_v = np.array([_val(v) for v in p_grid_ue], dtype=float)
    p_grid_elh_v = np.array([_val(v) for v in p_grid_elh], dtype=float)

    p_bess_in_v = np.array([_val(v) for v in p_bess_in], dtype=float)
    p_bess_out_v = np.array([_val(v) for v in p_bess_out], dtype=float)
    p_grid_bess_v = np.array([_val(v) for v in p_grid_bess], dtype=float)

    p_grid_import_v = np.array([_val(v) for v in p_grid_import], dtype=float)
    p_grid_export_v = np.array([_val(v) for v in p_grid_export], dtype=float)

    if bess_active:
        e_bess_v = np.array([_val(v) for v in e_bess], dtype=float)
        d_bess_ch_v = np.array([_val(v) if not run_lp else 0.0 for v in d_bess_ch], dtype=float)
        d_bess_dis_v = np.array([_val(v) if not run_lp else 0.0 for v in d_bess_dis], dtype=float)

    else:
        e_bess_v = np.zeros(T, dtype=float)
        d_bess_ch_v = np.zeros(T, dtype=float)
        d_bess_dis_v = np.zeros(T, dtype=float)


    d_grid_v = np.array([_val(v) if not run_lp else 0.0 for v in d_grid], dtype=float)

    e_grid_a_low_step_v = np.array([_val(v) for v in e_grid_a_low_step], dtype=float)
    e_grid_a_high_step_v = np.array([_val(v) for v in e_grid_a_high_step], dtype=float)

    e_grid_b_low_step_v = np.array([_val(v) for v in e_grid_b_low_step], dtype=float)
    e_grid_b_high_step_v = np.array([_val(v) for v in e_grid_b_high_step], dtype=float)

    e_grid_a_low_v = float(np.sum(e_grid_a_low_step_v))
    e_grid_a_high_v = float(np.sum(e_grid_a_high_step_v))

    e_grid_b_low_v = float(np.sum(e_grid_b_low_step_v))
    e_grid_b_high_v = float(np.sum(e_grid_b_high_step_v))

    import_cost_a_ft = (
            price_grid_a_low * e_grid_a_low_v
            + price_grid_a_high * e_grid_a_high_v
    )

    import_cost_b_ft = (
            price_grid_b_low * e_grid_b_low_v
            + price_grid_b_high * e_grid_b_high_v
    )

    export_revenue = float(np.sum(p_pv_grid_v) * dt * price_pv_grid)

    import_cost_ft = import_cost_a_ft + import_cost_b_ft

    brt_bill_ft = import_cost_ft - export_revenue

    if boiler_tariff == "A":
        p_grid_to_base_v = p_grid_ue_v + p_grid_elh_v + p_grid_bess_v
        p_grid_to_boiler_v = np.zeros(T, dtype=float)
    else:
        p_grid_to_base_v = p_grid_ue_v + p_grid_bess_v
        p_grid_to_boiler_v = p_grid_elh_v

    results = {
        "p_pv": p_pv,
        "p_ue": p_ue,
        "p_el_heater": p_el_heater,
        "p_total_load": total_load,
        "dt": dt,

        "size_bess": float(size_bess),
        "battery_power": float(battery_power),
        "soc_bess_min": float(soc_bess_min),
        "soc_bess_max": float(soc_bess_max),
        "soc_bess_init": float(soc_bess_init),
        "bess_active": int(bess_active),

        "p_pv_load": p_pv_ue_v + p_pv_elh_v,
        "p_pv_ue": p_pv_ue_v,
        "p_pv_elh": p_pv_elh_v,
        "p_pv_bess": p_pv_bess_v,
        "p_pv_grid": p_pv_grid_v,

        "p_bess_load": p_bess_ue_v + p_bess_elh_v,
        "p_grid_load": p_grid_ue_v + p_grid_elh_v,
        "p_bess_ue": p_bess_ue_v,
        "p_bess_elh": p_bess_elh_v,

        "p_grid_ue": p_grid_ue_v,
        "p_grid_elh": p_grid_elh_v,
        "p_grid_to_base": p_grid_to_base_v,
        "p_grid_to_boiler": p_grid_to_boiler_v,
        "boiler_tariff": boiler_tariff,

        "p_bess_in": p_bess_in_v,
        "p_bess_out": p_bess_out_v,
        "p_grid_bess": p_grid_bess_v,
        "e_bess": e_bess_v,
        "d_bess_ch": d_bess_ch_v,
        "d_bess_dis": d_bess_dis_v,

        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "d_grid": d_grid_v,

        "grid_import_total_kwh":
            e_grid_a_low_v
            + e_grid_a_high_v
            + e_grid_b_low_v
            + e_grid_b_high_v,


        "status": status_str,

        "grid_import_low_kwh":
            e_grid_a_low_v + e_grid_b_low_v,

        "grid_import_high_kwh":
            e_grid_a_high_v + e_grid_b_high_v,

        "grid_import_a_low_kwh": e_grid_a_low_v,
        "grid_import_a_high_kwh": e_grid_a_high_v,

        "grid_import_b_low_kwh": e_grid_b_low_v,
        "grid_import_b_high_kwh": e_grid_b_high_v,

        "import_cost_a_ft": import_cost_a_ft,
        "import_cost_b_ft": import_cost_b_ft,
        "import_cost_ft": import_cost_a_ft + import_cost_b_ft,
        "export_revenue_ft": export_revenue,
        "objective_Ft": float(brt_bill_ft),
        "brt_bill_ft": brt_bill_ft,
    }

    return results