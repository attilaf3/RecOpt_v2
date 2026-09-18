"""Thermal storage and controlled-load constraints shared by household models."""
import pulp

WATER_HEAT_CAPACITY_KWH_PER_L_K = 0.00116667

def add_boiler_step(prob, *, t, T, dt, hss_active, p_elh_in, p_hss_in,
                    p_hss_out, t_hss, d_cl, p_pv_elh, p_grid_elh, p_dhw,
                    size_elh, vol_hss_water, T_env, T_max, T_in, a_hss,
                    eta_elh, run_lp, p_bess_elh=None):
    k = (t + 1) % T
    c_hss = WATER_HEAT_CAPACITY_KWH_PER_L_K
    if hss_active:
        # ELH betáplálás
        prob += p_elh_in[t] == p_pv_elh[t] + p_grid_elh[t] + (p_bess_elh[t] if p_bess_elh is not None else 0), f"elh_supply_{t}"
        prob += p_hss_in[t] == eta_elh * p_elh_in[t], f"hss_in_def_{t}"

        # DHW igény kiszolgálása
        prob += p_hss_out[t] == p_dhw[t], f"dhw_balance_{t}"

        # Hőtároló dinamika
        prob += (
                vol_hss_water * c_hss * (t_hss[k] - t_hss[t])
                == dt * (p_hss_in[t] - p_hss_out[t])
                - a_hss * (t_hss[t] - T_env) * dt
        ), f"hss_balance_{t}"

        # következő lépésben kivehető max hő
        prob += (
            p_hss_out[k] <= vol_hss_water * c_hss * (t_hss[t] - T_in) / dt
        ), f"hss_max_out_{t}"

        # max betöltés
        prob += (
            p_hss_in[t] <= vol_hss_water * c_hss * (T_max - t_hss[t]) / dt
        ), f"hss_max_in_{t}"

        if not run_lp:
            prob += p_elh_in[t] <= size_elh * d_cl[t], f"elh_onoff_{t}"

        if not run_lp and t != 0 and t != T - 1:
            prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1], f"cl_min_on_time_{t}"

    else:
        prob += p_pv_elh[t] == 0, f"no_hss_pv_elh_{t}"
        prob += p_grid_elh[t] == 0, f"no_hss_grid_elh_{t}"


def add_boiler_schedule(prob, *, d_cl, T, dt, hss_active, enforce_cl_rules,
                        run_lp, cl_max_on_hours_per_day,
                        cl_min_midday_hours_per_day):
    if hss_active and enforce_cl_rules and not run_lp:
        n_timesteps_in_a_day = round(24 / dt)
        assert abs(n_timesteps_in_a_day * dt - 24) < 1e-9, "dt must divide 24h exactly"

        # 10:00–16:00 közötti "középső" időszak
        start_mid = round(10 / dt)
        mid_len = round(6 / dt)

        y_middle_day = [0] * n_timesteps_in_a_day
        for i in range(start_mid, min(start_mid + mid_len, n_timesteps_in_a_day)):
            y_middle_day[i] = 1

        max_on_steps = round(cl_max_on_hours_per_day / dt)
        min_mid_steps = round(cl_min_midday_hours_per_day / dt)

        for j in range(0, T, n_timesteps_in_a_day):
            day_idx = range(j, min(j + n_timesteps_in_a_day, T))
            if len(list(day_idx)) < n_timesteps_in_a_day:
                continue

            prob += pulp.lpSum(d_cl[t] for t in day_idx) <= max_on_steps, f"day_{j}_cl_maxon"
            prob += pulp.lpSum(d_cl[t] * y_middle_day[t - j] for t in day_idx) >= min_mid_steps, f"day_{j}_cl_midmin"


