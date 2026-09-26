"""Mért és rule-based villanybojlerprofilok kezelése."""

from __future__ import annotations

import numpy as np

from nonopt_data import C_HSS, DT, NonoptInput


EPS = 1e-12
T_COMFORT_DEFAULT = 40.0
COMFORT_REACH_INTERVAL_HOURS_DEFAULT = 6.0


def _rule_based_available(step: int) -> bool:
    """A bojler 6–9 és 17–22 között semmilyen forrásból nem működhet."""
    hour = step * DT % 24.0
    return not (6.0 <= hour < 9.0 or 17.0 <= hour < 22.0)


def _rolling_reach_violations(
    reached: np.ndarray,
    window_steps: int,
) -> np.ndarray:
    """Az ablak végén jelzi, ha az előző window_steps mintában nem volt elérés."""
    reached = np.asarray(reached, dtype=bool)
    violations = np.zeros(reached.size, dtype=bool)
    if reached.size < window_steps:
        return violations
    hits = np.convolve(
        reached.astype(np.int16),
        np.ones(window_steps, dtype=np.int16),
        mode="valid",
    )
    violations[window_steps - 1:] = hits == 0
    return violations


def simulate_hss(
    p_dhw: np.ndarray,
    *,
    size_elh: float,
    volume: float,
    t_env: float,
    t_max: float,
    t_min: float,
    t_in: float,
    t_set: float,
    a_hss: float,
    eta_elh: float,
    tariff: str,
    min_activation_hours: float = 2.0,
    t_comfort: float = T_COMFORT_DEFAULT,
    comfort_reach_interval_hours: float = COMFORT_REACH_INTERVAL_HOURS_DEFAULT,
) -> dict[str, np.ndarray]:
    n = len(p_dhw)
    result = {name: np.zeros(n) for name in (
        "e_boiler", "p_elh_in", "p_hss_in", "p_hss_out", "e_hss",
        "t_hss", "t_setpoint", "d_available", "heating_blocked",
        "d_enable", "d_start", "d_on", "energy_violation", "temperature_violation",
        "dhw_energy_shortfall", "dhw_energy_margin",
        "reached_comfort", "rolling_comfort_violation",
    )}
    p_dhw = np.maximum(np.asarray(p_dhw, dtype=float), 0.0)
    if size_elh <= EPS or volume <= EPS:
        return result

    tariff = tariff.upper()
    if tariff not in {"A", "B"}:
        raise ValueError("A bojler tarifája csak 'A' vagy 'B' lehet.")

    if not (t_in <= t_comfort <= t_set <= t_max):
        raise ValueError(
            "A bojlernél T_in <= T_comfort <= T_set <= T_max szükséges: "
            f"T_in={t_in}, T_comfort={t_comfort}, "
            f"T_set={t_set}, T_max={t_max}."
        )
    comfort_window_steps = int(round(comfort_reach_interval_hours / DT))
    if (
        comfort_reach_interval_hours <= 0
        or comfort_window_steps < 1
        or abs(comfort_window_steps * DT - comfort_reach_interval_hours) > 1e-9
    ):
        raise ValueError(
            "A comfort_reach_interval_hours legyen pozitív és essen a DT időrácsára."
        )

    # A korábbi t_min és minimum bekapcsolási idő csak API-kompatibilitás miatt
    # marad paraméter. Az abszolút energiaszint a belépő víz hőmérséklete.
    del t_min, min_activation_hours
    capacity = volume * C_HSS  # kWh/°C
    e_abs_min = 0.0
    e_max = capacity * (t_max - t_in)
    e_set = capacity * (t_set - t_in)
    energy = e_set
    was_on = False

    for t in range(n):
        available = _rule_based_available(t)
        result["d_available"][t] = float(available)
        result["heating_blocked"][t] = float(not available)
        result["e_hss"][t] = energy
        equivalent_temperature = t_in + energy / capacity
        result["t_hss"][t] = equivalent_temperature
        result["t_setpoint"][t] = t_set

        loss = a_hss * (equivalent_temperature - t_env)
        # A DHW-igényt a lépés elején már eltárolt energiának kell fedeznie;
        # az ugyanebben a lépésben bekapcsoló fűtőszál nem fedi el a hiányt.
        available_dhw_power = max(energy - e_abs_min, 0.0) / DT
        delivered_dhw = min(p_dhw[t], available_dhw_power)
        result["p_hss_out"][t] = delivered_dhw
        result["dhw_energy_shortfall"][t] = DT * (p_dhw[t] - delivered_dhw)
        result["dhw_energy_margin"][t] = energy - e_abs_min - DT * p_dhw[t]

        below_setpoint = energy < e_set - 1e-9
        if available and below_setpoint:
            # A 15 perces átlagteljesítmény csak akkora, hogy a DHW-igény és
            # veszteség után se lépje túl a YAML-ben megadott setpointot.
            required_thermal_power = (
                (e_set - energy) / DT + delivered_dhw + loss
            )
            p_elh = min(
                size_elh,
                max(required_thermal_power, 0.0) / max(eta_elh, EPS),
            )
        else:
            # A setpoint elérésekor a fűtőszál leáll.
            p_elh = 0.0

        is_on = p_elh > EPS
        result["d_enable"][t] = float(available and below_setpoint)
        result["d_start"][t] = float(is_on and not was_on)
        result["p_elh_in"][t] = p_elh
        result["p_hss_in"][t] = eta_elh * p_elh
        result["e_boiler"][t] = p_elh * DT
        result["d_on"][t] = float(is_on)

        next_energy = energy + DT * (
            eta_elh * p_elh - delivered_dhw - loss
        )
        result["energy_violation"][t] = float(
            next_energy < e_abs_min - 1e-6 or next_energy > e_max + 1e-6
        )
        # Régi kimeneti név megtartva a meglévő feldolgozók kompatibilitásához.
        result["temperature_violation"][t] = result["energy_violation"][t]
        energy = float(np.clip(next_energy, e_abs_min, e_max))
        was_on = is_on

    reached_comfort = result["t_hss"] >= t_comfort - 1e-6
    result["reached_comfort"][:] = reached_comfort.astype(float)
    result["rolling_comfort_violation"][:] = _rolling_reach_violations(
        reached_comfort,
        comfort_window_steps,
    ).astype(float)
    return result


def build_boiler_profiles(
    data: NonoptInput,
    *,
    rule_based: bool,
    tariff: str,
    min_activation_hours: float = 2.0,
    t_comfort: float = T_COMFORT_DEFAULT,
    comfort_reach_interval_hours: float = COMFORT_REACH_INTERVAL_HOURS_DEFAULT,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n, users = data.e_base.shape
    e_boiler = data.e_boiler_measured.copy()
    diagnostics = {name: np.zeros((n, users)) for name in (
        "p_dhw", "p_elh_in", "p_hss_in", "p_hss_out", "e_hss", "t_hss",
        "t_setpoint", "d_available", "heating_blocked", "d_enable", "d_start",
        "d_on", "energy_violation", "temperature_violation",
        "dhw_energy_shortfall", "dhw_energy_margin",
        "reached_comfort", "rolling_comfort_violation",
    )}
    dynamic = np.zeros(users, dtype=bool)

    if rule_based and tariff.upper() != "A":
        raise ValueError("A rule-based bojleres esetekben a bojlernek A tarifán kell lennie.")

    for u in range(users):
        valid = (
            data.e_pv[:, u].sum() > EPS
            and data.size_elh[u] > EPS
            and data.vol_hss_water[u] > EPS
            and data.p_dhw[:, u].sum() > EPS
        )
        if not rule_based or not valid:
            continue
        model = simulate_hss(
            data.p_dhw[:, u], size_elh=data.size_elh[u], volume=data.vol_hss_water[u],
            t_env=data.t_env[u], t_max=data.t_max[u], t_min=data.t_min[u],
            t_in=data.t_in[u], t_set=data.t_set[u], a_hss=data.a_hss[u],
            eta_elh=data.eta_elh[u], tariff=tariff,
            min_activation_hours=min_activation_hours,
            t_comfort=t_comfort,
            comfort_reach_interval_hours=comfort_reach_interval_hours,
        )
        e_boiler[:, u] = model["e_boiler"]
        diagnostics["p_dhw"][:, u] = data.p_dhw[:, u]
        for key in diagnostics:
            if key != "p_dhw":
                diagnostics[key][:, u] = model[key]
        dynamic[u] = True
    diagnostics["dynamic"] = dynamic
    return e_boiler, diagnostics