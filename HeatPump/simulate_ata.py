# -*- coding: utf-8 -*-
"""
TABULA → EN ISO 13790 5R1C → 5R2C (Sepsi C_m, C_a) + szimuláció
----------------------------------------------------------------
- 5R2C helyettesítő áramkör:

      T_out
        │
     H_em
        │
       T_m ── H_ms ── T_s ── H_is ── T_i ── H_v ── T_out
        │                         │
       C_m                       C_a

- Vezetők (H-k) az EN ISO 13790 5R1C szerint:
    * H_w   = H_tr,w   (ablakos rész)
    * H_em  = H_tr,em  (tömeg ↔ külvilág)
    * H_ms  = H_tr,ms  (tömeg ↔ belső felület)
    * H_is  = H_tr,is  (belső felület ↔ levegő)
    * H_v   = H_ve     (szellőzés)

- Kapacitások (C-k) Sepsi szerint:
    * C_m  = theta_m * c_m,heavy/3600 * A_ref      [kWh/K]
    * C_a  = m_air * 0.34 * V / 1000              [kWh/K],  V = A_ref*h

- 15 perces Euler szimuláció:
    * szabad futás (HP = 0)
    * termosztátos futás (23±2 °C)
    * HP hő- és villamos teljesítmény COP(T_out) alapján
"""

from dataclasses import dataclass
from os.path import join
from typing import Dict, Mapping, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===================== FÁJL UTAK =====================
root = "C:\\NextCloud\\Doktori\\8_felev\\Onlab\\EnergiakozossegOptimalizalas\\bemeneti_fajlok\\optimize_yaml_detailed_input_file"
CSV_TEMP = join(root, "homerseklet_sugarzas_interpolated_polynomial_fixed.csv")
CSV_SOLAR = join(root, "zsombo_window_irradiance_2021_fixed_interpolated_polynomial.csv")

SOLAR_COLS = {"S": "G_ablak_D", "E": "G_ablak_K", "W": "G_ablak_Ny", "N": "G_ablak_É"}  # [W/m²]

# ===================== ÁLLANDÓK =====================
DT_H = 0.25  # 15 perc
T_SET = 23.0  # termosztát setpoint [°C]
DEADBAND = 2.0  # ±2 °C

# Szoláris nyereség konstansok (Sepsi 2.14-hez hasonlóan)
F_F = 0.30
F_W = 0.90
G_GLN = 0.60
F_SH_WINTER, F_SH_SUMMER = 0.70, 0.50

# Sepsi-féle kapacitások
C_MASS_HEAVY_KJ_M2K = 280.0  # [kJ/(m²K)] → 0.0778 kWh/(m²K)
THETA_MASS = 1.0  # hozzáférhetőségi tényező C_m-re
M_AIR = 5.0  # levegő "m" szorzó

CPRHO_AIR_Wh_m3K = 0.34  # [Wh/(m³K)]

# ISO 13790 dinamikus paraméterek
H_IS_W_M2K = 3.45  # h_is
H_MS_W_M2K = 9.10  # h_ms
LAMBDA_AT = 4.5  # A_tot = 4.5 A_ref
LAMBDA_M = 3.0  # A_m   = 3.0 A_ref

# Hőszivattyú COP modell
EER_COOLING = 2.8
A0_COP = 8.6712
A1_COP = -0.1800
A2_COP = 0.0
T_SINK_HEAT_C = 35.0
COP_CLAMP_MIN, COP_CLAMP_MAX = 1.0, 6.0


def cop_heating_from_env(t_env_c: float, a0: float = A0_COP, a1: float = A1_COP, a2: float = A2_COP,
                         t_sink_c: float = T_SINK_HEAT_C) -> float:
    """COP(ΔT) polinom Sepsi alapján."""
    dT = max(0.0, t_sink_c - float(t_env_c))
    cop = a0 + a1 * dT + a2 * (dT ** 2)
    if not np.isfinite(cop):
        return COP_CLAMP_MIN
    return max(COP_CLAMP_MIN, min(COP_CLAMP_MAX, cop))


# ===================== TABULA HÁZAK (adatok a 3r2c.py-ból) =====================
# Megjegyzés: U_window értéket a TABULA PDF "actual U-value" ablak sorából érdemes venni.
HOUSES_RAW = {"SFH-01": dict(Aref=101.8, h=2.5, Htr=134.0, Hve=43.0, U_window=1.0, window_total=14.0,
    Awin_raw={"E": 0.0, "S": 0.0, "W": 7.2, "N": 6.8},  # csak a szoláris bontáshoz
), "SFH-02": dict(Aref=115.6, h=2.5, Htr=148.0, Hve=49.0, U_window=1.0, window_total=17.8,
    Awin_raw={"E": 8.6, "S": 0.2, "W": 8.8, "N": 0.2}, ),
    "SFH-03": dict(Aref=103.2, h=2.5, Htr=63.0, Hve=44.0, U_window=1.0, window_total=12.4,
        Awin_raw={"E": 3.6, "S": 2.6, "W": 3.6, "N": 2.6}, ),
    "SFH-04": dict(Aref=110.2, h=2.5, Htr=63.0, Hve=47.0, U_window=1.0, window_total=19.6,
        Awin_raw={"E": 6.0, "S": 3.8, "W": 6.0, "N": 3.8}, ),
    "SFH-05": dict(Aref=131.6, h=2.5, Htr=62.0, Hve=50.0, U_window=1.0, window_total=17.9,
        Awin_raw={"E": 4.1, "S": 7.2, "W": 4.1, "N": 2.5}, ), }


# ===================== SEGÉDFÜGGVÉNYEK: PARAMÉTEREK =====================
def normalize_window_orientations(raw_by_dir_m2: Mapping[str, float], total_window_area_m2: float) -> Dict[str, float]:
    """Orientációs arányok → irányonkénti m² a teljes ablak-területre skálázva."""
    s = float(sum(raw_by_dir_m2.values()))
    if s <= 0:
        return {k: 0.0 for k in ["E", "S", "W", "N"]}
    scale = total_window_area_m2 / s
    return {k: float(v) * scale for k, v in raw_by_dir_m2.items()}


def compute_C_mass_sepsi(Aref_m2: float, theta_mass: float = THETA_MASS,
                         c_m_kJ_m2K: float = C_MASS_HEAVY_KJ_M2K) -> float:
    """
    Szerkezeti tömeg hőkapacitása C_m [kWh/K] Sepsi szerint:
        C_m/A_ref = c_m,heavy [kJ/(m²K)].
    """
    Cm_kWh_per_m2K = c_m_kJ_m2K / 3600.0
    return theta_mass * Cm_kWh_per_m2K * Aref_m2


def compute_C_air_sepsi(Aref_m2: float, h_m: float, m_air_factor: float = M_AIR) -> float:
    """
    Belső levegő hőkapacitása C_a [kWh/K] Sepsi szerint:
        C_a = m * (ρc_p) * V / 1000,
        ρc_p ≈ 0.34 Wh/(m³K),  V = A_ref*h.
    """
    V_m3 = Aref_m2 * h_m
    Ca_Wh_per_K = m_air_factor * CPRHO_AIR_Wh_m3K * V_m3
    return Ca_Wh_per_K / 1000.0


@dataclass
class R5R2CParams:
    C_air_kWh_per_K: float
    C_mass_kWh_per_K: float
    H_em_kW_per_K: float
    H_w_kW_per_K: float
    H_ms_kW_per_K: float
    H_is_kW_per_K: float
    H_v_kW_per_K: float
    H_tr_kW_per_K: float
    H_tr_op_kW_per_K: float
    Awin_by_dir_m2: Dict[str, float]


def tabula_to_5r2c_iso_sepsi(Aref_m2: float, h_m: float, Htr_W_per_K: float, Hve_W_per_K: float, U_window_W_m2K: float,
                             window_total_m2: float, Awin_raw: Dict[str, float]) -> R5R2CParams:
    """
    TABULA → EN ISO 13790 5R1C → 5R2C paraméterek.
    - C_m, C_a Sepsi szerint.
    - H_w, H_tr,op, H_is, H_ms, H_em, H_v ISO 13790 szerint.
    """
    # Kapacitások
    C_mass_kWhK = compute_C_mass_sepsi(Aref_m2)
    C_air_kWhK = compute_C_air_sepsi(Aref_m2, h_m)

    # Ablakos transzmisszió
    H_tr_w = U_window_W_m2K * window_total_m2  # [W/K]
    # Teljes transzmisszió
    H_tr = float(Htr_W_per_K)
    # Opa k rész
    H_tr_op = H_tr - H_tr_w

    # Dinamikus felületek
    A_tot = LAMBDA_AT * Aref_m2  # belső felület
    A_m = LAMBDA_M * Aref_m2  # tömegfelület

    H_is = H_IS_W_M2K * A_tot  # [W/K]
    H_ms = H_MS_W_M2K * A_m  # [W/K]

    # Mass envelope: 1/H_em = 1/H_tr,op - 1/H_ms
    H_em = 1.0 / (1.0 / H_tr_op - 1.0 / H_ms)

    # Szellőzés
    H_v = float(Hve_W_per_K)

    # kW/K-ra váltás
    H_em_kW = H_em / 1000.0
    H_w_kW = H_tr_w / 1000.0
    H_ms_kW = H_ms / 1000.0
    H_is_kW = H_is / 1000.0
    H_v_kW = H_v / 1000.0
    H_tr_kW = H_tr / 1000.0
    H_tr_op_kW = H_tr_op / 1000.0

    Awin_by_dir = normalize_window_orientations(Awin_raw, window_total_m2)

    return R5R2CParams(C_air_kWh_per_K=C_air_kWhK, C_mass_kWh_per_K=C_mass_kWhK, H_em_kW_per_K=H_em_kW,
        H_w_kW_per_K=H_w_kW, H_ms_kW_per_K=H_ms_kW, H_is_kW_per_K=H_is_kW, H_v_kW_per_K=H_v_kW, H_tr_kW_per_K=H_tr_kW,
        H_tr_op_kW_per_K=H_tr_op_kW, Awin_by_dir_m2=Awin_by_dir, )


# ===================== SZOLÁRIS HŐNYERESÉG =====================
def solar_gain_sepsi(I_by_dir_Wm2: Mapping[str, pd.Series], Awin_by_dir_m2: Mapping[str, float],
                     idx: pd.DatetimeIndex) -> pd.Series:
    """
    Q_sol(t) [kW] Sepsi 2.14 logika szerint:
        Q_sol = Σ F_sh(t) (1-F_F) F_W g_gl,n A_win I(t) / 1000
    Itt F_sh-t évszakonként (tél/nyár), a többit globális konstansként kezeljük.
    """
    dirs = ["E", "S", "W", "N"]
    Q = pd.Series(0.0, index=idx)
    months = idx.month.values

    for k, t in enumerate(idx):
        F_sh = F_SH_WINTER if (months[k] <= 3 or months[k] >= 10) else F_SH_SUMMER
        coeff = F_sh * (1.0 - F_F) * F_W * G_GLN
        term_W = 0.0
        for d in dirs:
            I = I_by_dir_Wm2.get(d, None)
            I_t = float(I.iloc[k]) if I is not None else 0.0
            A = Awin_by_dir_m2.get(d, 0.0)
            term_W += A * I_t
        Q.iloc[k] = coeff * term_W / 1000.0  # W → kW

    return Q


def build_setpoint_profile(idx: pd.DatetimeIndex, t_cold: float = 22.0, t_summer: float = 25.0) -> np.ndarray:
    """Return a seasonal setpoint profile: summer uses t_summer, otherwise t_cold."""
    months = idx.month.values
    is_summer = (months >= 6) & (months <= 8)
    return np.where(is_summer, t_summer, t_cold).astype(float)


def calculate_participation_factor(idx: pd.DatetimeIndex, T_env: np.ndarray) -> np.ndarray:
    """
    Calculate participation factor for each timestep based on daily average temperature.
    
    Rules:
    - Tavg < 0: pf = 0
    - 0 <= Tavg <= 5: pf = 0.5
    - 5 < Tavg <= 7: pf = 0.8
    - Tavg > 7: pf = 1.0
    
    Returns an array with pf value for each timestep.
    """
    # Create a DataFrame with temperature and date
    df_temp = pd.DataFrame({'T': T_env, 'date': idx.date})
    
    # Calculate daily average temperature
    daily_avg = df_temp.groupby('date')['T'].mean()
    
    # Map daily average to participation factor
    def get_pf(tavg):
        if tavg < 0:
            return 0.0
        elif tavg <= 5:
            return 0.4
        elif tavg <= 7:
            return 0.7
        else:
            return 1.0
    
    daily_pf = daily_avg.apply(get_pf)
    
    # Expand daily pf back to timestep level
    df_temp['pf'] = df_temp['date'].map(daily_pf)
    
    return df_temp['pf'].values.astype(float)


# ===================== 5R2C SZIMULÁCIÓ =====================
def simulate_5r2c(idx: pd.DatetimeIndex, T_env: np.ndarray, Q_sol_kW: np.ndarray, pars: R5R2CParams, dt_h: float = DT_H,
                  T_set: float = T_SET, deadband: float = DEADBAND, min_on_min: float = 20.0, min_off_min: float = 20.0,
                  p_th_max_kW: float = 12.0, p_ramp_kW_per_step: float = 1.5,
                  hp_block_windows: Tuple[Tuple[int, int], ...] | dict = ((7, 9), (16, 18)),
                  t_set_profile: Optional[np.ndarray] = None,
                  enforce_hysteresis: bool = True,
                  enforce_min_on_off: bool = True,
                  enforce_ramp_limit: bool = True,
                  enforce_participation_factor: bool = True,
                  power_mode: str = "min_cap",
                  p_th_min_kW: float = 2.0,
                  staged_levels: Tuple[float, ...] = (0.0, 0.5, 1.0)) -> Dict[str, np.ndarray]:
    """
    5R2C modell explicit Ts csomóponttal, Sepsi C-k, ISO H-k.
    Anti-oscillation features:
    - Hysteresis with state memory (keeps HP mode in deadband)
    - Minimum on/off times (prevents short-cycling)
    - Power ramping (smooth transitions)

    Egyenletek:
        H_sum = H_w + H_ms + H_is
        T_s = (H_w T_out + H_ms T_m + H_is T_i) / H_sum

        dT_i/dt = [H_is (T_s - T_i) + H_v (T_out - T_i)
                   + Q_sol + P_hp_th] / C_a
        dT_m/dt = [H_ms (T_s - T_m) + H_em (T_out - T_m)] / C_m

    hp_block_windows: (start_hour, end_hour) tuple list, end exclusive.
    t_set_profile: optional per-timestep setpoint array; defaults to seasonal profile.
    power_mode: "continuous", "min_cap", or "staged".
    p_th_min_kW: minimum thermal capacity when on (used for "min_cap").
    staged_levels: fractional stages of p_th_max_kW (used for "staged").
    """
    n = len(T_env)
    if t_set_profile is None:
        t_set_profile = build_setpoint_profile(idx)

    # Calculate participation factor based on daily average temperature
    participation_factor = calculate_participation_factor(idx, T_env)

    T_i_free = np.zeros(n)
    T_m_free = np.zeros(n)
    T_i_ctrl = np.zeros(n)
    T_m_ctrl = np.zeros(n)
    P_hp_th = np.zeros(n)
    P_hp_el = np.zeros(n)

    T_i_free[0] = T_m_free[0] = T_i_ctrl[0] = T_m_ctrl[0] = float(t_set_profile[0])

    C_a = pars.C_air_kWh_per_K
    C_m = pars.C_mass_kWh_per_K
    H_em = pars.H_em_kW_per_K
    H_w = pars.H_w_kW_per_K
    H_ms = pars.H_ms_kW_per_K
    H_is = pars.H_is_kW_per_K
    H_v = pars.H_v_kW_per_K
    H_sum = H_w + H_ms + H_is

    # Szabad futás
    for t in range(n - 1):
        T_out = T_env[t]
        T_s = (H_w * T_out + H_ms * T_m_free[t] + H_is * T_i_free[t]) / H_sum

        dTi = (H_is * (T_s - T_i_free[t]) + H_v * (T_out - T_i_free[t]) + Q_sol_kW[t]) / C_a
        dTm = (H_ms * (T_s - T_m_free[t]) + H_em * (T_out - T_m_free[t])) / C_m

        T_i_free[t + 1] = T_i_free[t] + dTi * dt_h
        T_m_free[t + 1] = T_m_free[t] + dTm * dt_h

    # Termosztátos futás with anti-oscillation
    min_on_steps = max(1, int(round((min_on_min / 60.0) / dt_h)))
    min_off_steps = max(1, int(round((min_off_min / 60.0) / dt_h)))

    hp_mode = 0  # -1 cooling, 0 off, +1 heating
    steps_in_mode = min_off_steps
    p_prev = 0.0

    for t in range(n - 1):
        T_out = T_env[t]
        T_s = (H_w * T_out + H_ms * T_m_ctrl[t] + H_is * T_i_ctrl[t]) / H_sum

        t_set_t = float(t_set_profile[t])
        t_min = t_set_t - deadband
        t_max = t_set_t + deadband

        hour = idx[t].hour

        # hp_block_windows may be either a sequence of (start,end) tuples that
        # apply all year, or a dict with seasonal entries (e.g. 'winter'/'summer')
        # mapping to sequences of tuples. If it's a dict, pick the appropriate
        # window set based on the month of the current timestep.
        if isinstance(hp_block_windows, dict):
            month = idx[t].month
            if month in (12, 1, 2):
                windows = hp_block_windows.get("winter", hp_block_windows.get("default", ()))
            elif month in (6, 7, 8):
                windows = hp_block_windows.get("summer", hp_block_windows.get("default", ()))
            else:
                windows = hp_block_windows.get("default", ())
        else:
            windows = hp_block_windows

        hp_blocked = any(h0 <= hour < h1 for h0, h1 in windows)

        if hp_blocked:
            # Hard-off window: no HP operation, no ramp carryover.
            hp_mode = 0
            steps_in_mode = min_off_steps
            p_prev = 0.0
            P_hp_th[t] = 0.0
            P_hp_el[t] = 0.0

            dTi = (H_is * (T_s - T_i_ctrl[t]) + H_v * (T_out - T_i_ctrl[t]) + Q_sol_kW[t]) / C_a
            dTm = (H_ms * (T_s - T_m_ctrl[t]) + H_em * (T_out - T_m_ctrl[t])) / C_m

            T_i_ctrl[t + 1] = T_i_ctrl[t] + dTi * dt_h
            T_m_ctrl[t + 1] = T_m_ctrl[t] + dTm * dt_h
            continue

        # drift prediction without HP
        dTi_free = (H_is * (T_s - T_i_ctrl[t]) + H_v * (T_out - T_i_ctrl[t]) + Q_sol_kW[t]) / C_a
        T_i_pred = T_i_ctrl[t] + dTi_free * dt_h

        # hysteresis decision: stay in current mode if in deadband
        requested_mode = hp_mode
        if hp_mode == 0:
            if T_i_pred < t_min:
                requested_mode = +1  # turn on heating
            elif T_i_pred > t_max:
                requested_mode = -1  # turn on cooling
        elif hp_mode == +1 and T_i_pred >= t_set_t:
            requested_mode = 0  # turn off heating
        elif hp_mode == -1 and T_i_pred <= t_set_t:
            requested_mode = 0  # turn off cooling

        if not enforce_hysteresis:
            if T_i_pred < t_set_t:
                requested_mode = +1
            elif T_i_pred > t_set_t:
                requested_mode = -1
            else:
                requested_mode = 0

        # enforce minimum on/off times
        if enforce_min_on_off and requested_mode != hp_mode:
            if hp_mode == 0 and steps_in_mode < min_off_steps:
                requested_mode = 0  # stay off
            elif hp_mode != 0 and steps_in_mode < min_on_steps:
                requested_mode = hp_mode  # stay on

        if requested_mode == hp_mode:
            steps_in_mode += 1
        else:
            hp_mode = requested_mode
            steps_in_mode = 0

        # compute target thermal power only if ON; otherwise 0
        if hp_mode == 0:
            p_target = 0.0
        else:
            p_need = (C_a / dt_h) * (t_set_t - T_i_ctrl[t]) \
                     - H_is * (T_s - T_i_ctrl[t]) \
                     - H_v * (T_out - T_i_ctrl[t]) \
                     - Q_sol_kW[t]
            # force sign by mode
            if hp_mode == +1:
                p_target = max(0.0, p_need)
            else:
                p_target = min(0.0, p_need)

            p_target = float(np.clip(p_target, -p_th_max_kW, p_th_max_kW))

        if hp_mode != 0 and power_mode != "continuous":
            if power_mode == "min_cap":
                p_target = np.sign(p_target) * max(abs(p_target), p_th_min_kW)
            elif power_mode == "staged":
                levels = sorted(set(float(x) for x in staged_levels))
                if not levels:
                    levels = [0.0, 1.0]
                best = min(levels, key=lambda lvl: abs(abs(p_target) - lvl * p_th_max_kW))
                p_target = np.sign(p_target) * best * p_th_max_kW

        # ramp limit: smooth power transitions
        if enforce_ramp_limit:
            p_cmd = np.clip(p_target, p_prev - p_ramp_kW_per_step, p_prev + p_ramp_kW_per_step)
        else:
            p_cmd = p_target

        p_prev = p_cmd
        P_hp_th[t] = p_cmd

        if p_cmd >= 0:
            cop =  max(COP_CLAMP_MIN, cop_heating_from_env(T_out))
            pf = participation_factor[t] if enforce_participation_factor else 1
            P_hp_el[t] = p_cmd / cop * pf
        else:
            P_hp_el[t] = abs(p_cmd) / max(0.5, EER_COOLING)

        dTi = (H_is * (T_s - T_i_ctrl[t]) + H_v * (T_out - T_i_ctrl[t]) + Q_sol_kW[t] + p_cmd) / C_a
        dTm = (H_ms * (T_s - T_m_ctrl[t]) + H_em * (T_out - T_m_ctrl[t])) / C_m

        T_i_ctrl[t + 1] = T_i_ctrl[t] + dTi * dt_h
        T_m_ctrl[t + 1] = T_m_ctrl[t] + dTm * dt_h

    return dict(T_env=np.asarray(T_env), T_i_free=T_i_free, T_m_free=T_m_free, T_i_ctrl=T_i_ctrl, T_m_ctrl=T_m_ctrl,
        P_hp_th=P_hp_th, P_hp_el=P_hp_el, participation_factor=participation_factor)


# ===================== IDŐÁLLANDÓK =====================
def time_constants_from_5r2c(pars: R5R2CParams) -> Tuple[float, float]:
    """
    Gyors és lassú időállandók az 5R2C-ből.
    A = [[a11,a12],[a21,a22]], λ_i sajátértékek [1/h] → τ_i = -1/λ_i [h].
    Visszatérés: (τ_fast [perc], τ_slow [óra]).
    """
    C_a = float(pars.C_air_kWh_per_K)
    C_m = float(pars.C_mass_kWh_per_K)
    H_em = float(pars.H_em_kW_per_K)
    H_w = float(pars.H_w_kW_per_K)
    H_ms = float(pars.H_ms_kW_per_K)
    H_is = float(pars.H_is_kW_per_K)
    H_v = float(pars.H_v_kW_per_K)

    H_sum = H_w + H_ms + H_is

    a11 = (- H_is * H_ms - H_is * H_v - H_is * H_w - H_ms * H_v - H_v * H_w) / (C_a * H_sum)
    a12 = (H_is * H_ms) / (C_a * H_sum)
    a21 = (H_is * H_ms) / (C_m * H_sum)
    a22 = (- H_em * H_is - H_em * H_ms - H_em * H_w - H_is * H_ms - H_ms * H_w) / (C_m * H_sum)

    A = np.array([[a11, a12], [a21, a22]], dtype=float)

    eigvals = np.linalg.eigvals(A)
    taus_h = sorted([-1.0 / x for x in eigvals], key=lambda z: np.real(z))
    tau_fast_min = float(np.real(taus_h[0]) * 60.0)
    tau_slow_h = float(np.real(taus_h[1]))
    return tau_fast_min, tau_slow_h


# ===================== BEMENETEK BEOLVASÁSA =====================
def load_or_make_inputs() -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, pd.Series]]:
    """Külső T_env és homlokzati sugárzás beolvasása, vagy szintetikus generálása."""
    # Külső hőmérséklet
    df_temp = pd.read_csv(CSV_TEMP, index_col=0, parse_dates=True)
    T_env = df_temp["T"].astype(float)
    idx = T_env.index

    # Homlokzati globál W/m²
    df_solar = pd.read_csv(CSV_SOLAR, index_col=0, parse_dates=True).reindex(idx)
    I_by_dir = {d: df_solar[SOLAR_COLS[d]].astype(float).fillna(0.0) for d in SOLAR_COLS}

    return pd.DatetimeIndex(idx), T_env.values, I_by_dir


# ===================== FŐPROGRAM =====================
if __name__ == "__main__":
    idx, T_env, I_by_dir = load_or_make_inputs()

    all_params: Dict[str, R5R2CParams] = {}
    all_results: Dict[str, Tuple[pd.Series, Dict[str, np.ndarray]]] = {}

    for name, d in HOUSES_RAW.items():
        pars = tabula_to_5r2c_iso_sepsi(Aref_m2=d["Aref"], h_m=d["h"], Htr_W_per_K=d["Htr"], Hve_W_per_K=d["Hve"],
            U_window_W_m2K=d["U_window"], window_total_m2=d["window_total"], Awin_raw=d["Awin_raw"], )
        all_params[name] = pars

        Q_sol = solar_gain_sepsi(I_by_dir, pars.Awin_by_dir_m2, idx)
        res = simulate_5r2c(idx, T_env, Q_sol.values, pars, dt_h=DT_H, T_set=T_SET, deadband=DEADBAND)
        all_results[name] = (Q_sol, res)

        print(f"\n=== {name} – 5R2C (ISO H-k + Sepsi C-k) ===")
        print(f" C_a  = {pars.C_air_kWh_per_K:.3f} kWh/K,"
              f"   C_m = {pars.C_mass_kWh_per_K:.3f} kWh/K")
        print(f" H_w  = {pars.H_w_kW_per_K:.3f} kW/K  (ablak)")
        print(f" H_em = {pars.H_em_kW_per_K:.3f} kW/K  (T_out↔T_m, opak)")
        print(f" H_ms = {pars.H_ms_kW_per_K:.3f} kW/K  (T_m↔T_s)")
        print(f" H_is = {pars.H_is_kW_per_K:.3f} kW/K  (T_s↔T_i)")
        print(f" H_v  = {pars.H_v_kW_per_K:.3f} kW/K  (T_i↔T_out)")
        print(f" H_tr teljes = {pars.H_tr_kW_per_K:.3f} kW/K,"
              f"   H_tr,op = {pars.H_tr_op_kW_per_K:.3f} kW/K")
        print(f" A_win [m²]: {pars.Awin_by_dir_m2}")

        tau_fast_min, tau_slow_h = time_constants_from_5r2c(pars)
        print(f" τ_fast ≈ {tau_fast_min:.1f} perc,"
              f"   τ_slow ≈ {tau_slow_h:.1f} óra")

    # Példa ábrák – egy 4 napos szakasz
    try:
        start = idx.min() + pd.Timedelta(days=10)
        end = start + pd.Timedelta(days=4)
        mask = (idx >= start) & (idx <= end)
    except Exception:
        mask = slice(None)

    for name in HOUSES_RAW.keys():
        Q_sol, res = all_results[name]

        # Hőmérsékletek
        plt.figure(figsize=(11, 4.0))
        plt.plot(idx[mask], res["T_env"][mask], label="T_out (°C)", lw=1.0)
        plt.plot(idx[mask], res["T_i_free"][mask], label="T_i free (HP=0)", lw=1.1)
        plt.plot(idx[mask], res["T_i_ctrl"][mask], label="T_i ctrl (23±2 °C)", lw=1.2)
        plt.axhline(T_SET - DEADBAND, ls="--", c="k", lw=0.8)
        plt.axhline(T_SET + DEADBAND, ls="--", c="k", lw=0.8)
        plt.title(f"{name}: zónahőmérséklet – 5R2C (ISO+Sepsi)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # HP teljesítmények
        plt.figure(figsize=(11, 3.5))
        plt.plot(idx[mask], res["P_hp_th"][mask], label="P_hp_th (kW) – hő", lw=1.0)
        plt.plot(idx[mask], res["P_hp_el"][mask], label="P_hp_el (kW) – vill.", lw=1.0)
        plt.title(f"{name}: hőszivattyú teljesítmény – 5R2C")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Szoláris hőnyereség
        plt.figure(figsize=(11, 3.0))
        plt.plot(idx[mask], Q_sol[mask], label="Q_sol (kW) – Sepsi (2.14)", lw=1.0)
        plt.title(f"{name}: szoláris hőnyereség")
        plt.legend()
        plt.tight_layout()
        plt.show()
