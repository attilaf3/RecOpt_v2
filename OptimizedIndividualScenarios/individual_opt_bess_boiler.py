"""Compatibility entry point; model construction lives in individual_optimizer."""
import numpy as np
from Utility.configuration import config
from OptimizedIndividualScenarios.individual_optimizer import optimize_household


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

    price_grid_a_low=config.getfloat("tariffs", "grid_a_low_ft_per_kwh"),
    price_grid_a_high=config.getfloat("tariffs", "grid_a_high_ft_per_kwh"),
    price_grid_b_low=config.getfloat("tariffs", "grid_b_low_ft_per_kwh"),
    price_grid_b_high=config.getfloat("tariffs", "grid_b_high_ft_per_kwh"),
    price_pv_grid=config.getfloat("tariffs", "pv_export_ft_per_kwh"),
    grid_a_low_cap_kwh=config.getfloat("tariffs", "grid_a_low_limit_kwh"),
    grid_b_low_cap_kwh=config.getfloat("tariffs", "grid_b_low_limit_kwh"),
    objective="grid",
    # --- Solve ---
    run_lp=False,           # ha True: LP relax (binárisok kikapcsolva)
    msg=False,
    gapRel=None,
    timeLimit=None,
    allow_grid_charge_for_min_soc=False,
):
    options = locals().copy()
    return optimize_household(**options, include_boiler=True, include_bess=True,
                              bess_terminal="cyclic",
                              boiler_tariff="B")
