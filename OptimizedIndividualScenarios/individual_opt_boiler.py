"""Compatibility entry point; model construction lives in individual_optimizer."""
import numpy as np
from Utility.configuration import config
from OptimizedIndividualScenarios.individual_optimizer import optimize_household


def individual_opt_boiler(
    p_pv,                   # (T,)
    p_ue,                   # (T,)
    p_dhw,                  # (T,) hőigény kW
    dt=0.25,
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
    objective="bill"
):
    options = locals().copy()
    return optimize_household(**options, include_boiler=True, include_bess=False)
