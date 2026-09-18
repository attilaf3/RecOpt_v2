"""Compatibility entry point; model construction lives in individual_optimizer."""
import numpy as np
from Utility.configuration import config
from OptimizedIndividualScenarios.individual_optimizer import optimize_household


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
    price_grid_a_low=config.getfloat("tariffs", "grid_a_low_ft_per_kwh"),
    price_grid_a_high=config.getfloat("tariffs", "grid_a_high_ft_per_kwh"),
    price_grid_b_low=config.getfloat("tariffs", "grid_b_low_ft_per_kwh"),
    price_grid_b_high=config.getfloat("tariffs", "grid_b_high_ft_per_kwh"),
    price_pv_grid=config.getfloat("tariffs", "pv_export_ft_per_kwh"),
    grid_a_low_cap_kwh=config.getfloat("tariffs", "grid_a_low_limit_kwh"),
    grid_b_low_cap_kwh=config.getfloat("tariffs", "grid_b_low_limit_kwh"),
    boiler_tariff="B",
    objective="bill",
    run_lp=False,
    msg=False,
    gapRel=None,
    timeLimit=None,
    allow_grid_charge_for_min_soc=False,
):
    options = locals().copy()
    options["p_el_heater_fixed"] = options.pop("p_el_heater")
    result = optimize_household(**options, p_dhw=np.zeros_like(p_ue),
                               include_boiler=False, include_bess=True,
                               bess_terminal="cyclic",
                               fixed_pv_priority=True)
    # Preserve the BESS-only public result convention for fixed boiler loads.
    result["p_pv_elh"] = result["p_pv_elh_fixed"]
    result["p_grid_elh"] = result["p_grid_elh_fixed"]
    result["p_pv_load"] = result["p_pv_ue"] + result["p_pv_elh"]
    result["p_grid_load"] = result["p_grid_ue"] + result["p_grid_elh"]
    result["p_bess_load"] = result["p_bess_ue"] + result["p_bess_elh"]
    if boiler_tariff.upper().strip() == "A":
        result["p_grid_to_base"] = result["p_grid_import"]
        result["p_grid_to_boiler"] = np.zeros_like(p_ue, dtype=float)
    else:
        result["p_grid_to_base"] = result["p_grid_ue"] + result["p_grid_bess"]
    return result
