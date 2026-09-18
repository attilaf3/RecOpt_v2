"""Numerical invariants for the composed household model (CBC, short horizons)."""
import importlib.util
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

PULP_AVAILABLE = importlib.util.find_spec("pulp") is not None
if PULP_AVAILABLE:
    from Optimization.runtime import solve_problem
    from OptimizedIndividualScenarios.individual_optimizer import optimize_household
    from OptimizedIndividualScenarios.individual_opt_boiler import individual_opt_boiler
    from OptimizedIndividualScenarios.individual_opt_bess import individual_opt_bess
    from OptimizedIndividualScenarios.individual_opt_bess_boiler import individual_opt_bess_boiler


@unittest.skipUnless(PULP_AVAILABLE, "PuLP required")
class IndividualOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.solver = patch(
            "OptimizedIndividualScenarios.individual_optimizer.solve_problem",
            side_effect=lambda p, **kw: solve_problem(p, solver_name="CBC", **kw))
        self.solver.start()
        self.addCleanup(self.solver.stop)

    def assert_balance(self, r):
        np.testing.assert_allclose(
            r["p_pv"]+r["p_grid_import"]+r["p_bess_out"],
            r["p_ue"]+r["p_el_heater_total"]+r["p_grid_export"]+r["p_bess_in"],
            atol=1e-6)
        self.assertAlmostEqual(r["grid_import_a_kwh"]+r["grid_import_b_kwh"],
                               r["p_grid_import"].sum()*r["dt"], places=6)

    def test_no_devices_and_boiler_wrapper(self):
        args = dict(p_pv=[0,2,0], p_ue=[1,1,1], p_dhw=[0,0,0], msg=False)
        plain = optimize_household(**args, include_boiler=False)
        wrapped = individual_opt_boiler(**args)
        self.assert_balance(plain)
        self.assertAlmostEqual(plain["grid_import_a_kwh"], 0.5)
        self.assertAlmostEqual(plain["grid_export_kwh"], 0.25)
        self.assertAlmostEqual(plain["net_cost_Ft"], wrapped["net_cost_Ft"])

    def test_combined_model_both_tariffs_and_objectives(self):
        for tariff in ("A", "B"):
            for objective in ("bill", "grid"):
                with self.subTest(tariff=tariff, objective=objective):
                    r = optimize_household(
                        [3,3,0,0], [1]*4, [0.2]*4, msg=False,
                        include_bess=True, size_bess=2, soc_bess_init=0.1,
                        size_elh=2, vol_hss_water=100, T_min=38, T_max=65,
                        T_env=20, a_hss=0.001, enforce_cl_rules=False,
                        bess_min_mode_steps=1,
                        boiler_tariff=tariff, objective=objective)
                    self.assert_balance(r)
                    states = r["e_bess_boundary"]
                    np.testing.assert_allclose(np.diff(states),
                        states[:-1]*(0.995-1)+0.25*(r["p_bess_in"]*0.98-r["p_bess_out"]/0.96),
                        atol=1e-6)
                    self.assertAlmostEqual(states[0], states[-1], places=6)
                    if tariff == "B":
                        np.testing.assert_allclose(r["p_pv_to_boiler"], 0, atol=1e-6)
                        np.testing.assert_allclose(r["p_bess_to_boiler"], 0, atol=1e-6)

    def test_cyclic_horizon_cannot_consume_initial_energy(self):
        r = individual_opt_bess([0], [10], size_bess=1,
            soc_bess_min=0, soc_bess_init=0.5, eta_bess_stor=1,
            eta_bess_out=1, eta_bess_in=1, dt=1, t_bess_min=0.1)
        self.assert_balance(r)
        self.assertAlmostEqual(r["p_bess_out"][0], 0, places=6)
        self.assertAlmostEqual(r["p_grid_import"][0], 10, places=6)
        self.assertAlmostEqual(r["e_bess_boundary"][0], r["e_bess_boundary"][-1])

    def test_combined_compatibility_entry_point(self):
        r = individual_opt_bess_boiler([0,2], [1,1], [0,0], msg=False)
        self.assert_balance(r)
        self.assertEqual(r["objective_unit"], "kWh")

    def test_hp_and_unknown_objectives_fail_explicitly(self):
        with self.assertRaises(NotImplementedError):
            optimize_household([0], [1], [0], include_heat_pump=True)
        with self.assertRaises(ValueError):
            optimize_household([0], [1], [0], objective="DSO")

    def test_energy_objective_reports_actual_tariff_blocks(self):
        r = optimize_household([0,0], [4,4], [0,0], dt=1,
            objective="grid", grid_a_low_cap_kwh=5,
            price_grid_a_low=10, price_grid_a_high=20, msg=False)
        self.assertAlmostEqual(r["import_cost_ft"], 110)
        np.testing.assert_allclose(r["e_grid_a_low_step"], [4,1])

    @unittest.skipUnless(importlib.util.find_spec("yaml") is not None, "PyYAML required")
    def test_common_runner_preserves_dispatch_between_settlements(self):
        from OptimizedScenarios.run_individual_optimization import run
        import pandas as pd
        zero = np.zeros(2)
        one = np.ones(2)
        inputs = SimpleNamespace(
            p_pv_kw=np.array([[4,0],[4,0],[0,0],[0,0]], dtype=float),
            p_ue_kw=np.ones((4,2)), p_dhw_kw=np.zeros((4,2)),
            p_el_heater_kw=np.zeros((4,2)), size_elh=zero,
            vol_hss_water=zero, T_env=one*20, T_max=one*65,
            T_min=one*38, T_in=one*10, a_hss=zero, eta_elh=one,
            t_hss_min_in=one, user_names=["a", "b"], size_bess=one*2,
            eta_bess_in=one, eta_bess_out=one, eta_bess_stor=one,
            soc_bess_min=zero, soc_bess_max=one, t_bess_min=one)
        with tempfile.TemporaryDirectory() as directory, patch(
            "OptimizedScenarios.run_individual_optimization.read_simulation_inputs",
            return_value=inputs):
            results = []
            for mode in ("individual", "community"):
                out = Path(directory)/mode
                summary = run("unused", "unused", "unused", out,
                    include_bess=True, run_lp=False, boiler_tariff="A",
                    settlement_mode=mode)
                self.assertTrue(summary["include_bess"])
                results.append(pd.read_csv(out/"timeseries_a.csv"))
            for key in ("p_grid_import", "p_grid_export", "p_bess_in", "p_bess_out"):
                np.testing.assert_allclose(results[0][key], results[1][key])
