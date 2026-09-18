import importlib.util
import unittest
import tempfile
from pathlib import Path
import numpy as np

READY = all(importlib.util.find_spec(name) is not None for name in ("pulp", "yaml"))
if READY:
    from Optimization.runtime import OptimizationSolveError
    from OptimizedIndividualScenarios.individual_optimizer import optimize_household
    from OptimizedCommunityScenarios.optimize_disaggregated import (
        disaggregated_opt_bess_shared, settle_shared_payments, save_disaggregated_opt_results)
    from NotOptimizedScenarios.noopt_community_1b import (
        simulate_nonopt_disaggregated_shared, settle_shared_payments_buyer_tiered,
        _simulate_nonopt_disaggregated_shared)
    from NotOptimizedScenarios.nonopt_common import simulate_one_user_greedy, _simulate_one_user_greedy
    from Economics.calculate_economics import settle_shared_payments as settle


@unittest.skipUnless(READY, "PuLP and PyYAML required")
class BessUnificationTests(unittest.TestCase):
    def test_minimum_guard_is_opt_in_and_exact_even_in_lp_mode(self):
        for lp in (False, True):
            kwargs = dict(include_boiler=False, include_bess=True, size_bess=1,
                soc_bess_min=0.1, soc_bess_init=0.1, eta_bess_stor=0.9,
                eta_bess_in=1, eta_bess_out=1, dt=1, run_lp=lp, msg=False)
            with self.assertRaises(OptimizationSolveError):
                optimize_household([0], [1], [0], **kwargs)
            result = optimize_household([0], [1], [0], **kwargs,
                allow_grid_charge_for_min_soc=True)
            np.testing.assert_allclose(result["p_grid_bess"], [0.01], atol=1e-6)
            np.testing.assert_allclose(result["e_bess_boundary"], [0.1,0.1], atol=1e-6)

    def test_community_uses_cyclic_component_and_default_no_grid_charge(self):
        result = disaggregated_opt_bess_shared(
            np.array([[3.,0],[0,0]]), np.ones((2,2)), None,
            size_bess=np.array([1.,0]), eta_bess_stor=np.ones(2),
            soc_bess_min=np.zeros(2), bess_min_mode_steps=1)
        ts = result["timeseries"]
        np.testing.assert_allclose(ts["e_bess_boundary"][0], ts["e_bess_boundary"][-1])
        np.testing.assert_allclose(ts["p_grid_bess"], 0)
        self.assertTrue(np.all(ts["d_bess_ch"]+ts["d_bess_dis"] <= 1+1e-6))
        states = ts["e_bess_boundary"]
        np.testing.assert_allclose(np.diff(states, axis=0),
            0.25*(0.98*ts["p_bess_in"]-ts["p_bess_out"]/0.96), atol=1e-6)
        with tempfile.TemporaryDirectory() as folder:
            save_disaggregated_opt_results(result, folder)
            # Header plus T+1 boundary states; per-step exports remain T rows.
            self.assertEqual(len((Path(folder)/"e_bess_boundary.csv").read_text().splitlines()), 4)
            self.assertEqual(len((Path(folder)/"p_bess_in.csv").read_text().splitlines()), 3)
            self.assertTrue((Path(folder)/"bess"/"schema.json").is_file())

    def test_open_horizon_and_unrestricted_grid_charge_rejected(self):
        with self.assertRaises(ValueError):
            optimize_household([0], [1], [0], bess_terminal="open")
        with self.assertRaises(ValueError):
            optimize_household([0], [1], [0], bess_grid_charge="unrestricted")

    def test_greedy_individual_and_community_are_periodic(self):
        z = np.zeros(8)
        pv = np.array([1,1,1,1,0,0,0,0], dtype=float)
        load = np.ones(8)*0.2
        individual = simulate_one_user_greedy(load,z,z,pv,True,1,1,1,1,0,1,1)
        community = simulate_nonopt_disaggregated_shared(
            pv[:,None], load[:,None], z[:,None], np.ones(1),
            np.ones(1),np.ones(1),np.ones(1),np.zeros(1),np.ones(1),
            np.ones(1),["a"])
        for result in (individual, community):
            ts = result["timeseries"]
            np.testing.assert_allclose(ts["e_bess_boundary"][0], ts["e_bess_boundary"][-1], atol=1e-7)
            np.testing.assert_allclose(ts["e_grid_to_bess"], 0)
            np.testing.assert_allclose(ts["e_bess"], ts["e_bess_boundary"][:-1])
            np.testing.assert_allclose(ts["e_bess_end"], ts["e_bess_boundary"][1:])
            np.testing.assert_allclose(ts["e_bess_in"], 0.25*ts["p_bess_in"])

    def test_both_greedy_kernels_use_pv_before_minimum_grid_topup(self):
        z = np.zeros(1)
        individual = _simulate_one_user_greedy(z,z,z,np.array([0.02]),
            True,1,1,1,0.9,0.1,1,1, initial_soc_kwh=0.1,
            allow_grid_charge_for_min_soc=True)
        community = _simulate_nonopt_disaggregated_shared(
            np.array([[0.02]]),z[:,None],z[:,None],np.ones(1),np.ones(1),
            np.ones(1),np.array([0.9]),np.array([0.1]),np.ones(1),np.ones(1),
            ["a"],initial_soc_fraction=0.1,allow_grid_charge_for_min_soc=True)
        for result in (individual, community):
            np.testing.assert_allclose(result["timeseries"]["e_grid_to_bess"], 0)
            np.testing.assert_allclose(result["timeseries"]["e_bess"], 0.11)

    def test_disabled_bess_has_complete_state_schema(self):
        z = np.zeros(2)
        result = simulate_one_user_greedy(z,z,z,z,False,0,1,1,1,0,1,1)
        self.assertEqual(result["timeseries"]["e_bess_boundary"].shape, (3,))
        result = simulate_nonopt_disaggregated_shared(z[:,None],z[:,None],z[:,None],
            np.ones(1),np.ones(1),np.ones(1),np.ones(1),np.zeros(1),np.ones(1),
            np.ones(1),["a"],include_bess=False)
        self.assertEqual(result["timeseries"]["e_bess_boundary"].shape, (3,1))

    def test_settlement_adapters_preserve_units_and_block_policy(self):
        grid = np.array([[2522.5,0],[1,0]], dtype=float)
        shared = np.array([[0,0],[1,0]], dtype=float)
        sold = np.array([[0,0],[0,1]], dtype=float)
        proportional = settle_shared_payments_buyer_tiered(shared,sold,grid)
        expected = settle(shared,sold,grid,shared_low_cap_mode="proportional")
        np.testing.assert_allclose(proportional["grid_a_low_kwh"], expected["buyer_grid_a_low_kwh"])
        grid_first = settle_shared_payments(shared/0.25,sold/0.25,grid/0.25,dt=0.25)
        expected = settle(shared,sold,grid,shared_low_cap_mode="grid_first")
        for key in ("pair_kwh","seller_revenue_ft","buyer_energy_cost_ft","buyer_shared_low_kwh"):
            np.testing.assert_allclose(grid_first[key], expected[key])
        self.assertNotEqual(proportional["buyer_energy_cost_ft"][0], grid_first["buyer_energy_cost_ft"][0])
