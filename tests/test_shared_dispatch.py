import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from Utility.bess_dispatch import pv_charge, load_discharge, minimum_grid_charge
from Utility.energy_allocation import allocate_pool, match_energy
from Utility.result_schema import add_bess_states, add_bess_flows, save_bess_result


class SharedDispatchTests(unittest.TestCase):
    def test_pv_precedes_grid_and_combined_charge_is_limited(self):
        soc = 0.09
        pv = pv_charge(soc, 0.02, 1, 1, 0.25)
        self.assertEqual(minimum_grid_charge(soc+pv, 0.1, 1, 0.25, pv, enabled=True), 0)
        pv = pv_charge(soc, 0.004, 1, 1, 0.005)
        grid = minimum_grid_charge(soc+pv, 0.1, 1, 0.005, pv, enabled=True)
        self.assertAlmostEqual(pv+grid, 0.005)
        self.assertEqual(minimum_grid_charge(0, 0.1, 1, 1), 0)
        self.assertEqual(minimum_grid_charge(0, 0.1, 1, 1, enabled=True, allowed=False), 0)

    def test_discharge_efficiency_and_mode_lock(self):
        self.assertAlmostEqual(load_discharge(0.5, 1, 0.1, 0.8, 1), 0.32)
        self.assertEqual(load_discharge(0.5, 1, 0.1, 0.8, 1, False), 0)

    def test_allocation_capacity_balance_and_policies(self):
        np.testing.assert_allclose(allocate_pool([1, 3, 0], 3, "equal"), [1, 2, 0])
        np.testing.assert_allclose(allocate_pool([1, 3, 0], 3), [0.75, 2.25, 0])
        for buyer in ("equal", "proportional"):
            for seller in ("equal", "proportional"):
                received, sent = match_energy(np.array([1, 3, 0]), np.array([0, 0, 3]), buyer, seller)
                self.assertAlmostEqual(received.sum(), sent.sum())
                self.assertTrue(np.all(received <= [1, 3, 0]))
        with self.assertRaises(ValueError):
            allocate_pool([0], 0, "invalid")

    def test_canonical_state_and_flow_units(self):
        ts = {"e_pv_to_bess": np.array([0.1, 0]), "e_grid_to_bess": np.zeros(2),
              "e_bess_to_load": np.array([0, 0.1])}
        add_bess_states(ts, [0.2, 0.3, 0.2])
        add_bess_flows(ts, 0.25)
        np.testing.assert_allclose(ts["e_bess"], [0.2, 0.3])
        np.testing.assert_allclose(ts["e_bess_end"], [0.3, 0.2])
        np.testing.assert_allclose(ts["p_bess_in"], [0.4, 0])
        np.testing.assert_allclose(ts["p_bess_out"], [0, 0.4])
        with tempfile.TemporaryDirectory() as folder:
            save_bess_result(ts, folder, 0.25, ["home"])
            meta = json.loads((Path(folder)/"bess"/"schema.json").read_text())
            self.assertEqual(meta["dt_hours"], 0.25)
            self.assertEqual(meta["units"]["e_bess_start"], "kWh")
            self.assertEqual(meta["units"]["e_bess_in"], "kWh/step")
            self.assertEqual(meta["steps"], 2)
            with self.assertRaises(ValueError):
                save_bess_result(ts, folder, 0.25, ["wrong", "shape"])
