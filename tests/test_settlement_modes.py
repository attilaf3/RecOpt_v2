import unittest

import numpy as np

from Economics import (
    allocate_virtual_community_energy,
    settle_community_optimization_as_individual,
    settle_individual_optimization_as_community,
)


class SettlementModeTests(unittest.TestCase):
    def setUp(self):
        self.import_a = np.array([[4.0, 0.0], [0.0, 2.0]])
        self.import_b = np.zeros_like(self.import_a)
        self.export = np.array([[0.0, 3.0], [1.0, 0.0]])
        self.names = ["buyer_a", "buyer_b"]

    def test_3d_k_preserves_dispatch_and_balances_shared_energy(self):
        result = settle_individual_optimization_as_community(
            self.import_a,
            self.import_b,
            self.export,
            user_names=self.names,
        )
        ts = result["timeseries"]
        np.testing.assert_allclose(
            self.import_a,
            ts["e_grid_import_a"] + ts["e_shared_in"],
        )
        np.testing.assert_allclose(
            self.export,
            ts["e_grid_export"] + ts["e_shared_out"],
        )
        np.testing.assert_allclose(
            ts["e_shared_in"].sum(axis=1),
            ts["e_shared_out"].sum(axis=1),
        )
        self.assertEqual(result["scenario_code"], "3d-K")

    def test_4d_i_reclassifies_internal_flows_to_individual_meters(self):
        community = allocate_virtual_community_energy(self.import_a, self.export)
        result = settle_community_optimization_as_individual(
            e_grid_import_a=community["e_grid_import_a"],
            e_grid_import_b=self.import_b,
            e_grid_export=community["e_grid_export"],
            e_shared_to_a=community["e_shared_in"],
            e_shared_to_b=self.import_b,
            e_shared_out=community["e_shared_out"],
            user_names=self.names,
        )
        np.testing.assert_allclose(result["timeseries"]["e_grid_import_a"], self.import_a)
        np.testing.assert_allclose(result["timeseries"]["e_grid_export"], self.export)
        self.assertEqual(result["summary"]["shared_in_kwh"], 0.0)
        self.assertEqual(result["summary"]["shared_revenue_ft"], 0.0)
        self.assertEqual(result["scenario_code"], "4d-I")

    def test_equal_allocation_respects_participant_limits(self):
        imports = np.array([[1.0, 4.0, 4.0]])
        exports = np.array([[0.0, 0.0, 6.0]])
        result = allocate_virtual_community_energy(imports, exports, allocation_mode="equal")
        np.testing.assert_allclose(result["e_shared_in"], [[1.0, 2.5, 2.5]])
        self.assertTrue(np.all(result["e_shared_in"] <= imports + 1e-12))


if __name__ == "__main__":
    unittest.main()
