import importlib.util
import unittest

PULP_AVAILABLE = importlib.util.find_spec("pulp") is not None
if PULP_AVAILABLE:
    import pulp

    from OptimizedIndividualScenarios import opt_bess, opt_boiler, opt_hp
    from OptimizedIndividualScenarios.optimization_constraints import (
        add_import_export_exclusivity,
        add_two_tier_step_constraints,
        create_two_tier_tariff_block,
    )
    from Optimization import extract_matrix, extract_vector, variable_value


@unittest.skipUnless(PULP_AVAILABLE, "PuLP is not installed in this Python environment")
class OptimizationCompatibilityTests(unittest.TestCase):
    def test_device_modules_delegate_to_working_optimizers(self):
        self.assertIs(opt_bess.opt_bess, opt_bess.optimize_bess)
        self.assertIs(opt_boiler.opt_boiler, opt_boiler.optimize_boiler)
        self.assertIs(opt_hp.opt_hp, opt_hp.optimize_hp)
        self.assertIs(opt_bess.add_bess_constraints, opt_bess.optimize_bess)
        self.assertIs(opt_boiler.add_boiler_constraints, opt_boiler.optimize_boiler)
        self.assertTrue(callable(opt_bess.individual_opt_bess))
        self.assertTrue(callable(opt_boiler.individual_opt_boiler))
        self.assertTrue(callable(opt_hp.individual_opt_hp))

    def test_two_tier_tariff_constraints_are_created_consistently(self):
        problem = pulp.LpProblem("tariff_test", pulp.LpMinimize)
        block = create_two_tier_tariff_block(2, 3.0, "test_energy")
        imports = [pulp.LpVariable(f"import_{t}", lowBound=0) for t in range(2)]

        for step in range(2):
            add_two_tier_step_constraints(
                problem, block, step, imports[step], 3.0, "test_tariff"
            )

        self.assertEqual(len(problem.constraints), 6)
        self.assertIn("test_tariff_remaining_low_init", problem.constraints)
        self.assertIn("test_tariff_remaining_low_balance_1", problem.constraints)

    def test_import_export_gate_supports_both_selector_conventions(self):
        problem = pulp.LpProblem("gate_test", pulp.LpMinimize)
        import_flow = pulp.LpVariable("import_flow", lowBound=0)
        export_flow = pulp.LpVariable("export_flow", lowBound=0)
        selector = pulp.LpVariable("direction", cat=pulp.LpBinary)

        add_import_export_exclusivity(
            problem,
            import_flow,
            export_flow,
            selector,
            10.0,
            8.0,
            0,
            prefix="normal",
        )
        add_import_export_exclusivity(
            problem,
            import_flow,
            export_flow,
            selector,
            10.0,
            8.0,
            0,
            prefix="inverse",
            selector_is_import=False,
        )

        self.assertEqual(len(problem.constraints), 4)
        self.assertIn("normal_import_gate_0", problem.constraints)
        self.assertIn("inverse_export_gate_0", problem.constraints)

    def test_common_result_extractors_handle_scalars_vectors_and_matrices(self):
        first = pulp.LpVariable("first")
        second = pulp.LpVariable("second")
        first.varValue = 1.25
        second.varValue = 2.5

        self.assertEqual(variable_value(first), 1.25)
        self.assertEqual(variable_value(pulp.LpVariable("unset")), 0.0)
        self.assertEqual(extract_vector([first, second]).tolist(), [1.25, 2.5])
        self.assertEqual(
            extract_matrix([[first, second], [0.0, first]]).tolist(),
            [[1.25, 2.5], [0.0, 1.25]],
        )


if __name__ == "__main__":
    unittest.main()
