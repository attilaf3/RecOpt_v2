import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Scenarios import (
    ScenarioError,
    ScenarioIncompleteError,
    build_execution_plan,
    get_scenario,
    list_scenarios,
    normalize_scenario_code,
    run_scenario,
)


EXPECTED_CODES = (
    "0-I", "0-K", "1d-I", "1d-K", "2d-K", "3d-I", "3d-K",
    "4d-I", "4d-K-E", "4d-K-C", "5d-I", "5d-K",
)


class ScenarioManagerTests(unittest.TestCase):
    def test_registry_contains_every_official_code(self):
        self.assertEqual(tuple(item.code for item in list_scenarios()), EXPECTED_CODES)
        self.assertEqual(normalize_scenario_code("4D_k_c"), "4d-K-C")

    def test_0_i_maps_to_individual_baseline(self):
        plan = build_execution_plan("0-I")
        self.assertEqual(plan.runner, "nonoptimized_individual")
        self.assertEqual(plan.kwargs["scenario"], "a")
        self.assertFalse(plan.kwargs["include_bess"])
        self.assertTrue(plan.kwargs["include_heat_pump"])
        self.assertEqual(plan.kwargs["boiler_tariff"], "B")

    def test_3d_k_maps_optimization_to_community_settlement(self):
        plan = build_execution_plan("3d-k")
        self.assertEqual(plan.runner, "individual_optimized")
        self.assertEqual(plan.kwargs["settlement_mode"], "community")
        self.assertEqual(plan.kwargs["boiler_tariff"], "A")

    def test_4d_variants_select_correct_objective_and_settlement(self):
        plan_i = build_execution_plan("4d-I")
        plan_e = build_execution_plan("4d-K-E")
        plan_c = build_execution_plan("4d-K-C")
        self.assertEqual((plan_i.kwargs["objective"], plan_i.kwargs["settlement_mode"]), ("grid", "individual"))
        self.assertEqual((plan_e.kwargs["objective"], plan_e.kwargs["settlement_mode"]), ("grid", "community"))
        self.assertEqual((plan_c.kwargs["objective"], plan_c.kwargs["settlement_mode"]), ("bill", "community"))

    def test_unavailable_and_partial_scenarios_are_guarded(self):
        with self.assertRaises(ScenarioIncompleteError):
            build_execution_plan("5d-K")
        with self.assertRaises(ScenarioIncompleteError):
            run_scenario("3d-K")

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(ScenarioError):
            build_execution_plan("0-I", typo_option=True)

    def test_successful_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("Scenarios.scenario_manager._resolve_runner", return_value=lambda **kwargs: {"ok": True}):
                result = run_scenario("0-I", out_dir=directory, max_users=2)
            self.assertTrue(result["ok"])
            manifest = json.loads((Path(directory) / "scenario_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["definition"]["code"], "0-I")
            self.assertEqual(manifest["runtime"]["max_users"], 2)


if __name__ == "__main__":
    unittest.main()
