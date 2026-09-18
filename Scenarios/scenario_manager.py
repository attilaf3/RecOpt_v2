from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from Utility.configuration import config


ImplementationStatus = Literal["implemented", "partial", "unavailable"]


class ScenarioError(RuntimeError):
    pass


class ScenarioNotFoundError(ScenarioError):
    pass


class ScenarioIncompleteError(ScenarioError):
    pass


@dataclass(frozen=True)
class ScenarioDefinition:
    code: str
    optimization: str
    settlement: Literal["individual", "community"]
    boiler_control: str
    heat_pump_control: str
    bess_control: str
    base_tariff: str
    purpose: str
    devices: str
    objective: str | None
    sensitivities: tuple[str, ...]
    runner: str | None
    status: ImplementationStatus
    limitations: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.runner is not None and self.status != "unavailable"


@dataclass(frozen=True)
class ScenarioExecutionPlan:
    definition: ScenarioDefinition
    runner: str
    kwargs: dict[str, Any]


SENSITIVITIES = ("boiler_B_tariff", "heat_pump_GEO_tariff", "community_energy_price")


def _scenario(
    code: str,
    optimization: str,
    settlement: Literal["individual", "community"],
    boiler: str,
    hp: str,
    bess: str,
    tariff: str,
    purpose: str,
    objective: str | None,
    *,
    runner: str | None,
    status: ImplementationStatus,
    sensitivities: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
) -> ScenarioDefinition:
    return ScenarioDefinition(
        code=code,
        optimization=optimization,
        settlement=settlement,
        boiler_control=boiler,
        heat_pump_control=hp,
        bess_control=bess,
        base_tariff=tariff,
        purpose=purpose,
        devices="ALL" if code not in {"0-I", "0-K"} else "baseline",
        objective=objective,
        sensitivities=sensitivities,
        runner=runner,
        status=status,
        limitations=limitations,
    )


SCENARIOS: dict[str, ScenarioDefinition] = {
    item.code: item
    for item in (
        _scenario("0-I", "none; measured data", "individual", "baseline", "thermostat", "none", "B/GEO",
                  "Measured reference operation and cost.", None, runner="nonoptimized_individual", status="implemented"),
        _scenario("0-K", "none; measured data", "community", "baseline", "thermostat", "none", "B/GEO",
                  "Effect of community settlement without control.", None, runner="nonoptimized_community", status="partial",
                  limitations=("Community non-optimized execution does not yet model heat pumps.",)),
        _scenario("1d-I", "individual rule-based", "individual", "rule-based", "rule-based", "rule-based", "A/A",
                  "Uncoordinated household-level rule-based control.", None, runner="nonoptimized_individual", status="partial",
                  sensitivities=SENSITIVITIES,
                  limitations=("The current runner uses greedy BESS control and baseline boiler/thermostatic HP logic.",)),
        _scenario("1d-K", "individual rule-based", "community", "rule-based", "rule-based", "rule-based", "A/A",
                  "Local rule-based control with community settlement.", None, runner="nonoptimized_community", status="partial",
                  sensitivities=SENSITIVITIES,
                  limitations=("Community heat-pump and complete rule-based device control are not yet implemented.",)),
        _scenario("2d-K", "community rule-based", "community", "community rule-based", "community rule-based",
                  "community rule-based", "A/A", "Additional value of simple community coordination.", None,
                  runner=None, status="unavailable",
                  limitations=("No community rule-based controller is implemented.",)),
        _scenario("3d-I", "individual optimization", "individual", "optimized", "optimized", "optimized", "A/A",
                  "Household financial optimization of flexible devices.", "financial", runner="individual_optimized",
                  status="partial", sensitivities=SENSITIVITIES,
                  limitations=("The common individual model supports boiler and BESS; flexible heat-pump control is pending.",)),
        _scenario("3d-K", "individual optimization", "community", "optimized", "optimized", "optimized", "A/A",
                  "Individual optimization followed by community settlement.", "financial", runner="individual_optimized",
                  status="partial", sensitivities=SENSITIVITIES,
                  limitations=("Community settlement and combined boiler-BESS optimization are available; flexible heat-pump control is pending.",)),
        _scenario("4d-I", "community optimization", "individual", "community optimized", "community optimized",
                  "community optimized", "A/A", "Technical effect of community coordination under individual billing.",
                  "energy", runner="community_optimized", status="partial",
                  limitations=("Individual settlement is complete; the community optimizer currently covers BESS and boiler, not HP.",)),
        _scenario("4d-K-E", "community optimization", "community", "community optimized", "community optimized",
                  "community optimized", "A/A", "Technical and settlement value of full community coordination.",
                  "energy", runner="community_optimized", status="partial", sensitivities=SENSITIVITIES,
                  limitations=("The community optimizer currently covers BESS and boiler, not HP.",)),
        _scenario("4d-K-C", "community optimization", "community", "community optimized", "community optimized",
                  "community optimized", "A/A", "Financial value of full community coordination.",
                  "financial", runner="community_optimized", status="partial", sensitivities=SENSITIVITIES,
                  limitations=("The community optimizer currently covers BESS and boiler, not HP.",)),
        _scenario("5d-I", "aggregator optimization", "individual", "aggregator optimized", "aggregator optimized",
                  "aggregator optimized", "A/A", "Grid-service potential under individual billing.", "DSO",
                  runner=None, status="unavailable",
                  limitations=("No aggregator/DSO objective or network constraint model is implemented.",)),
        _scenario("5d-K", "aggregator optimization", "community", "aggregator optimized", "aggregator optimized",
                  "aggregator optimized", "A/A", "DSO-oriented operation with community settlement.", "DSO",
                  runner=None, status="unavailable", sensitivities=SENSITIVITIES,
                  limitations=("No aggregator/DSO objective or network constraint model is implemented.",)),
    )
}


def normalize_scenario_code(code: str) -> str:
    normalized = str(code).strip().upper().replace("_", "-")
    for registered in SCENARIOS:
        if registered.upper() == normalized:
            return registered
    raise ScenarioNotFoundError(
        f"Ismeretlen szcenáriókód: {code!r}. Érvényes kódok: {', '.join(SCENARIOS)}"
    )


def get_scenario(code: str) -> ScenarioDefinition:
    return SCENARIOS[normalize_scenario_code(code)]


def list_scenarios() -> tuple[ScenarioDefinition, ...]:
    return tuple(SCENARIOS.values())


def _runtime_defaults(code: str) -> dict[str, Any]:
    output_key = code.lower().replace("-", "_") + "_output"
    return {
        "sim_yaml": config.getpath("paths", "simulation_yaml"),
        "profiles_csv": config.getpath("paths", "profiles_csv"),
        "dhw_profiles_csv": config.getpath("paths", "dhw_profiles_csv"),
        "out_dir": config.getpath("scenario_outputs", output_key),
        "max_users": config.getint("scenario", "max_users"),
        "pv_ratio": config.getfloat("scenario", "pv_ratio"),
        "bess_share_pct": config.getfloat("scenario", "bess_share_pct"),
        "heat_pump_share_pct": config.getfloat("scenario", "heat_pump_share_pct"),
        "sharing_mode": config.getstring("scenario", "sharing_mode"),
        "run_lp": config.getboolean("scenario", "run_lp"),
        "allow_grid_charge_for_min_soc": config.getboolean("bess", "allow_grid_charge_for_min_soc", fallback=False),
    }


def build_execution_plan(code: str, **overrides: Any) -> ScenarioExecutionPlan:
    definition = get_scenario(code)
    if not definition.runnable or definition.runner is None:
        detail = " ".join(definition.limitations) or "Nincs hozzárendelt futtató."
        raise ScenarioIncompleteError(f"A(z) {definition.code} szcenárió még nem futtatható. {detail}")

    allowed_overrides = {
        "sim_yaml", "profiles_csv", "dhw_profiles_csv", "out_dir", "max_users",
        "pv_ratio", "bess_share_pct", "heat_pump_share_pct", "sharing_mode", "run_lp",
        "allow_grid_charge_for_min_soc",
    }
    unknown = set(overrides) - allowed_overrides
    if unknown:
        raise ScenarioError(f"Ismeretlen futtatási paraméter(ek): {', '.join(sorted(unknown))}")

    options = _runtime_defaults(definition.code)
    options.update({key: value for key, value in overrides.items() if value is not None})
    common = {
        "sim_yaml": options["sim_yaml"],
        "profiles_csv": options["profiles_csv"],
        "dhw_profiles_csv": options["dhw_profiles_csv"],
        "out_dir": options["out_dir"],
        "max_users": options["max_users"],
    }

    if definition.runner.startswith("nonoptimized"):
        is_community = definition.settlement == "community"
        is_reference = definition.code.startswith("0-")
        kwargs = {
            "case_name": definition.code,
            **common,
            "include_bess": not is_reference,
            "include_boiler": True,
            "include_heat_pump": not is_community,
            "heat_pump_share_pct": options["heat_pump_share_pct"],
            "bess_share_pct": 0.0 if is_reference else options["bess_share_pct"],
            "boiler_tariff": "B" if is_reference else "A",
            "pv_ratio": options["pv_ratio"],
            "scenario": "b" if is_community else "a",
            "allow_grid_charge_for_min_soc": options["allow_grid_charge_for_min_soc"],
            "sharing_mode": options["sharing_mode"],
        }
    elif definition.runner == "individual_optimized":
        kwargs = {
            **common,
            "include_bess": True,
            "allow_grid_charge_for_min_soc": options["allow_grid_charge_for_min_soc"],
            "bess_share_pct": options["bess_share_pct"],
            "pv_ratio": options["pv_ratio"],
            "run_lp": options["run_lp"],
            "boiler_tariff": "A",
            "settlement_mode": definition.settlement,
            "sharing_mode": options["sharing_mode"],
        }
    elif definition.runner == "community_optimized":
        kwargs = {
            "allow_grid_charge_for_min_soc": options["allow_grid_charge_for_min_soc"],
            "case_name": definition.code,
            **common,
            "include_bess": True,
            "bess_share_pct": options["bess_share_pct"],
            "boiler_tariff": "A",
            "sharing_mode": options["sharing_mode"],
            "objective": "grid" if definition.objective == "energy" else "bill",
            "run_lp": options["run_lp"],
            "pv_ratio": options["pv_ratio"],
            "settlement_mode": definition.settlement,
        }
    else:
        raise ScenarioIncompleteError(f"A(z) {definition.code} futtatója nincs bekötve.")

    return ScenarioExecutionPlan(definition=definition, runner=definition.runner, kwargs=kwargs)


def _resolve_runner(runner: str) -> Callable[..., dict]:
    if runner.startswith("nonoptimized"):
        from NotOptimizedScenarios.nonopt_common import run_case

        return run_case
    if runner == "individual_optimized":
        from OptimizedScenarios.run_individual_optimization import run

        return run
    if runner == "community_optimized":
        from OptimizedCommunityScenarios.call_disaggregated import run_case

        return run_case
    raise ScenarioIncompleteError(f"Ismeretlen futtató: {runner}")


def run_scenario(code: str, *, allow_partial: bool = False, **overrides: Any) -> dict:
    plan = build_execution_plan(code, **overrides)
    definition = plan.definition
    if definition.status == "partial" and not allow_partial:
        raise ScenarioIncompleteError(
            f"A(z) {definition.code} csak részlegesen támogatott. "
            f"{' '.join(definition.limitations)} Használd az allow_partial=True kapcsolót a tudatos futtatáshoz."
        )

    result = _resolve_runner(plan.runner)(**plan.kwargs)
    summary = dict(result)
    summary.update(
        {
            "scenario_code": definition.code,
            "implementation_status": definition.status,
            "optimization": definition.optimization,
            "settlement": definition.settlement,
            "objective_type": definition.objective,
            "limitations": list(definition.limitations),
        }
    )
    out_dir = Path(plan.kwargs["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "definition": asdict(definition),
                "runtime": {key: str(value) if isinstance(value, Path) else value for key, value in plan.kwargs.items()},
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary
