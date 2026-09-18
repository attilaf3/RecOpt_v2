"""Common solver selection, status validation and PuLP result extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pulp

from Utility.configuration import config


class OptimizationSolveError(RuntimeError):
    """Raised when no solver succeeds or the model has no accepted solution."""


@dataclass(frozen=True)
class SolveResult:
    status_code: int
    status: str
    solver: str


def variable_value(value: Any, default: float = 0.0) -> float:
    """Return a numeric PuLP value, using ``default`` for an unset variable."""
    resolved = pulp.value(value)
    return float(default) if resolved is None else float(resolved)


def extract_vector(values: Iterable[Any], default: float = 0.0) -> np.ndarray:
    """Extract a one-dimensional sequence of PuLP expressions as floats."""
    return np.asarray([variable_value(value, default) for value in values], dtype=float)


def extract_matrix(values: Sequence[Sequence[Any]], default: float = 0.0) -> np.ndarray:
    """Extract a rectangular matrix of PuLP expressions as a float array."""
    rows = [[variable_value(value, default) for value in row] for row in values]
    return np.asarray(rows, dtype=float)


def _configured_statuses() -> frozenset[str]:
    raw = config.getstring("optimization", "accepted_statuses", fallback="Optimal,Integer Feasible")
    statuses = frozenset(item.strip() for item in raw.split(",") if item.strip())
    if not statuses:
        raise ValueError("optimization.accepted_statuses must contain at least one status")
    return statuses


def _solver_candidates(requested: str | None, allow_fallback: bool) -> list[str]:
    selected = (requested or config.getstring("optimization", "solver", fallback="auto")).strip().upper()
    preferred = config.getstring("optimization", "preferred_solver", fallback="GUROBI").strip().upper()
    fallback = config.getstring("optimization", "fallback_solver", fallback="CBC").strip().upper()

    candidates = [preferred] if selected == "AUTO" else [selected]
    if allow_fallback and fallback not in candidates:
        candidates.append(fallback)
    return candidates


def _create_solver(name: str, *, msg: bool, gap_rel: float | None, time_limit: float | None):
    kwargs: dict[str, Any] = {"msg": bool(msg)}
    if gap_rel is not None:
        kwargs["gapRel"] = float(gap_rel)
    if time_limit is not None:
        kwargs["timeLimit"] = float(time_limit)

    if name == "GUROBI":
        return pulp.GUROBI_CMD(**kwargs)
    if name in {"CBC", "PULP_CBC_CMD"}:
        return pulp.PULP_CBC_CMD(**kwargs)
    raise ValueError(f"Unsupported solver: {name}. Supported values: auto, GUROBI, CBC")


def _is_available(solver: Any) -> bool:
    try:
        return bool(solver.available())
    except Exception:
        return False


def solve_problem(
    problem: pulp.LpProblem,
    *,
    solver_name: str | None = None,
    msg: bool = False,
    gap_rel: float | None = None,
    time_limit: float | None = None,
    accepted_statuses: Iterable[str] | None = None,
    allow_fallback: bool | None = None,
) -> SolveResult:
    """Solve a PuLP problem with configured GUROBI/CBC selection and fallback."""
    if allow_fallback is None:
        allow_fallback = config.getboolean("optimization", "allow_fallback", fallback=True)
    accepted = frozenset(accepted_statuses) if accepted_statuses is not None else _configured_statuses()

    failures: list[str] = []
    for candidate in _solver_candidates(solver_name, allow_fallback):
        try:
            solver = _create_solver(
                candidate,
                msg=msg,
                gap_rel=gap_rel,
                time_limit=time_limit,
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"{candidate}: {exc}")
            continue

        if not _is_available(solver):
            failures.append(f"{candidate}: unavailable")
            continue

        try:
            status_code = problem.solve(solver)
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
            continue

        status = pulp.LpStatus.get(status_code, str(status_code))
        if status not in accepted:
            raise OptimizationSolveError(
                f"Optimization failed with solver {candidate}: {status}. "
                f"Accepted statuses: {', '.join(sorted(accepted))}"
            )
        return SolveResult(status_code=int(status_code), status=status, solver=candidate)

    detail = "; ".join(failures) if failures else "no solver candidates"
    raise OptimizationSolveError(f"No configured optimization solver could run: {detail}")
