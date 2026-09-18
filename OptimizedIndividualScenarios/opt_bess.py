"""Backward-compatible entry points for the implemented BESS optimizer."""

from OptimizedIndividualScenarios.individual_opt_bess import individual_opt_bess


def optimize_bess(*args, **kwargs):
    """Delegate to :func:`individual_opt_bess`."""
    return individual_opt_bess(*args, **kwargs)


opt_bess = optimize_bess
add_bess_constraints = optimize_bess

__all__ = [
    "add_bess_constraints",
    "individual_opt_bess",
    "opt_bess",
    "optimize_bess",
]
