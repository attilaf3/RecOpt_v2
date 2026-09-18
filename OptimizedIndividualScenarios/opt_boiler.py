"""Backward-compatible entry points for the implemented boiler optimizer."""

from OptimizedIndividualScenarios.individual_opt_boiler import individual_opt_boiler


def optimize_boiler(*args, **kwargs):
    """Delegate to :func:`individual_opt_boiler`."""
    return individual_opt_boiler(*args, **kwargs)


opt_boiler = optimize_boiler
add_boiler_constraints = optimize_boiler

__all__ = [
    "add_boiler_constraints",
    "individual_opt_boiler",
    "opt_boiler",
    "optimize_boiler",
]

