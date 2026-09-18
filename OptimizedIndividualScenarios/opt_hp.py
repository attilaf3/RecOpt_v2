"""Backward-compatible entry points for the implemented HP optimizer.

The underlying model optimizes PV/grid supply for a fixed heat-pump electric
profile. It does not yet shift heat-pump operation using a building thermal
state model.
"""

from OptimizedIndividualScenarios.individual_opt_hp import (
    hp_profiles_from_temperature,
    individual_opt_hp,
)


def optimize_hp(*args, **kwargs):
    """Delegate to :func:`individual_opt_hp`."""
    return individual_opt_hp(*args, **kwargs)


opt_hp = optimize_hp

__all__ = [
    "hp_profiles_from_temperature",
    "individual_opt_hp",
    "opt_hp",
    "optimize_hp",
]
