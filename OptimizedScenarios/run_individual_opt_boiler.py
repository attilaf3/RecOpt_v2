"""Compatibility entry point for the common individual optimization runner."""
from OptimizedScenarios.run_individual_optimization import build_inputs, main, run, run_3d_k

__all__ = ["build_inputs", "main", "run", "run_3d_k"]

if __name__ == "__main__":
    import runpy
    runpy.run_module("OptimizedScenarios.run_individual_optimization", run_name="__main__")
