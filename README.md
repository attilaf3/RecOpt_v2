# EnergyCommunityFlexible

## Aggregate heat pump profile

This repo includes a script to simulate 100 perturbed houses, aggregate their heat pump electric profiles, and compare the sum to the measured geoprofile.

### Quick run

```powershell
python aggregate_hp_profile.py
```

### Options

- `--num-houses` number of simulated houses (default: 100)
- `--seed` random seed for perturbations (default: random)
- `--progress-every` progress message interval in houses (default: 10)
- `--geoprofil` path to the comparison CSV
- `--output-csv` path to save the aggregated profile as CSV
- `--no-seasonal-plots` disable seasonal example plots
- `--smoke` run a small synthetic smoke test without plotting

Example:

```powershell
python aggregate_hp_profile.py --num-houses 100 --seed 123 --output-csv sum_profile.csv
```

Example with seasonal plots disabled:

```powershell
python aggregate_hp_profile.py --no-seasonal-plots
```

## Non-optimized community case with heat pumps

Call `run_case(...)` from Python and enable the new heat-pump option:

```powershell
python -c "from NotOptimizedScenarios.nonopt_common import run_case; run_case(case_name='demo_hp', sim_yaml=r'PATH\\TO\\sim.yaml', profiles_csv=r'PATH\\TO\\profiles.csv', dhw_profiles_csv=r'PATH\\TO\\dhw_profiles.csv', out_dir=r'PATH\\TO\\output', max_users=100, pv_ratio=1.0, include_bess=True, include_boiler=False, include_heat_pump=True, heat_pump_share_pct=30.0, bess_share_pct=100.0)"
```

The HP households are selected randomly, and the output folder also includes heat-pump-specific CSVs and plots.

## Non-optimized runner script

Use `run_noopt.py` when you want a simple constant-based entry point for `run_case(...)`.

Edit the constants near the top of `run_noopt.py`. The script now builds paths as `join(ROOT, RELATIVE_PATH)`, then run:

```powershell
python run_noopt.py
```

Set `SCENARIO = "a"` for independent household simulations. Set
`SCENARIO = "b"` to enable community surplus sharing and virtual seller-buyer
energy flows; `SHARING_MODE` controls proportional or equal allocation.

## 3d-K and 4d-I settlement modes

- `from OptimizedScenarios import run_3d_k` runs household-level optimization,
  then applies virtual community settlement to the unchanged dispatch.
- `from OptimizedCommunityScenarios import run_4d_i` runs community optimization,
  then reclassifies internal exchanges as individual grid import/export for billing.

The configured default output folders are `results/3d-K` and `results/4d-I`.

## Unified scenario runner

All official scenario codes are registered in `Scenarios.scenario_manager`.

```powershell
python run_scenario.py --list
python run_scenario.py --describe 3d-K
python run_scenario.py 0-I
python run_scenario.py 3d-K --allow-partial
```

Partial scenarios require explicit acknowledgement because their current device
model does not yet cover every component prescribed by the scenario table.

## Composed individual optimization

The individual scenarios use one pipeline:
`run_scenario.py` → `Scenarios/scenario_manager.py` →
`OptimizedScenarios/run_individual_optimization.py` →
`OptimizedIndividualScenarios/individual_optimizer.py`.

The model combines optional boiler and BESS components from
`OptimizedIndividualScenarios/boiler_constraints.py` and
`Optimization/bess_constraints.py`. The community optimizer uses the same
battery component. Inputs remain in
`InputReading`, tariff constraints in `optimization_constraints.py`, and
solver selection/result extraction in `Optimization`.

```python
from OptimizedIndividualScenarios.individual_optimizer import optimize_household

result = optimize_household(
    p_pv=[3, 0], p_ue=[1, 1], p_dhw=[0, 0], dt=0.25,
    include_boiler=False, include_bess=True, size_bess=2,
    boiler_tariff="A", objective="bill", msg=False,
)
```

All power profiles are kW; reported energies are kWh. `bill` minimizes the
household bill; `grid` minimizes imported plus exported energy. The result
reports `objective_value` and `objective_unit`; `objective_Ft` is null for
the energy objective. Physical component flows (`p_grid_to_base`,
`p_grid_to_boiler`) are distinct from A/B tariff totals.

BESS SOC is always cyclic: the final SOC equals the initial SOC. The legacy
`bess_terminal` parameter accepts only `"cyclic"`.
`e_bess_boundary` contains T+1 states, so the last timestep also obeys energy
conservation. `e_bess` retains the T start-of-step states for compatibility.
`allow_grid_charge_for_min_soc=False` disables all grid charging by default.
Setting it to true permits only the energy needed to restore minimum SOC,
not price arbitrage. The guard remains exact even with `run_lp=True`, so
enabling it can introduce binary variables into an otherwise relaxed model.
The charge/discharge mode gates are relaxed when `run_lp=True`.

The legacy `individual_opt_boiler`, `individual_opt_bess`, and
`individual_opt_bess_boiler` functions retain their call signatures and
delegate to the shared model. The BESS-only adapter retains local PV priority.
All adapters now use cyclic SOC and default to disabled grid charging.
B-tariff boiler loads cannot receive PV or BESS
energy in the common model. This corrects the old combined model's
inconsistent B-tariff PV routing. Cyclic SOC changes old results that consumed
initial stored energy without restoring it.

`3d-I` and `3d-K` enable the configured BESS share alongside the boiler;
community settlement is applied after solving the individual schedules.
Flexible HP and DSO objectives remain unsupported and fail explicitly.
The existing fixed-profile HP optimizer remains separately available through
`opt_hp.py`. Disabling boiler optimization leaves an explicitly supplied
`p_el_heater_fixed` profile as a fixed load; omit it to model no boiler load.

The old individual runner paths are compatibility adapters. For direct
multi-household execution use:

```powershell
python -m OptimizedScenarios.run_individual_optimization --sim sim.yaml --profiles profiles.csv --dhw_profiles dhw.csv --bess --mip --boiler-tariff A --objective bill
```

The unified scenario runner reads `[bess] allow_grid_charge_for_min_soc = False`
from config.ini. It also accepts `--allow-grid-charge-for-min-soc` and
`--no-allow-grid-charge-for-min-soc`. Python runners expose the same boolean.
With self-discharge and insufficient PV, cyclic SOC may be infeasible with
the guard disabled; no implicit grid top-up is inserted.

Greedy simulations use periodic initialization in `Utility/bess_periodic.py`:
the causal controller is repeated to find a fixed point for the initial SOC.
Failure to converge or violation of SOC bounds is reported explicitly.
This can require multiple passes over the input horizon. Their existing
`e_bess` series now consistently contains start-of-step states in all public
simulation and optimization results. `e_bess_start` and `e_bess_end` contain T
states; `e_bess_boundary` contains T+1 states including both horizon boundaries.
Consumers that previously used greedy `e_bess` as an end state must switch to
`e_bess_end`. Internal single-pass greedy kernels still return end states to the
periodic initializer.

Every runner also writes the same `bess/` result package through
`Utility/result_schema.py`, with `schema.json` recording version, timestep,
household order, units and indexing. BESS flows are provided in both AC-side
kWh/step (`e_*`) and average kW (`p_*`). Existing scenario-specific financial
and thermal output files are retained; they are not part of this BESS schema.

Greedy charge, discharge and grid top-up limits are shared in
`Utility/bess_dispatch.py`. Both controllers apply self-discharge, use PV for
loads and battery charging, discharge to remaining eligible loads, then apply
optional minimum-SOC grid top-up. PV and grid charging share one step power
limit. The individual controller retains its mode lock; the community greedy
controller has no such lock. Insufficient charging capacity can still make the
minimum-SOC constraint infeasible; no energy is silently created.

Physical and virtual matching share `Utility/energy_allocation.py`. Greedy
community sellers remain proportional to surplus; virtual settlement preserves
its selected seller mode. Tariff eligibility and household component priorities
remain in their respective callers, outside the allocation arithmetic.

Community monetary settlement now uses only
`Economics.calculate_economics.settle_shared_payments`. Compatibility adapters
preserve the old rules: nonoptimized sharing uses `proportional` tariff-cap
allocation, optimized sharing uses `grid_first`, with kW converted to kWh.
Choosing one common tariff-cap policy for all scenarios is a separate modeling
decision; this refactor does not silently change that policy.

