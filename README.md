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

