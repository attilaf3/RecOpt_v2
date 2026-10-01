# RecOpt_v2 – Modelling and Optimization of Renewable Energy Communities

This project analyses individual households and renewable energy communities at a 15-minute resolution. The models combine photovoltaic generation (PV), battery energy storage systems (BESS), electric water heaters and hot water storage systems (HSS) with household electricity demand and energy settlement.

The aim is to assess how energy sharing and flexible device control affect grid imports, grid exports, local energy utilisation and annual household electricity bills.

Repository: [attilaf3/RecOpt_v2](https://github.com/attilaf3/RecOpt_v2)

## Scenarios

| Scenario | Settlement | Device operation | Energy sharing |
| --- | --- | --- | --- |
| `0-I` | Individual | Baseline demand and PV profiles, without BESS | No |
| `0-K` | Community | Baseline demand and PV profiles, without BESS | Yes |
| `1d-I` | Individual | Rule-based device control | No |
| `1d-K` | Community | Rule-based device control | Yes |
| `3d-I` | Individual | Optimised BESS and electric water heater operation | No |
| `4d-K-C` | Community | Community-level cost minimisation | Yes |

In individual scenarios, each household manages its own generation, storage and grid connection. In community scenarios, available surplus generation can also supply eligible demand from other members.

Use the same household list, input period, PV sizing and BESS allocation when comparing scenarios.

## Modelled Components

- **Base load:** non-controllable electricity demand represented by an input time series.
- **PV:** an input generation profile that can be scaled using a factor defined in the runner.
- **BESS:** battery storage with charging and discharging power limits, efficiencies and state-of-charge constraints.
- **Electric water heater / HSS:** a thermal model describing the tank energy balance, heat losses, hot water withdrawals and electric heating.
- **Grid:** imports supplying the remaining demand and exports of unused surplus generation.

BESS can be assigned to selected households with PV. The physical and operational constraints of water heaters are defined in the optimiser or rule-based controller for each scenario.

## Environment and Installation

Running the project requires Python, data-processing packages and an available solver for optimisation scenarios. The documented runners use type annotations compatible with Python 3.10 or later.

```bash
git clone https://github.com/attilaf3/RecOpt_v2.git
cd RecOpt_v2
python -m venv .venv
```

Activate the virtual environment on Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the core packages:

```bash
python -m pip install numpy pandas PyYAML pulp matplotlib
```

If the repository includes a dependency file for the selected version, use it:

```bash
python -m pip install -r requirements.txt
```

The optimisers can use Gurobi or CBC, depending on the implementation. Gurobi requires an appropriate installation and a valid licence. When using `GUROBI_CMD`, the `gurobi_cl` executable must also be available; installing the Python package alone does not replace this requirement.

## Input Data

| Input | Contents |
| --- | --- |
| `simulation_config_disaggregated_with_userlist.yaml` | Simulation configuration and the `users_list` of households to process |
| `measurements_disaggregated_v2.csv` or `measurements_disaggregated_v3.csv` | Household electricity demand, PV generation and fixed water heater profiles |
| `dhw_v2.csv` | Household hot water withdrawal profiles |
| Household YAML files | Profile references and PV, BESS and HSS parameters |

File names and the household YAML directory must match the loading logic of the selected runner. On Linux, for example, `Users_v2` and `users_v2` are different directory names.

The documented annual runners expect **35,040 time steps** with a **0.25-hour** time step, corresponding to a non-leap year. The first CSV column is read as the index; the remaining column names must match the profile references in the YAML files.

Electrical input profiles are expressed in **kWh per time step**. Internal power values are calculated as:

```text
p[t] = e[t] / dt
dt = 0.25 h
```

Hot water profiles contain **litres per time step**. The loader calculates thermal demand from the water volume and the temperature difference between cold and hot water. Thermal demand and the water heater's electricity consumption are separate quantities.

## Running the Simulations

Configure each simulation at the beginning of the corresponding `run_*.py` file. In PyCharm, select the desired runner and Python environment, then start the simulation using **Shift+F10**.

The documented individual runner is `run_individual_opt_bess_boiler.py`. A separate `run_one_individual_opt_bess_boiler.py` file supports simulations for a single household. The community runner name may vary between project versions; use each `run_*.py` file together with its matching optimiser.

### Main Settings

| Parameter | Purpose |
| --- | --- |
| `INPUT_DIR` | Directory containing the input files |
| `SIM_YAML` | Path to the simulation configuration |
| `PROFILES_CSV` | Path to the electrical profiles |
| `DHW_PROFILES_CSV` | Path to the hot water profiles |
| `OUT_DIR` / `out_dir` | Output directory, where supported by the runner |
| `MAX_USERS` | Maximum number of valid households to process |
| `PV_RATIO` | Scaling factor for PV profiles |
| `BESS_SHARE_PCT` | Percentage of PV households assigned BESS |
| `BESS_HOUSEHOLDS` | Explicit selection of households assigned BESS |
| `BESS_DEFAULT_SIZE_KWH` | Fallback capacity when a selected household has no positive BESS size |
| `SOLVER` | Optimisation solver selection |
| `SHARING_MODE` | Community sharing mode: `equal` or `proportional` |
| `OBJECTIVE` | Set to `"bill"` for community cost minimisation |

Not every runner provides all of these parameters. The settings available in the selected file are authoritative.

### BESS Allocation

```python
BESS_HOUSEHOLDS = None  # Allocate using BESS_SHARE_PCT
BESS_SHARE_PCT = 40.0
```

For explicit allocation, provide actual YAML keys or household names accepted by the runner:

```python
BESS_HOUSEHOLDS = [
    "load_0420144653449093",
]
```

An explicit list overrides percentage-based selection. An empty list (`[]`) assigns BESS to no households. Use the same allocation across the scenarios being compared.

## Energy Sharing

The community distributes available surplus from eligible households among members with an import requirement. Shared energy is a settlement quantity; the model does not calculate network power flows or line voltages.

- **`equal`:** equal quotas for eligible buyers with demand in the current time step.
- **`proportional`:** allocation in proportion to eligible buyers' demand in the current time step.

Allocation is limited by each buyer's demand and the available surplus. Remaining surplus becomes grid export, while unmet demand becomes grid import. The redistribution of unused quotas and the calculation of eligible demand depend on the sharing algorithm implemented in the selected version.

## Tariffs and Settlement

Tariffs are model parameters. Check and configure unit prices, annual discounted consumption allowances, export remuneration and community settlement rules in the settlement or optimisation code. Apply a consistent treatment of VAT throughout the comparison.

In the model, an optimised water heater on tariff A can receive energy from its own PV and BESS, as well as eligible shared energy in community scenarios. Fixed water heater consumption assigned to a separate tariff B circuit is settled exclusively through that grid circuit and cannot receive PV, BESS or community energy.

Cost minimisation reduces the difference between the modelled electricity purchasing costs and electricity sales revenue. This is an operating-cost assessment; investment costs and payback periods require a separate economic calculation.

## Outputs

Results are saved to the directory specified by the runner. The documented optimisation runners use outputs including:

- `summary.json`: aggregate results and run information.
- `household_summary.csv`: annual energy quantities and costs for each household.
- `timeseries_*.csv`: household energy flows, BESS state and HSS temperature, when time-series export is enabled.

Community runners may also save aggregate time series. Output names can differ between versions; point the plotting scripts to the actual result directory.

The community cost-minimisation scenario is labelled **`4d-K-C`**. A label including proportional sharing is, for example, **`4d-K-C_proportional`**. The output directory may include an additional prefix.

### Performance Indicators

| Indicator | Meaning |
| --- | --- |
| Grid import | Energy purchased from the external grid |
| Grid export | Energy fed into the external grid |
| Shared energy | Energy settled between community members |
| Annual net electricity bill | Electricity purchasing costs minus electricity sales revenue |
| SCI | Self-consumption index |
| SSI | Self-sufficiency index |

In the project's earlier sharing assessment, SCI is the ratio of shared energy to surplus energy before sharing, while SSI is the ratio of shared energy to import demand before sharing. Indicators describing total local energy utilisation may also include direct self-consumption. Follow the calculation definitions used in the relevant plots when comparing scenarios.

Annual savings relative to the baseline are calculated as:

```text
savings_HUF = annual_bill_0-I - annual_bill_selected_scenario
savings_pct = 100 * savings_HUF / annual_bill_0-I
```
