"""1d-I egyéni rule-based eredmények ábrázolása."""

from nonopt_plot_individual import plot_individual_case


RESULTS_DIR = "../results_for_comparison/results_1d-I"
HOUSEHOLD = "0420144888235070"
WINDOW_DAYS = 3
DT = 0.25


if __name__ == "__main__":
    plot_individual_case(
        RESULTS_DIR, HOUSEHOLD, case_name="1d-I", boiler_tariff="A",
        with_bess=True, with_boiler_model=True,
        window_days=WINDOW_DAYS, dt=DT,
    )
