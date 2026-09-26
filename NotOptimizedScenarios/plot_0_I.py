"""0-I egyéni baseline eredmények ábrázolása."""

from nonopt_plot_individual import plot_individual_case


RESULTS_DIR = "../results_for_comparison/results_0-I"
HOUSEHOLD = "0420144888235070"
WINDOW_DAYS = 3
DT = 0.25


if __name__ == "__main__":
    plot_individual_case(
        RESULTS_DIR, HOUSEHOLD, case_name="0-I", boiler_tariff="B",
        with_bess=False, with_boiler_model=False,
        window_days=WINDOW_DAYS, dt=DT,
    )
