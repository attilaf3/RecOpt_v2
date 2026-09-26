"""0-K közösségi baseline eredmények ábrázolása."""

from nonopt_plot_community import plot_community_case


SHARING_MODE = "proportional"
RESULTS_DIR = f"results_0-K_{SHARING_MODE}"
WINDOW_DAYS = 3
DT = 0.25


if __name__ == "__main__":
    plot_community_case(
        RESULTS_DIR, case_name="0-K", with_bess=False,
        window_days=WINDOW_DAYS, dt=DT,
    )
