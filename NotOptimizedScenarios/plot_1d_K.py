"""1d-K közösségi rule-based eredmények ábrázolása."""

from nonopt_plot_community import plot_community_case


BESS_SHARE_PCT = 100.0 # 10 bess van csak
SHARING_MODE = "proportional"
RESULTS_DIR = f"results_1d-K_{SHARING_MODE}_{BESS_SHARE_PCT:g}pct_bess"
WINDOW_DAYS = 3
DT = 0.25


if __name__ == "__main__":
    plot_community_case(
        RESULTS_DIR, case_name="1d-K", with_bess=True,
        window_days=WINDOW_DAYS, dt=DT,
    )
