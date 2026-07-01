from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


__all__ = [
    "Tariffs",
    "DEFAULT_TARIFFS",
    "two_tier_cost_steps",
    "calculate_economics",
    "calculate_grid_bill",
    "settle_shared_payments",
]


EPS = 1e-12
PairingMode = Literal["proportional", "equal"]
SharedLowCapMode = Literal["proportional", "grid_first"]


@dataclass(frozen=True)
class Tariffs:
    """
    Központi tarifa-adatok.

    Egységek:
        - energia: kWh
        - árak: Ft/kWh
        - költségek/bevételek: Ft
    """

    grid_a_low_limit_kwh: float = 2523.0
    grid_b_low_limit_kwh: float = 2523.0
    grid_hp_low_limit_kwh: float = 2523.0

    grid_a_low_ft_per_kwh: float = 36.0
    grid_a_high_ft_per_kwh: float = 71.0

    grid_b_low_ft_per_kwh: float = 23.0
    grid_b_high_ft_per_kwh: float = 61.0

    grid_hp_low_ft_per_kwh: float = 29.34
    grid_hp_high_ft_per_kwh: float = 60.1

    pv_export_ft_per_kwh: float = 5.0

    shared_buyer_low_limit_kwh: float = 2523.0
    shared_buyer_low_ft_per_kwh: float = 5.0
    shared_buyer_high_ft_per_kwh: float = 21.0
    shared_rhd_ft_per_kwh: float = 31.0


DEFAULT_TARIFFS = Tariffs()


def _as_nonnegative_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name}: üres tömböt kaptam.")
    return np.maximum(arr, 0.0)


def _as_nonnegative_2d(values, name: str, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = _as_nonnegative_array(values, name=name)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"{name}: csak 1D vagy 2D tömb lehet, kapott dimenzió: {arr.ndim}")

    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"{name}: alakja {arr.shape}, de {expected_shape} kellene.")

    return arr


def _zeros_like_shape(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=float)


def _names_or_default(user_names: list[str] | None, n_users: int) -> list[str]:
    if user_names is None or len(user_names) != n_users:
        return [f"user_{u + 1:03d}" for u in range(n_users)]
    return [str(x) for x in user_names]


def two_tier_cost_steps(
    e_steps_kwh: np.ndarray,
    low_limit_kwh: float,
    low_rate_ft_per_kwh: float,
    high_rate_ft_per_kwh: float,
) -> dict:
    """
    Éves, időrendben haladó kétlépcsős díjszámítás.

    Bemenet:
        e_steps_kwh: kWh / időlépés

    Visszatérés:
        low_kwh, high_kwh, cost_ft
    """
    remaining_low = float(low_limit_kwh)
    low_kwh = 0.0
    high_kwh = 0.0
    cost_ft = 0.0

    for value in np.maximum(np.asarray(e_steps_kwh, dtype=float).ravel(), 0.0):
        e = float(value)
        low_part = min(e, max(remaining_low, 0.0))
        high_part = max(e - low_part, 0.0)

        low_kwh += low_part
        high_kwh += high_part
        cost_ft += low_part * float(low_rate_ft_per_kwh)
        cost_ft += high_part * float(high_rate_ft_per_kwh)
        remaining_low -= low_part

    return {
        "low_kwh": float(low_kwh),
        "high_kwh": float(high_kwh),
        "cost_ft": float(cost_ft),
    }


def _split_low_high_step(e_kwh: float, remaining_low_kwh: float) -> tuple[float, float, float]:
    e = max(float(e_kwh), 0.0)
    remaining = max(float(remaining_low_kwh), 0.0)
    low = min(e, remaining)
    high = max(e - low, 0.0)
    return low, high, remaining - low


def _split_grid_and_shared_proportional(
    e_grid_a_step: float,
    e_shared_in_step: float,
    remaining_low_kwh: float,
) -> tuple[float, float, float, float, float]:
    """
    Ugyanabban az időlépésben az A-import és a shared vásárlás arányosan
    kap a még megmaradó kedvezményes sávból.
    """
    e_grid = max(float(e_grid_a_step), 0.0)
    e_shared = max(float(e_shared_in_step), 0.0)
    total = e_grid + e_shared
    remaining = max(float(remaining_low_kwh), 0.0)

    if total <= EPS:
        return 0.0, 0.0, 0.0, 0.0, remaining

    low_total = min(total, remaining)
    high_total = total - low_total

    grid_weight = e_grid / total
    shared_weight = e_shared / total

    grid_low = low_total * grid_weight
    grid_high = high_total * grid_weight
    shared_low = low_total * shared_weight
    shared_high = high_total * shared_weight

    return grid_low, grid_high, shared_low, shared_high, remaining - low_total


def _split_grid_and_shared_grid_first(
    e_grid_a_step: float,
    e_shared_in_step: float,
    remaining_low_kwh: float,
) -> tuple[float, float, float, float, float]:
    """
    Először az A-tarifás hálózati import fogyasztja a kedvezményes sávot,
    utána a közösségből vett energia.
    """
    grid_low, grid_high, remaining = _split_low_high_step(e_grid_a_step, remaining_low_kwh)
    shared_low, shared_high, remaining = _split_low_high_step(e_shared_in_step, remaining)
    return grid_low, grid_high, shared_low, shared_high, remaining


def _allocate_seller_to_buyers(
    seller_energy_kwh: float,
    remaining_buyer_need_kwh: np.ndarray,
    pairing_mode: PairingMode,
) -> np.ndarray:
    need = np.maximum(np.asarray(remaining_buyer_need_kwh, dtype=float), 0.0)
    energy = min(max(float(seller_energy_kwh), 0.0), float(need.sum()))
    out = np.zeros_like(need)

    if energy <= EPS or need.sum() <= EPS:
        return out

    if pairing_mode == "proportional":
        return need * (energy / need.sum())

    if pairing_mode == "equal":
        remaining = energy
        while remaining > EPS:
            room = need - out
            active = room > EPS
            n_active = int(active.sum())
            if n_active == 0:
                break

            quota = remaining / n_active
            take = np.minimum(room[active], quota)
            out[active] += take
            remaining -= float(take.sum())
        return out

    raise ValueError("pairing_mode csak 'proportional' vagy 'equal' lehet.")


def settle_shared_payments(
    e_shared_in: np.ndarray,
    e_shared_out: np.ndarray,
    e_grid_import_a: np.ndarray,
    *,
    tariffs: Tariffs = DEFAULT_TARIFFS,
    pairing_mode: PairingMode = "proportional",
    shared_low_cap_mode: SharedLowCapMode = "proportional",
) -> dict:
    """
    Közösségi megosztás pénzügyi elszámolása.

    Bemenetek egysége:
        kWh / időlépés

    Fontos logika:
        - a 2523 kWh/év kedvezményes sáv a vevő éves kerete;
        - ugyanazt a keretet fogyasztja az A-tarifás hálózati import és
          a közösségből vett energia;
        - a shared energiadíj low sávban 5 Ft/kWh, high sávban 21 Ft/kWh;
        - az RHD mindig a teljes shared_in után fizetendő;
        - az eladó a vevői sáv szerinti energiadíjat kapja meg.
    """
    e_shared_in = _as_nonnegative_2d(e_shared_in, "e_shared_in")
    e_shared_out = _as_nonnegative_2d(e_shared_out, "e_shared_out", expected_shape=e_shared_in.shape)
    e_grid_import_a = _as_nonnegative_2d(e_grid_import_a, "e_grid_import_a", expected_shape=e_shared_in.shape)

    if shared_low_cap_mode == "proportional":
        split_func = _split_grid_and_shared_proportional
    elif shared_low_cap_mode == "grid_first":
        split_func = _split_grid_and_shared_grid_first
    else:
        raise ValueError("shared_low_cap_mode csak 'proportional' vagy 'grid_first' lehet.")

    T, U = e_shared_in.shape

    pair_kwh = np.zeros((U, U), dtype=float)
    pair_energy_payment_ft = np.zeros((U, U), dtype=float)

    buyer_low_remaining = np.full(U, float(tariffs.shared_buyer_low_limit_kwh), dtype=float)
    buyer_grid_a_low_kwh = np.zeros(U, dtype=float)
    buyer_grid_a_high_kwh = np.zeros(U, dtype=float)
    buyer_shared_low_kwh = np.zeros(U, dtype=float)
    buyer_shared_high_kwh = np.zeros(U, dtype=float)

    buyer_energy_cost_ft = np.zeros(U, dtype=float)
    buyer_rhd_ft = e_shared_in.sum(axis=0) * float(tariffs.shared_rhd_ft_per_kwh)
    seller_revenue_ft = np.zeros(U, dtype=float)

    for t in range(T):
        shared_low_t = np.zeros(U, dtype=float)
        shared_high_t = np.zeros(U, dtype=float)

        for buyer in range(U):
            grid_low, grid_high, shared_low, shared_high, remaining = split_func(
                e_grid_a_step=float(e_grid_import_a[t, buyer]),
                e_shared_in_step=float(e_shared_in[t, buyer]),
                remaining_low_kwh=float(buyer_low_remaining[buyer]),
            )
            buyer_low_remaining[buyer] = remaining

            buyer_grid_a_low_kwh[buyer] += grid_low
            buyer_grid_a_high_kwh[buyer] += grid_high
            buyer_shared_low_kwh[buyer] += shared_low
            buyer_shared_high_kwh[buyer] += shared_high
            shared_low_t[buyer] = shared_low
            shared_high_t[buyer] = shared_high

        shared_energy_cost_t = (
            shared_low_t * float(tariffs.shared_buyer_low_ft_per_kwh)
            + shared_high_t * float(tariffs.shared_buyer_high_ft_per_kwh)
        )
        buyer_energy_cost_ft += shared_energy_cost_t

        sin = e_shared_in[t, :]
        sout = e_shared_out[t, :]
        total_share = min(float(sin.sum()), float(sout.sum()))

        if total_share <= EPS:
            continue

        sellers = np.flatnonzero(sout > EPS)
        buyers = np.flatnonzero(sin > EPS)

        if len(sellers) == 0 or len(buyers) == 0:
            continue

        buyer_avg_energy_rate = np.divide(
            shared_energy_cost_t,
            sin,
            out=np.zeros(U, dtype=float),
            where=sin > EPS,
        )

        remaining_buyers = sin[buyers].copy()
        remaining_buyer_rates = buyer_avg_energy_rate[buyers].copy()

        for seller in sellers:
            alloc_to_buyers = _allocate_seller_to_buyers(
                seller_energy_kwh=float(sout[seller]),
                remaining_buyer_need_kwh=remaining_buyers,
                pairing_mode=pairing_mode,
            )
            if alloc_to_buyers.sum() <= EPS:
                continue

            pair_kwh[seller, buyers] += alloc_to_buyers
            pair_payment = alloc_to_buyers * remaining_buyer_rates
            pair_energy_payment_ft[seller, buyers] += pair_payment
            seller_revenue_ft[seller] += float(pair_payment.sum())
            remaining_buyers = np.maximum(remaining_buyers - alloc_to_buyers, 0.0)

    return {
        "pair_kwh": pair_kwh,
        "pair_energy_payment_ft": pair_energy_payment_ft,
        "buyer_grid_a_low_kwh": buyer_grid_a_low_kwh,
        "buyer_grid_a_high_kwh": buyer_grid_a_high_kwh,
        "buyer_shared_low_kwh": buyer_shared_low_kwh,
        "buyer_shared_high_kwh": buyer_shared_high_kwh,
        "buyer_energy_cost_ft": buyer_energy_cost_ft,
        "buyer_rhd_ft": buyer_rhd_ft,
        "buyer_total_shared_cost_ft": buyer_energy_cost_ft + buyer_rhd_ft,
        "seller_revenue_ft": seller_revenue_ft,
    }


def calculate_grid_bill(
    e_grid_import_a: np.ndarray,
    e_grid_import_b: np.ndarray | None = None,
    e_grid_export: np.ndarray | None = None,
    *,
    tariffs: Tariffs = DEFAULT_TARIFFS,
) -> dict:
    """
    Egy háztartás vagy aggregált idősor hálózati bruttó villanyszámlája.

    Bemenetek egysége:
        kWh / időlépés

    Ez nem számol közösségi megosztással, csak:
        A-import + B-import - PV exportbevétel.
    """
    e_a = _as_nonnegative_array(e_grid_import_a, "e_grid_import_a")

    if e_grid_import_b is None:
        e_b = np.zeros_like(e_a)
    else:
        e_b = _as_nonnegative_array(e_grid_import_b, "e_grid_import_b")
        if e_b.shape != e_a.shape:
            raise ValueError(f"e_grid_import_b alakja {e_b.shape}, de {e_a.shape} kellene.")

    if e_grid_export is None:
        e_export = np.zeros_like(e_a)
    else:
        e_export = _as_nonnegative_array(e_grid_export, "e_grid_export")
        if e_export.shape != e_a.shape:
            raise ValueError(f"e_grid_export alakja {e_export.shape}, de {e_a.shape} kellene.")

    a = two_tier_cost_steps(
        e_steps_kwh=e_a,
        low_limit_kwh=tariffs.grid_a_low_limit_kwh,
        low_rate_ft_per_kwh=tariffs.grid_a_low_ft_per_kwh,
        high_rate_ft_per_kwh=tariffs.grid_a_high_ft_per_kwh,
    )
    b = two_tier_cost_steps(
        e_steps_kwh=e_b,
        low_limit_kwh=tariffs.grid_b_low_limit_kwh,
        low_rate_ft_per_kwh=tariffs.grid_b_low_ft_per_kwh,
        high_rate_ft_per_kwh=tariffs.grid_b_high_ft_per_kwh,
    )

    export_revenue_ft = float(e_export.sum() * tariffs.pv_export_ft_per_kwh)
    import_cost_ft = float(a["cost_ft"] + b["cost_ft"])

    return {
        "grid_import_a_kwh": float(e_a.sum()),
        "grid_import_b_kwh": float(e_b.sum()),
        "grid_import_kwh": float(e_a.sum() + e_b.sum()),
        "grid_export_kwh": float(e_export.sum()),
        "grid_import_a_low_kwh": a["low_kwh"],
        "grid_import_a_high_kwh": a["high_kwh"],
        "grid_import_b_low_kwh": b["low_kwh"],
        "grid_import_b_high_kwh": b["high_kwh"],
        "grid_import_a_cost_ft": a["cost_ft"],
        "grid_import_b_cost_ft": b["cost_ft"],
        "grid_import_cost_ft": import_cost_ft,
        "grid_export_revenue_ft": export_revenue_ft,
        "brt_bill_ft": import_cost_ft - export_revenue_ft,
    }


def calculate_economics(
    e_grid_import_a: np.ndarray,
    e_grid_import_b: np.ndarray | None = None,
    e_grid_export: np.ndarray | None = None,
    e_shared_in: np.ndarray | None = None,
    e_shared_out: np.ndarray | None = None,
    *,
    user_names: list[str] | None = None,
    tariffs: Tariffs = DEFAULT_TARIFFS,
    pairing_mode: PairingMode = "proportional",
    shared_low_cap_mode: SharedLowCapMode = "proportional",
) -> dict:
    """
    Központi gazdasági számítás egyéni és közösségi esethez is.

    Bemenetek egysége:
        minden idősor kWh / időlépés.

    Egyéni eset:
        csak e_grid_import_a / e_grid_import_b / e_grid_export kell,
        a shared mezők maradhatnak None értéken.

    Közösségi eset:
        add meg az e_shared_in és e_shared_out idősorokat is.
    """
    e_grid_import_a = _as_nonnegative_2d(e_grid_import_a, "e_grid_import_a")
    T, U = e_grid_import_a.shape
    shape = (T, U)

    if e_grid_import_b is None:
        e_grid_import_b = _zeros_like_shape(shape)
    else:
        e_grid_import_b = _as_nonnegative_2d(e_grid_import_b, "e_grid_import_b", expected_shape=shape)

    if e_grid_export is None:
        e_grid_export = _zeros_like_shape(shape)
    else:
        e_grid_export = _as_nonnegative_2d(e_grid_export, "e_grid_export", expected_shape=shape)

    if e_shared_in is None:
        e_shared_in = _zeros_like_shape(shape)
    else:
        e_shared_in = _as_nonnegative_2d(e_shared_in, "e_shared_in", expected_shape=shape)

    if e_shared_out is None:
        e_shared_out = _zeros_like_shape(shape)
    else:
        e_shared_out = _as_nonnegative_2d(e_shared_out, "e_shared_out", expected_shape=shape)

    user_names = _names_or_default(user_names, U)

    has_shared = bool(e_shared_in.sum() > EPS or e_shared_out.sum() > EPS)

    if has_shared:
        shared_settlement = settle_shared_payments(
            e_shared_in=e_shared_in,
            e_shared_out=e_shared_out,
            e_grid_import_a=e_grid_import_a,
            tariffs=tariffs,
            pairing_mode=pairing_mode,
            shared_low_cap_mode=shared_low_cap_mode,
        )
    else:
        shared_settlement = {
            "pair_kwh": np.zeros((U, U), dtype=float),
            "pair_energy_payment_ft": np.zeros((U, U), dtype=float),
            "buyer_grid_a_low_kwh": np.zeros(U, dtype=float),
            "buyer_grid_a_high_kwh": np.zeros(U, dtype=float),
            "buyer_shared_low_kwh": np.zeros(U, dtype=float),
            "buyer_shared_high_kwh": np.zeros(U, dtype=float),
            "buyer_energy_cost_ft": np.zeros(U, dtype=float),
            "buyer_rhd_ft": np.zeros(U, dtype=float),
            "buyer_total_shared_cost_ft": np.zeros(U, dtype=float),
            "seller_revenue_ft": np.zeros(U, dtype=float),
        }

    rows: list[dict] = []

    for u in range(U):
        if has_shared:
            # Shared esetben az A-sáv low/high bontása a shared elszámolásból jön,
            # mert az A-hálózati import és a shared ugyanazt a vevői keretet fogyasztja.
            a_low = float(shared_settlement["buyer_grid_a_low_kwh"][u])
            a_high = float(shared_settlement["buyer_grid_a_high_kwh"][u])
            import_cost_a = (
                a_low * float(tariffs.grid_a_low_ft_per_kwh)
                + a_high * float(tariffs.grid_a_high_ft_per_kwh)
            )
        else:
            a_bill = two_tier_cost_steps(
                e_steps_kwh=e_grid_import_a[:, u],
                low_limit_kwh=tariffs.grid_a_low_limit_kwh,
                low_rate_ft_per_kwh=tariffs.grid_a_low_ft_per_kwh,
                high_rate_ft_per_kwh=tariffs.grid_a_high_ft_per_kwh,
            )
            a_low = float(a_bill["low_kwh"])
            a_high = float(a_bill["high_kwh"])
            import_cost_a = float(a_bill["cost_ft"])

        b_bill = two_tier_cost_steps(
            e_steps_kwh=e_grid_import_b[:, u],
            low_limit_kwh=tariffs.grid_b_low_limit_kwh,
            low_rate_ft_per_kwh=tariffs.grid_b_low_ft_per_kwh,
            high_rate_ft_per_kwh=tariffs.grid_b_high_ft_per_kwh,
        )

        b_low = float(b_bill["low_kwh"])
        b_high = float(b_bill["high_kwh"])
        import_cost_b = float(b_bill["cost_ft"])
        import_cost = import_cost_a + import_cost_b

        grid_export_revenue = float(e_grid_export[:, u].sum() * tariffs.pv_export_ft_per_kwh)
        shared_purchase_energy_cost = float(shared_settlement["buyer_energy_cost_ft"][u])
        shared_purchase_rhd = float(shared_settlement["buyer_rhd_ft"][u])
        shared_purchase_cost = float(shared_settlement["buyer_total_shared_cost_ft"][u])
        shared_revenue = float(shared_settlement["seller_revenue_ft"][u])

        cash_out = import_cost + shared_purchase_cost
        cash_in = grid_export_revenue + shared_revenue
        brt_bill = cash_out - cash_in

        rows.append(
            {
                "household": user_names[u],
                "grid_import_a_kwh": float(e_grid_import_a[:, u].sum()),
                "grid_import_b_kwh": float(e_grid_import_b[:, u].sum()),
                "grid_import_kwh": float(e_grid_import_a[:, u].sum() + e_grid_import_b[:, u].sum()),
                "grid_export_kwh": float(e_grid_export[:, u].sum()),
                "grid_import_a_low_kwh": a_low,
                "grid_import_a_high_kwh": a_high,
                "grid_import_b_low_kwh": b_low,
                "grid_import_b_high_kwh": b_high,
                "grid_import_a_cost_ft": import_cost_a,
                "grid_import_b_cost_ft": import_cost_b,
                "grid_import_cost_ft": import_cost,
                "shared_in_kwh": float(e_shared_in[:, u].sum()),
                "shared_out_kwh": float(e_shared_out[:, u].sum()),
                "shared_purchase_low_kwh": float(shared_settlement["buyer_shared_low_kwh"][u]),
                "shared_purchase_high_kwh": float(shared_settlement["buyer_shared_high_kwh"][u]),
                "shared_purchase_energy_cost_ft": shared_purchase_energy_cost,
                "shared_purchase_rhd_ft": shared_purchase_rhd,
                "shared_purchase_cost_ft": shared_purchase_cost,
                "shared_revenue_ft": shared_revenue,
                "grid_export_revenue_ft": grid_export_revenue,
                "export_revenue_ft": grid_export_revenue,
                "cash_out_ft": cash_out,
                "cash_in_ft": cash_in,
                "brt_bill_ft": brt_bill,
            }
        )

    per_user_df = pd.DataFrame(rows)

    sum_cols = [
        "grid_import_a_kwh",
        "grid_import_b_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "grid_import_a_low_kwh",
        "grid_import_a_high_kwh",
        "grid_import_b_low_kwh",
        "grid_import_b_high_kwh",
        "grid_import_a_cost_ft",
        "grid_import_b_cost_ft",
        "grid_import_cost_ft",
        "shared_in_kwh",
        "shared_out_kwh",
        "shared_purchase_low_kwh",
        "shared_purchase_high_kwh",
        "shared_purchase_energy_cost_ft",
        "shared_purchase_rhd_ft",
        "shared_purchase_cost_ft",
        "shared_revenue_ft",
        "grid_export_revenue_ft",
        "export_revenue_ft",
        "cash_out_ft",
        "cash_in_ft",
        "brt_bill_ft",
    ]

    summary = {col: float(per_user_df[col].sum()) for col in sum_cols}
    summary.update(
        {
            "grid_a_low_limit_kwh": float(tariffs.grid_a_low_limit_kwh),
            "grid_a_low_ft_per_kwh": float(tariffs.grid_a_low_ft_per_kwh),
            "grid_a_high_ft_per_kwh": float(tariffs.grid_a_high_ft_per_kwh),
            "grid_b_low_limit_kwh": float(tariffs.grid_b_low_limit_kwh),
            "grid_b_low_ft_per_kwh": float(tariffs.grid_b_low_ft_per_kwh),
            "grid_b_high_ft_per_kwh": float(tariffs.grid_b_high_ft_per_kwh),
            "pv_export_ft_per_kwh": float(tariffs.pv_export_ft_per_kwh),
            "shared_buyer_low_limit_kwh": float(tariffs.shared_buyer_low_limit_kwh),
            "shared_buyer_low_ft_per_kwh": float(tariffs.shared_buyer_low_ft_per_kwh),
            "shared_buyer_high_ft_per_kwh": float(tariffs.shared_buyer_high_ft_per_kwh),
            "shared_rhd_ft_per_kwh": float(tariffs.shared_rhd_ft_per_kwh),
        }
    )

    return {
        "per_user_df": per_user_df,
        "summary": summary,
        "shared_settlement": shared_settlement,
    }