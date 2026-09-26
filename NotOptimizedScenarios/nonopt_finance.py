"""A-, B- és közösségi tarifák éves pénzügyi elszámolása."""

from __future__ import annotations

import numpy as np


A_LOW_LIMIT = 2523.0
A_LOW_PRICE = 36.386
A_HIGH_PRICE = 70.104
B_LOW_LIMIT = 2523.0
B_LOW_PRICE = 22.962
B_HIGH_PRICE = 60.935
GRID_EXPORT_PRICE = 5.25
SHARED_LOW_PRICE = 5.25
SHARED_HIGH_PRICE = 22.0
SHARED_RHD = A_LOW_PRICE - SHARED_LOW_PRICE
EPS = 1e-12


def two_tier(energy: np.ndarray, limit: float, low_price: float, high_price: float) -> tuple[float, float, float]:
    total = float(np.maximum(np.asarray(energy, dtype=float), 0.0).sum())
    low = min(total, limit)
    high = max(total - low, 0.0)
    return low, high, low * low_price + high * high_price


def settle(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    grid_export: np.ndarray,
    shared_in: np.ndarray,
    shared_out: np.ndarray,
) -> dict[str, np.ndarray]:
    """Felhasználói elszámolás; a shared és grid A ugyanazt a kedvezményes sávot használja."""
    n, users = grid_a.shape
    remaining = np.full(users, A_LOW_LIMIT)
    arrays = {name: np.zeros(users) for name in (
        "grid_a_low", "grid_a_high", "grid_b_low", "grid_b_high",
        "grid_a_cost", "grid_b_cost", "shared_low", "shared_high",
        "shared_energy_cost", "shared_rhd_cost", "shared_revenue",
        "grid_export_revenue", "bill"
    )}
    pair_kwh = np.zeros((users, users))
    pair_payment = np.zeros((users, users))

    for t in range(n):
        shared_cost_t = np.zeros(users)
        for u in range(users):
            s = max(float(shared_in[t, u]), 0.0)
            g = max(float(grid_a[t, u]), 0.0)
            s_low = min(s, remaining[u])
            s_high = s - s_low
            remaining[u] -= s_low
            g_low = min(g, remaining[u])
            g_high = g - g_low
            remaining[u] -= g_low
            arrays["shared_low"][u] += s_low
            arrays["shared_high"][u] += s_high
            arrays["grid_a_low"][u] += g_low
            arrays["grid_a_high"][u] += g_high
            shared_cost_t[u] = s_low * SHARED_LOW_PRICE + s_high * SHARED_HIGH_PRICE
            arrays["shared_energy_cost"][u] += shared_cost_t[u]

        sent = shared_out[t]
        received = shared_in[t]
        total_shared = float(received.sum())
        if total_shared > EPS:
            sellers = np.flatnonzero(sent > EPS)
            buyers = np.flatnonzero(received > EPS)
            buyer_weights = received[buyers] / total_shared
            pair_t = np.outer(sent[sellers], buyer_weights)
            buyer_rates = shared_cost_t[buyers] / received[buyers]
            payment_t = pair_t * buyer_rates[None, :]
            index = np.ix_(sellers, buyers)
            pair_kwh[index] += pair_t
            pair_payment[index] += payment_t
            arrays["shared_revenue"][sellers] += payment_t.sum(axis=1)

    for u in range(users):
        b_low, b_high, b_cost = two_tier(grid_b[:, u], B_LOW_LIMIT, B_LOW_PRICE, B_HIGH_PRICE)
        arrays["grid_b_low"][u] = b_low
        arrays["grid_b_high"][u] = b_high
        arrays["grid_b_cost"][u] = b_cost
    arrays["grid_a_cost"] = arrays["grid_a_low"] * A_LOW_PRICE + arrays["grid_a_high"] * A_HIGH_PRICE
    arrays["shared_rhd_cost"] = shared_in.sum(axis=0) * SHARED_RHD
    arrays["grid_export_revenue"] = grid_export.sum(axis=0) * GRID_EXPORT_PRICE
    arrays["bill"] = (
        arrays["grid_a_cost"] + arrays["grid_b_cost"]
        + arrays["shared_energy_cost"] + arrays["shared_rhd_cost"]
        - arrays["shared_revenue"] - arrays["grid_export_revenue"]
    )
    arrays["pair_kwh"] = pair_kwh
    arrays["pair_payment"] = pair_payment
    return arrays