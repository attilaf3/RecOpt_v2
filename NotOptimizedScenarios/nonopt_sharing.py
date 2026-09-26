"""Közösségi energia nettóigény-alapú kiosztása."""

from __future__ import annotations

from typing import Literal

import numpy as np


SharingMode = Literal["proportional", "equal"]
EPS = 1e-12


def allocate(need: np.ndarray, available: float, mode: SharingMode) -> np.ndarray:
    """Kiosztás a max(fogyasztás - saját PV - BESS-kisütés, 0) igényre."""
    need = np.maximum(np.asarray(need, dtype=float), 0.0)
    offered = min(max(float(available), 0.0), float(need.sum()))
    result = np.zeros_like(need)
    if offered <= EPS:
        return result
    if mode == "proportional":
        return need * offered / need.sum()
    if mode != "equal":
        raise ValueError("A sharing_mode csak 'proportional' vagy 'equal' lehet.")
    active = need > EPS
    quota = offered / int(active.sum())
    result[active] = np.minimum(need[active], quota)
    return result


def share_timestep(
    surplus: np.ndarray,
    need: np.ndarray,
    mode: SharingMode,
) -> tuple[np.ndarray, np.ndarray]:
    surplus = np.maximum(np.asarray(surplus, dtype=float), 0.0)
    received = allocate(need, float(surplus.sum()), mode)
    shared = float(received.sum())
    sent = surplus * shared / surplus.sum() if surplus.sum() > EPS else np.zeros_like(surplus)
    return received, sent
