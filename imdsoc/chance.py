"""Chance-constraint screening, Eq. (6).

A plan enters S only if  P(T_s <= min{D_s, H_ps}) >= 1 - alpha,
implemented as  Q_hat_{1-alpha}(T_s) <= min{D_s, H_ps}
using quantile regression, with a smaller alpha for cold-chain products.
Because the screen runs before optimization, no cost trade-off can admit a
plan that violates it.
"""
from __future__ import annotations

from typing import List, Tuple

from .plans import Plan


def chance_filter(plans: List[Plan]) -> Tuple[List[Plan], List[Plan]]:
    """Split plans into (kept, rejected) by the chance constraint."""
    kept, rejected = [], []
    for p in plans:
        (kept if p.q_1_minus_alpha <= p.binding_limit else rejected).append(p)
    return kept, rejected
