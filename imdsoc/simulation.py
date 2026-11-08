"""Discrete-event simulation of site workload (Section VIII, stage 2).

Each dispensing site is a two-stage tandem queue (technician fill, then
pharmacist verification) with FIFO multi-server stations. Shipments arrive
over the business day; any shipment not verified and packed by the carrier
pickup cut-off misses the pickup and gains 24 h. The simulation turns site
workload into delivery delay, which historical replay cannot observe, and
yields per-site backlog for hypothesis H4.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .models import Network
from .plans import FILL_MIN_PER_ORDER, VERIFY_MIN_PER_ORDER, Plan

OPEN_MIN = 10 * 60          # business day length (minutes) before carrier pickup
ARRIVAL_WINDOW_MIN = 6 * 60 # orders released to the floor during the first 6 hours


@dataclass
class SiteDayResult:
    site_id: str
    completion_min: Dict[int, float]     # plan_id -> completion time (min after open)
    missed_pickup: Dict[int, bool]
    backlog: int
    utilization: float


def _fifo_multiserver(arrivals: np.ndarray, service: np.ndarray, servers: int) -> np.ndarray:
    """Event-driven FIFO c-server queue; returns departure times in arrival order."""
    free = [0.0] * max(1, servers)
    heapq.heapify(free)
    out = np.empty(len(arrivals))
    for i in np.argsort(arrivals, kind="stable"):
        t_free = heapq.heappop(free)
        start = max(arrivals[i], t_free)
        out[i] = start + service[i]
        heapq.heappush(free, out[i])
    return out


def simulate_site_day(site_id: str, plans: Sequence[Plan], net: Network, rng: np.random.Generator,
                      demand_scale: float = 1.0) -> SiteDayResult:
    site = net.sites[site_id]
    if not plans:
        return SiteDayResult(site_id, {}, {}, 0, 0.0)
    n = len(plans)
    pharmacists = max(1, math.ceil(site.capacity["pharmacist_verify_min"] / OPEN_MIN))
    techs = max(1, math.ceil(site.capacity["tech_fill_min"] / OPEN_MIN))
    arrivals = np.sort(rng.uniform(0, ARRIVAL_WINDOW_MIN, size=n))
    k = np.array([p.n_orders for p in plans], dtype=float)
    fill = rng.gamma(4.0, FILL_MIN_PER_ORDER / 4.0, size=n) * k * demand_scale
    fill += np.array([6.0 if p.is_cold else 0.0 for p in plans])           # cold pack-out
    verify = rng.gamma(4.0, VERIFY_MIN_PER_ORDER / 4.0, size=n) * k * demand_scale
    # Staff are shared with walk-in work: planned capacity (spare minutes) is
    # spread over the day, so fulfillment work runs slower than wall clock.
    fill *= techs * OPEN_MIN / site.capacity["tech_fill_min"]
    verify_wall = verify * pharmacists * OPEN_MIN / site.capacity["pharmacist_verify_min"]
    t_fill = _fifo_multiserver(arrivals, fill, techs)
    t_done = _fifo_multiserver(t_fill, verify_wall, pharmacists)
    missed = t_done > OPEN_MIN
    util = float(verify.sum() / site.capacity["pharmacist_verify_min"])
    return SiteDayResult(site_id, {p.plan_id: float(t) for p, t in zip(plans, t_done)},
                         {p.plan_id: bool(m) for p, m in zip(plans, missed)}, int(missed.sum()), util)


def simulate_day(plans: Sequence[Plan], net: Network, rng: np.random.Generator,
                 demand_scale: float = 1.0) -> Dict[str, SiteDayResult]:
    by_site: Dict[str, List[Plan]] = {s: [] for s in net.sites}
    for p in plans:
        by_site[p.site_id].append(p)
    return {s: simulate_site_day(s, ps, net, rng, demand_scale) for s, ps in by_site.items()}
