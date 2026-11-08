"""Set-partitioning mixed-integer program, Eqs. (2)-(5) (Section V.C-D).

    min  sum_s c_s z_s                                   (2)
    s.t. sum_{s: i in O_s} z_s = 1            for all i   (3)
         sum_{s: j_s = j} q_sd z_s <= Inv_jd  for all j,d (4)
         sum_{s: j_s = j} l_sr z_s <= Cap_jr  for all j,r (5)
         z_s in {0, 1}

Solved with HiGHS (via ``scipy.optimize.milp``) under a time limit; the
incumbent and optimality gap are recorded. An order with no feasible plan, or
one that cannot be served once inventory and capacity are shared, is picked
up by a heavily penalized "unassigned" variable so the model stays feasible
and the order is escalated with its reason instead of silently dropped.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import coo_matrix

from .models import RESOURCES, Network, Order
from .plans import Plan

UNASSIGNED_PENALTY = 1.0e6


@dataclass
class SolveResult:
    selected: List[Plan]
    unassigned: List[str]
    objective: float
    gap: float
    status: str
    solve_seconds: float
    n_plans: int
    n_orders: int
    binding: Dict[str, float] = field(default_factory=dict)


def site_inventory(net: Network, orders: Sequence[Order]) -> Dict[Tuple[str, str], float]:
    ndcs = {o.ndc for o in orders}
    inv = {}
    for sid, site in net.sites.items():
        for ndc in ndcs:
            prod = net.products[ndc]
            inv[(sid, ndc)] = float(site.available_qty(ndc, net.today, prod.min_remaining_dating_days))
    return inv


def solve_set_partitioning(plans: Sequence[Plan], orders: Sequence[Order], net: Network,
                           capacity: Optional[Dict[Tuple[str, str], float]] = None,
                           inventory: Optional[Dict[Tuple[str, str], float]] = None,
                           time_limit: float = 30.0, mip_rel_gap: float = 1e-4) -> SolveResult:
    t0 = time.perf_counter()
    order_idx = {o.order_id: k for k, o in enumerate(orders)}
    nP, nI = len(plans), len(orders)
    n = nP + nI                                           # z_s ... then u_i

    c = np.empty(n)
    for k, p in enumerate(plans):
        c[k] = p.cost.total
    c[nP:] = UNASSIGNED_PENALTY

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    # (3) assignment
    for k, p in enumerate(plans):
        for oid in p.order_ids:
            rows.append(order_idx[oid]); cols.append(k); vals.append(1.0)
    for i in range(nI):
        rows.append(i); cols.append(nP + i); vals.append(1.0)
    lb += [1.0] * nI; ub += [1.0] * nI
    r = nI

    # (4) inventory
    inventory = inventory if inventory is not None else site_inventory(net, orders)
    inv_rows: Dict[Tuple[str, str], int] = {}
    for k, p in enumerate(plans):
        for ndc, q in p.qty_by_ndc.items():
            key = (p.site_id, ndc)
            if key not in inv_rows:
                inv_rows[key] = r; r += 1
                lb.append(-np.inf); ub.append(inventory.get(key, 0.0))
            rows.append(inv_rows[key]); cols.append(k); vals.append(float(q))

    # (5) site resource capacity
    if capacity is None:
        capacity = {(sid, res): site.capacity[res] for sid, site in net.sites.items() for res in RESOURCES}
    cap_rows: Dict[Tuple[str, str], int] = {}
    for k, p in enumerate(plans):
        for res, l in p.load.items():
            if l <= 0:
                continue
            key = (p.site_id, res)
            if key not in cap_rows:
                cap_rows[key] = r; r += 1
                lb.append(-np.inf); ub.append(capacity.get(key, 0.0))
            rows.append(cap_rows[key]); cols.append(k); vals.append(float(l))

    A = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
               integrality=np.ones(n), bounds=(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False})
    dt = time.perf_counter() - t0
    if res.x is None:
        return SolveResult([], [o.order_id for o in orders], float("nan"), float("inf"),
                           f"no incumbent ({res.message})", dt, nP, nI)
    x = np.round(res.x).astype(int)
    selected = [plans[k] for k in range(nP) if x[k] == 1]
    unassigned = [orders[i].order_id for i in range(nI) if x[nP + i] == 1]
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    obj = float(sum(p.cost.total for p in selected))
    # Report which inventory/capacity rows are binding (useful for explanations).
    Ax = A @ x
    binding = {}
    for key, row in {**{("INV",) + k: v for k, v in inv_rows.items()},
                     **{("CAP",) + k: v for k, v in cap_rows.items()}}.items():
        if np.isfinite(ub[row]) and Ax[row] >= ub[row] - 1e-6 and ub[row] > 0:
            binding[":".join(key)] = float(ub[row])
    status = {0: "optimal", 1: "time_limit"}.get(res.status, res.message)
    return SolveResult(selected, unassigned, obj, gap, status, dt, nP, nI, binding)
