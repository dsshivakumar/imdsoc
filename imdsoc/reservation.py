"""Time-limited, atomic inventory reservations (Section IV, Fig. 3 steps 5-6, 11).

The orchestrator reserves inventory at the selected site *before* committing
an assignment. Reservations are acquired atomically, expire after a TTL, and
are released on rejection or timeout, so inventory is never committed at two
sites for the same order. In production this is a conditional write with TTL
in the inventory store (e.g. Aurora PostgreSQL row locks / DynamoDB
conditional put); here it is an in-process lock.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .models import Network


class ReservationError(RuntimeError):
    pass


@dataclass
class Reservation:
    reservation_id: str
    order_id: str
    site_id: str
    ndc: str
    allocations: List[Tuple[str, int]]      # (lot_id, qty)
    expires_at: float
    state: str = "held"                     # held | committed | released | expired


class InventoryLedger:
    def __init__(self, net: Network, clock: Callable[[], float] = time.monotonic):
        self.net = net
        self.clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._by_id: Dict[str, Reservation] = {}
        self._active_by_order: Dict[str, str] = {}

    # -- internal -----------------------------------------------------------
    def _lot(self, site_id: str, lot_id: str):
        return next(l for l in self.net.sites[site_id].lots if l.lot_id == lot_id)

    def _expire_locked(self) -> None:
        now = self.clock()
        for r in list(self._by_id.values()):
            if r.state == "held" and r.expires_at <= now:
                self._release_locked(r, "expired")

    def _release_locked(self, r: Reservation, state: str) -> None:
        for lot_id, q in r.allocations:
            self._lot(r.site_id, lot_id).reserved -= q
        r.state = state
        self._active_by_order.pop(r.order_id, None)

    # -- API ----------------------------------------------------------------
    def reserve(self, order_id: str, site_id: str, ndc: str, qty: int, ttl_s: float = 120.0) -> str:
        with self._lock:
            self._expire_locked()
            if order_id in self._active_by_order:
                raise ReservationError(f"order {order_id} already holds reservation "
                                       f"{self._active_by_order[order_id]}")
            prod = self.net.products[ndc]
            site = self.net.sites[site_id]
            lots = site.usable_lots(ndc, self.net.today, prod.min_remaining_dating_days)
            if sum(l.available(self.net.today, prod.min_remaining_dating_days) for l in lots) < qty:
                raise ReservationError(f"insufficient inventory for {ndc} at {site_id}")
            need, allocs = qty, []
            for lot in lots:                     # first-expiry-first-out
                take = min(need, lot.available(self.net.today, prod.min_remaining_dating_days))
                if take > 0:
                    lot.reserved += take
                    allocs.append((lot.lot_id, take))
                    need -= take
                if need == 0:
                    break
            rid = f"R{next(self._ids):07d}"
            self._by_id[rid] = Reservation(rid, order_id, site_id, ndc, allocs, self.clock() + ttl_s)
            self._active_by_order[order_id] = rid
            return rid

    def commit(self, reservation_id: str) -> Reservation:
        with self._lock:
            self._expire_locked()
            r = self._by_id[reservation_id]
            if r.state != "held":
                raise ReservationError(f"reservation {reservation_id} is {r.state}")
            for lot_id, q in r.allocations:
                lot = self._lot(r.site_id, lot_id)
                lot.reserved -= q
                lot.qty -= q
            r.state = "committed"
            self._active_by_order.pop(r.order_id, None)
            return r

    def release(self, reservation_id: str) -> None:
        with self._lock:
            r = self._by_id[reservation_id]
            if r.state == "held":
                self._release_locked(r, "released")

    def get(self, reservation_id: str) -> Optional[Reservation]:
        return self._by_id.get(reservation_id)
