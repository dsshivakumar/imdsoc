"""Shipment-plan generation (Section V.A, Algorithm 1 line 5).

A shipment plan ``s`` fixes a set of orders O_s for one patient/address, a
dispensing site j_s, a carrier k_s, a service level m_s and packaging p_s.
Only plans whose every order is eligible at j_s, whose packaging is qualified
for every temperature class in O_s and whose carrier accepts the contents are
generated. Single-order plans are always generated, so consolidation is an
option, not a requirement.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .eligibility import EligibilityResult, carrier_accepts, packaging_qualifies
from .models import Network, Order, TempClass, haversine_miles

# Per-order site workload (minutes) and per-shipment cold pack-out stations.
VERIFY_MIN_PER_ORDER = 3.0
FILL_MIN_PER_ORDER = 4.0
COLD_PACK_LABOR_MIN = 6.0


@dataclass
class Plan:
    plan_id: int
    order_ids: Tuple[str, ...]
    patient_id: str
    site_id: str
    carrier_id: str
    service: str
    mode: str
    pkg_id: str
    distance_miles: float
    temp_classes: Tuple[TempClass, ...]
    hold_hours: float                     # H_p under the forecast ambient profile
    deadline_hours: float                 # D_s, tightest deadline in O_s
    signature: bool
    has_controlled: bool
    weight_kg: float
    qty_by_ndc: Dict[str, int]
    load: Dict[str, float]                # l_{s,r}
    transfer_orders: int                  # orders fulfilled away from origin site
    leaves_usual_site: bool
    rural: bool
    ambient_c: float
    weather_severity: float
    service_base_hours: float
    # ---- filled by predictive models / cost model ---------------------------
    quantiles: Dict[float, float] = field(default_factory=dict)
    alpha: float = 0.1
    q_1_minus_alpha: float = 0.0
    pi_exc: float = 0.0
    pi_late: float = 0.0
    pi_fail: float = 0.0
    cost: Optional["CostBreakdown"] = None  # noqa: F821  (set by cost.py)

    @property
    def n_orders(self) -> int:
        return len(self.order_ids)

    @property
    def is_cold(self) -> bool:
        return any(t.is_cold for t in self.temp_classes)

    @property
    def binding_limit(self) -> float:
        return min(self.deadline_hours, self.hold_hours)

    @property
    def slack_hours(self) -> float:
        return self.binding_limit - self.q_1_minus_alpha

    def describe(self) -> str:
        return (f"site={self.site_id} carrier={self.carrier_id}/{self.service} "
                f"pkg={self.pkg_id} orders={','.join(self.order_ids)}")


class PlanGenerator:
    def __init__(self, max_consolidation: int = 3, consolidate: bool = True):
        self.max_consolidation = max_consolidation
        self.consolidate = consolidate

    def generate(self, orders: Iterable[Order], elig: Dict[str, EligibilityResult],
                 net: Network) -> List[Plan]:
        by_patient: Dict[str, List[Order]] = defaultdict(list)
        for o in orders:
            if elig[o.order_id].feasible:
                by_patient[o.patient_id].append(o)

        plans: List[Plan] = []
        next_id = 0
        for pid, p_orders in by_patient.items():
            max_k = min(len(p_orders), self.max_consolidation) if self.consolidate else 1
            for k in range(1, max_k + 1):
                for subset in itertools.combinations(p_orders, k):
                    for plan in self._plans_for_subset(subset, elig, net):
                        plan.plan_id = next_id
                        next_id += 1
                        plans.append(plan)
        return plans

    def _plans_for_subset(self, subset: Tuple[Order, ...], elig: Dict[str, EligibilityResult],
                          net: Network) -> List[Plan]:
        sites = set(elig[subset[0].order_id].eligible_sites)
        for o in subset[1:]:
            sites &= set(elig[o.order_id].eligible_sites)
        if not sites:
            return []

        patient = net.patients[subset[0].patient_id]
        products = [net.products[o.ndc] for o in subset]
        temp_classes = tuple(sorted({p.temp_class for p in products}, key=lambda t: t.value))
        controlled = {t for t in temp_classes if t != TempClass.AMBIENT}
        has_cs = any(p.cs_schedule is not None for p in products)
        signature = any(p.needs_signature for p in products)
        deadline = min(o.deadline_hours for o in subset)
        product_weight = sum(p.weight_kg * o.qty / 30.0 for p, o in zip(products, subset))
        qty_by_ndc: Dict[str, int] = defaultdict(int)
        for o in subset:
            qty_by_ndc[o.ndc] += o.qty
        ambient = net.ambient(patient.state)
        severity = net.severity(patient.state)
        # The most restrictive controlled class sets the hold time.
        limiting_tc = (TempClass.FROZEN if TempClass.FROZEN in controlled else
                       TempClass.REFRIGERATED if TempClass.REFRIGERATED in controlled else
                       TempClass.CRT if TempClass.CRT in controlled else TempClass.AMBIENT)

        out: List[Plan] = []
        for site_id in sorted(sites):
            site = net.sites[site_id]
            dist = haversine_miles(site.lat, site.lon, patient.lat, patient.lon)
            transfers = sum(1 for o in subset if o.origin_site != site_id)
            load = {
                "pharmacist_verify_min": VERIFY_MIN_PER_ORDER * len(subset),
                "tech_fill_min": FILL_MIN_PER_ORDER * len(subset),
                "cold_packout": 1.0 if limiting_tc.is_cold else 0.0,
            }
            for pkg_id in sorted(site.packaging):
                pkg = net.packaging[pkg_id]
                if not packaging_qualifies(pkg, set(temp_classes)):
                    continue
                # Prune dominated options: ambient-only shipments never need a
                # temperature-controlled box.
                if not controlled and set(pkg.temp_classes) != {TempClass.AMBIENT}:
                    continue
                hold = pkg.hold_hours(limiting_tc, ambient)
                for carrier in net.carriers.values():
                    if carrier_accepts(carrier, pkg, has_cs):
                        continue
                    for svc in carrier.services:
                        if dist > svc.max_miles:
                            continue
                        out.append(Plan(
                            plan_id=-1,
                            order_ids=tuple(o.order_id for o in subset),
                            patient_id=patient.patient_id,
                            site_id=site_id,
                            carrier_id=carrier.carrier_id,
                            service=svc.name,
                            mode=svc.mode,
                            pkg_id=pkg_id,
                            distance_miles=dist,
                            temp_classes=temp_classes,
                            hold_hours=hold,
                            deadline_hours=deadline,
                            signature=signature,
                            has_controlled=has_cs,
                            weight_kg=product_weight + pkg.weight_kg,
                            qty_by_ndc=dict(qty_by_ndc),
                            load=load,
                            transfer_orders=transfers,
                            leaves_usual_site=site_id != patient.usual_site,
                            rural=patient.rural,
                            ambient_c=ambient,
                            weather_severity=severity,
                            service_base_hours=svc.base_hours,
                        ))
        return out
