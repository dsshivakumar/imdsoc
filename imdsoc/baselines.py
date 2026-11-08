"""Allocation strategies of Table I, used as evaluation baselines.

All baselines respect the same mandatory eligibility rules as IMDSOC (legal
and clinical constraints are never optional); they differ in *what they
optimize*. They allocate greedily, order by order, without consolidation,
decrementing inventory and site capacity as they go.

| Strategy               | Optimizes                       | Ignores                         |
|------------------------|---------------------------------|---------------------------------|
| fixed_site             | predictability                  | cost, distance, workload, risk  |
| nearest_eligible       | distance                        | inventory age, workload, cost   |
| lowest_shipping_cost   | carrier charge                  | deadline and excursion risk     |
| inventory_only         | stock availability              | distance, workload              |
| rules_only             | feasibility (nominal transit)   | predicted risk and transit time |
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .cost import CostModel
from .eligibility import EligibilityEngine
from .models import RESOURCES, Network, Order, TempClass
from .optimizer import site_inventory
from .plans import Plan, PlanGenerator
from .predictive import PredictorSuite


@dataclass
class BaselineResult:
    selected: List[Plan]
    escalated: Dict[str, List[str]]


def _sop_rank(p: Plan) -> Tuple:
    """Standard operating procedure used by strategies that do not choose shipping:
    national carrier; overnight for cold chain or <=36 h deadlines, else ground;
    seasonal pack-out = cheapest packaging whose hold time at the forecast
    ambient covers 1.5x the nominal service time."""
    want = "overnight" if (p.is_cold or p.deadline_hours <= 36) else "ground"
    short_hold = p.hold_hours < 1.5 * p.service_base_hours
    return (p.carrier_id != "CAR-A", p.service != want, short_hold, p.cost.pack)


class BaselineStrategy:
    name = "baseline"

    def __init__(self, net: Network, predictors: PredictorSuite, cost_model: Optional[CostModel] = None):
        self.net = net
        self.predictors = predictors
        self.cost_model = cost_model or CostModel(net, rate_card=predictors.rate_card)
        self.engine = EligibilityEngine()

    # Subclasses implement: choose(order, plans_for_order) -> plan or None
    def choose(self, order: Order, plans: List[Plan]) -> Optional[Plan]:
        raise NotImplementedError

    def run(self, orders: Sequence[Order],
            capacity: Optional[Dict[Tuple[str, str], float]] = None) -> BaselineResult:
        elig = self.engine.evaluate_all(list(orders), self.net)
        plans = PlanGenerator(consolidate=False).generate(orders, elig, self.net)
        self.predictors.annotate(plans, self.net, 0.02, 0.10)
        self.cost_model.refresh()
        self.cost_model.annotate(plans)
        by_order: Dict[str, List[Plan]] = defaultdict(list)
        for p in plans:
            by_order[p.order_ids[0]].append(p)
        inv = site_inventory(self.net, orders)
        cap = dict(capacity) if capacity is not None else \
            {(sid, r): s.capacity[r] for sid, s in self.net.sites.items() for r in RESOURCES}
        selected, escalated = [], {}
        for o in orders:
            if not elig[o.order_id].feasible:
                escalated[o.order_id] = ["no eligible site"]
                continue
            ok = [p for p in by_order[o.order_id]
                  if inv[(p.site_id, o.ndc)] >= o.qty
                  and all(cap[(p.site_id, r)] >= l for r, l in p.load.items() if l > 0)]
            pick = self.choose(o, ok) if ok else None
            if pick is None:
                escalated[o.order_id] = ["no feasible plan under remaining inventory/capacity"]
                continue
            inv[(pick.site_id, o.ndc)] -= o.qty
            for r, l in pick.load.items():
                cap[(pick.site_id, r)] -= l
            selected.append(pick)
        return BaselineResult(selected, escalated)


class FixedSite(BaselineStrategy):
    name = "fixed_site"

    def choose(self, order, plans):
        usual = self.net.patients[order.patient_id].usual_site
        at = [p for p in plans if p.site_id == usual] or \
             [p for p in plans if p.site_id == order.origin_site] or \
             sorted(plans, key=lambda p: p.distance_miles)[:1]
        site = at[0].site_id
        return min([p for p in plans if p.site_id == site], key=_sop_rank)


class NearestEligible(BaselineStrategy):
    name = "nearest_eligible"

    def choose(self, order, plans):
        d = min(p.distance_miles for p in plans)
        return min([p for p in plans if p.distance_miles == d], key=_sop_rank)


class LowestShippingCost(BaselineStrategy):
    name = "lowest_shipping_cost"

    def choose(self, order, plans):
        return min(plans, key=lambda p: (p.cost.ship, p.cost.pack))


class InventoryOnly(BaselineStrategy):
    name = "inventory_only"

    def choose(self, order, plans):
        prod = self.net.products[order.ndc]
        stock = lambda sid: self.net.sites[sid].available_qty(order.ndc, self.net.today,
                                                               prod.min_remaining_dating_days)
        best = max({p.site_id for p in plans}, key=stock)
        return min([p for p in plans if p.site_id == best], key=_sop_rank)


class RulesOnly(BaselineStrategy):
    """Eligibility plus nominal service-level transit; minimizes deterministic cost."""
    name = "rules_only"

    def choose(self, order, plans):
        ok = [p for p in plans if p.service_base_hours <= p.binding_limit]
        return min(ok or plans, key=lambda p: p.cost.deterministic)


BASELINES = {c.name: c for c in (FixedSite, NearestEligible, LowestShippingCost, InventoryOnly, RulesOnly)}
