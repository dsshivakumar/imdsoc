"""Decision orchestration: Algorithm 1 (eligibility, plan generation, optimization).

Mirrors the Step Functions workflow of Fig. 2 / Fig. 3 as a single in-process
service: intake -> eligibility -> plan generation -> prediction -> chance
filter -> per-region MILP -> explanation -> gate (assign | human review |
escalate) -> reserve and dispatch. Every decision, alternative and
explanation is appended to an audit log.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .chance import chance_filter
from .cost import CostModel, PolicyParameters
from .eligibility import EligibilityEngine, EligibilityResult, Rule
from .explain import Explanation, Narrator, TemplateNarrator, explain
from .models import RESOURCES, Network, Order
from .optimizer import SolveResult, site_inventory, solve_set_partitioning
from .plans import Plan, PlanGenerator
from .predictive import PredictorSuite
from .reservation import InventoryLedger, ReservationError
from .synthetic import REGION_OF_STATE


@dataclass
class IMDSOCConfig:
    alpha_cold: float = 0.02            # chance-constraint risk level, cold chain
    alpha_other: float = 0.10
    chance_constraint: bool = True      # False -> point (median) estimates (ablation)
    consolidate: bool = True
    max_consolidation: int = 3
    time_limit_s: float = 30.0
    capacity_headroom: float = 0.85     # plan to this share of Cap_jr; buffer absorbs workload variability
    gap_tolerance: float = 0.01
    slack_margin_h: float = 2.0
    exc_threshold: float = 0.05
    escalate_on_site_change: bool = True
    reservation_ttl_s: float = 120.0


@dataclass
class Decision:
    order_id: str
    status: str                          # assigned | review | escalated
    plan: Optional[Plan]
    explanation: Optional[Explanation]
    reasons: List[str] = field(default_factory=list)
    reservation_ids: List[str] = field(default_factory=list)


@dataclass
class PlanningResult:
    decisions: Dict[str, Decision]
    solves: List[SolveResult]
    n_plans_generated: int
    n_plans_after_chance: int
    seconds: float

    @property
    def selected_plans(self) -> List[Plan]:
        seen, out = set(), []
        for d in self.decisions.values():
            if d.plan is not None and d.plan.plan_id not in seen:
                seen.add(d.plan.plan_id)
                out.append(d.plan)
        return out


class Orchestrator:
    def __init__(self, net: Network, predictors: PredictorSuite,
                 config: Optional[IMDSOCConfig] = None,
                 policy: Optional[PolicyParameters] = None,
                 rules: Optional[Dict[str, Rule]] = None,
                 narrator: Optional[Narrator] = None,
                 region_of: Optional[Callable[[Order], str]] = None):
        self.net = net
        self.predictors = predictors
        self.config = config or IMDSOCConfig()
        self.eligibility = EligibilityEngine(rules)
        self.cost_model = CostModel(net, policy, predictors.rate_card)
        self.narrator = narrator or TemplateNarrator()
        self.ledger = InventoryLedger(net)
        self.region_of = region_of or (lambda o: REGION_OF_STATE[net.patients[o.patient_id].state])
        self.audit_log: List[Dict] = []

    # ------------------------------------------------------------------
    def plan(self, orders: Sequence[Order], execute: bool = True) -> PlanningResult:
        """Run Algorithm 1 on the orders of one planning window."""
        t0 = time.perf_counter()
        cfg = self.config
        self._order_lookup = {o.order_id: (o.ndc, o.qty) for o in orders}
        # Line 2-4: eliminate ineligible sites, record reasons.
        elig = self.eligibility.evaluate_all(list(orders), self.net)
        # Line 5: generate plans, predict, keep plans satisfying Eq. (6).
        gen = PlanGenerator(cfg.max_consolidation, cfg.consolidate)
        plans = gen.generate(orders, elig, self.net)
        self.predictors.annotate(plans, self.net, cfg.alpha_cold, cfg.alpha_other,
                                 point_estimate=not cfg.chance_constraint)
        self.cost_model.refresh()
        self.cost_model.annotate(plans)
        kept, rejected = chance_filter(plans)
        rejected_count = defaultdict(int)
        for p in rejected:
            for oid in p.order_ids:
                rejected_count[oid] += 1

        # Line 6: solve per region, sequentially sharing inventory and capacity.
        capacity = {(sid, r): s.capacity[r] * cfg.capacity_headroom
                    for sid, s in self.net.sites.items() for r in RESOURCES}
        inventory = site_inventory(self.net, orders)
        by_region: Dict[str, List[Order]] = defaultdict(list)
        for o in orders:
            by_region[self.region_of(o)].append(o)
        order_region = {o.order_id: self.region_of(o) for o in orders}
        solves: List[SolveResult] = []
        selected: List[Plan] = []
        unassigned: List[str] = []
        for region, r_orders in sorted(by_region.items()):
            r_ids = {o.order_id for o in r_orders}
            r_plans = [p for p in kept if p.order_ids[0] in r_ids]
            res = solve_set_partitioning(r_plans, r_orders, self.net, capacity, inventory,
                                         time_limit=cfg.time_limit_s)
            solves.append(res)
            selected += res.selected
            unassigned += res.unassigned
            for p in res.selected:
                for ndc, q in p.qty_by_ndc.items():
                    inventory[(p.site_id, ndc)] -= q
                for r, l in p.load.items():
                    capacity[(p.site_id, r)] -= l

        gap_by_region = {region: s.gap for region, s in zip(sorted(by_region), solves)}
        decisions: Dict[str, Decision] = {}
        # Lines 7-8: orders with no feasible plan -> escalate with eliminating rules.
        for oid in unassigned:
            decisions[oid] = Decision(oid, "escalated", None, None,
                                      self._infeasibility_reasons(oid, elig[oid], rejected_count[oid], plans))
        # Lines 9-13: gate each selected plan.
        for p in selected:
            reasons = self._escalation_reasons(p, gap_by_region[order_region[p.order_ids[0]]])
            for oid in p.order_ids:
                ex = explain(oid, p, kept, elig[oid], rejected_count[oid])
                ex.escalation_reasons = reasons
                ex.narrative = self.narrator.narrate(ex)
                decisions[oid] = Decision(oid, "review" if reasons else "assigned", p, ex, reasons)

        if execute:
            self._execute(selected, decisions)
        result = PlanningResult(decisions, solves, len(plans), len(kept), time.perf_counter() - t0)
        self._log(result)
        return result

    # ------------------------------------------------------------------
    def _escalation_reasons(self, p: Plan, gap: float) -> List[str]:
        cfg, out = self.config, []
        if p.binding_limit < 1e9 and p.slack_hours < cfg.slack_margin_h:
            out.append(f"chance-constraint slack {p.slack_hours:.1f} h below margin {cfg.slack_margin_h} h")
        if p.pi_exc > cfg.exc_threshold:
            out.append(f"predicted excursion probability {p.pi_exc:.3f} above {cfg.exc_threshold}")
        if cfg.escalate_on_site_change and p.leaves_usual_site:
            out.append("changes patient's usual dispensing pharmacy (notification/consent)")
        if gap > cfg.gap_tolerance:
            out.append(f"optimality gap {gap:.3%} above tolerance")
        return out

    def _infeasibility_reasons(self, oid: str, er: EligibilityResult, n_chance_rejected: int,
                               plans: List[Plan]) -> List[str]:
        if not er.feasible:
            return [f"no eligible site; {site}: {'; '.join(r)}" for site, r in sorted(er.eliminated.items())]
        if n_chance_rejected and not any(oid in p.order_ids for p in plans
                                         if p.q_1_minus_alpha <= p.binding_limit):
            return ["eligible sites exist but no plan meets the deadline / packaging hold-time "
                    f"chance constraint ({n_chance_rejected} plans rejected)"]
        return ["eligible plans exist but shared inventory or site capacity is exhausted in this window"]

    def _execute(self, selected: List[Plan], decisions: Dict[str, Decision]) -> None:
        """Reserve atomically per plan, then commit (Fig. 3 steps 5-11)."""
        for p in selected:
            held: List[str] = []
            try:
                for oid in p.order_ids:
                    ndc, qty = self._order_lookup[oid]
                    held.append(self.ledger.reserve(oid, p.site_id, ndc, qty, self.config.reservation_ttl_s))
            except ReservationError as e:
                for rid in held:
                    self.ledger.release(rid)
                for oid in p.order_ids:
                    d = decisions[oid]
                    d.status, d.reasons = "escalated", d.reasons + [f"reservation failed: {e}"]
                continue
            for oid, rid in zip(p.order_ids, held):
                self.ledger.commit(rid)
                decisions[oid].reservation_ids.append(rid)

    def _log(self, result: PlanningResult) -> None:
        for d in result.decisions.values():
            self.audit_log.append({
                "ts": time.time(), "order_id": d.order_id, "status": d.status, "reasons": d.reasons,
                "plan": d.explanation.selected if d.explanation else None,
                "alternative": d.explanation.alternative if d.explanation else None,
                "eliminated_sites": d.explanation.eliminated_sites if d.explanation else None,
                "explanation": d.explanation.narrative if d.explanation else None,
            })

    def write_audit_log(self, path: str) -> None:
        with open(path, "w") as f:
            for row in self.audit_log:
                f.write(json.dumps(row, default=str) + "\n")

    def run(self, orders: Sequence[Order], execute: bool = True) -> PlanningResult:
        return self.plan(orders, execute)

    # ------------------------------------------------------------------
    def resolve_exception(self, orders: Sequence[Order], *, down_sites: Iterable[str] = (),
                          note: str = "") -> PlanningResult:
        """Common exception path: re-solve affected orders over the updated feasible set.

        ``down_sites`` (stockout, outage, cold-chain alarm at a site) are removed
        by an extra deterministic rule for this re-solve only.
        """
        down = set(down_sites)
        extra = dict(self.eligibility.rules)
        extra["EXCEPTION"] = lambda o, s, n: (f"excluded by exception: {note or 'site unavailable'}"
                                              if s.site_id in down else None)
        saved = self.eligibility
        self.eligibility = EligibilityEngine(extra)
        try:
            return self.run(orders)
        finally:
            self.eligibility = saved
