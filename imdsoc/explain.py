"""Grounded explanations (Section VI).

Each recommendation states the selected plan, the rules that eliminated every
other site, and the Eq. (1) cost components for the chosen plan and the best
alternative, so the reason for a choice can be read off directly, e.g.
"Site B: $18 lower expected cost, of which $11 is lower excursion risk".

A ``Narrator`` turns the structured explanation into prose. The default is a
deterministic template. An LLM / retrieval-augmented narrator can be plugged
in (e.g. Amazon Bedrock with a knowledge base of carrier contracts, SOPs and
payer policies), but it is advisory: it only rewrites the structured facts
and can never authorize dispensing or override a rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from .eligibility import EligibilityResult
from .plans import Plan


@dataclass
class Explanation:
    order_id: str
    selected: Dict
    eliminated_sites: Dict[str, List[str]]
    chance_rejected: int
    alternative: Optional[Dict] = None
    delta_components: Dict[str, float] = field(default_factory=dict)
    headline: str = ""
    escalation_reasons: List[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def _plan_summary(p: Plan) -> Dict:
    return {
        "plan_id": p.plan_id, "site": p.site_id, "carrier": p.carrier_id, "service": p.service,
        "packaging": p.pkg_id, "orders": list(p.order_ids), "consolidated": p.n_orders > 1,
        "expected_cost": round(p.cost.total, 2),
        "components": {k: round(v, 2) for k, v in p.cost.components().items()},
        "q_1_minus_alpha_h": round(p.q_1_minus_alpha, 1), "alpha": p.alpha,
        "limit_h": round(p.binding_limit, 1) if p.binding_limit < 1e9 else None,
        "pi_exc": round(p.pi_exc, 4), "pi_late": round(p.pi_late, 4), "pi_fail": round(p.pi_fail, 4),
    }


PRETTY = {"shipping": "shipping", "packaging": "packaging", "labor": "labor",
          "transfer": "transfer", "inventory": "inventory-age", "excursion_risk": "excursion risk",
          "late_risk": "late-delivery risk", "failed_delivery_risk": "failed-delivery risk",
          "carbon": "emissions"}


def explain(order_id: str, selected: Plan, candidates: List[Plan], elig: EligibilityResult,
            chance_rejected: int) -> Explanation:
    ex = Explanation(order_id, _plan_summary(selected), dict(elig.eliminated), chance_rejected)
    # Best alternative: cheapest feasible plan for this order at a different site,
    # else a different carrier/service/packaging at the same site.
    others = [p for p in candidates if order_id in p.order_ids and p.plan_id != selected.plan_id]
    diff_site = [p for p in others if p.site_id != selected.site_id]
    pool = diff_site or others
    if pool:
        # Compare per order so consolidated and single plans are comparable.
        alt = min(pool, key=lambda p: p.cost.total / p.n_orders)
        ex.alternative = _plan_summary(alt)
        s_c, a_c = selected.cost.components(), alt.cost.components()
        n_s, n_a = selected.n_orders, alt.n_orders
        ex.delta_components = {k: round(a_c[k] / n_a - s_c[k] / n_s, 2) for k in s_c}
        total = round(alt.cost.total / n_a - selected.cost.total / n_s, 2)
        top = max(ex.delta_components, key=lambda k: ex.delta_components[k])
        where = (f"Site {selected.site_id}" if alt.site_id != selected.site_id
                 else f"{selected.carrier_id}/{selected.service}/{selected.pkg_id}")
        vs = (f"site {alt.site_id}" if alt.site_id != selected.site_id
              else f"{alt.carrier_id}/{alt.service}/{alt.pkg_id}")
        if total >= 0:
            ex.headline = (f"{where}: ${total:,.2f} lower expected cost per order than {vs}, "
                           f"of which ${ex.delta_components[top]:,.2f} is lower {PRETTY[top]}")
        else:
            ex.headline = (f"{where}: selected despite ${-total:,.2f} higher stand-alone cost than {vs} "
                           f"because shared inventory/capacity or consolidation lowers network cost")
    else:
        ex.headline = f"Site {selected.site_id}: only feasible plan"
    return ex


class Narrator(Protocol):
    def narrate(self, ex: Explanation) -> str: ...


class TemplateNarrator:
    def narrate(self, ex: Explanation) -> str:
        s = ex.selected
        lines = [ex.headline + ".",
                 f"Ship from {s['site']} via {s['carrier']} {s['service']} in {s['packaging']}"
                 + (f" (consolidated with {len(s['orders']) - 1} other order(s))" if s["consolidated"] else "")
                 + f"; expected cost ${s['expected_cost']:,.2f}."]
        if s["limit_h"] is not None:
            lines.append(f"Transit {int((1 - s['alpha']) * 100)}th percentile {s['q_1_minus_alpha_h']} h "
                         f"<= limit {s['limit_h']} h (deadline / packaging hold).")
        if ex.eliminated_sites:
            elim = "; ".join(f"{site}: {rs[0].split(': ', 1)[0]}" for site, rs in sorted(ex.eliminated_sites.items()))
            lines.append(f"Ineligible sites -> {elim}.")
        if ex.chance_rejected:
            lines.append(f"{ex.chance_rejected} plan(s) removed by the transit/hold-time chance constraint.")
        if ex.escalation_reasons:
            lines.append("Escalated: " + "; ".join(ex.escalation_reasons) + ".")
        return " ".join(lines)
