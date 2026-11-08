"""Deterministic eligibility layer (Section IV, "Eligibility then ranking").

Every rule is a pure function that returns ``None`` when the site passes, or a
short human-readable reason when it eliminates the site. Rules are never
learned: they encode licensure, payer, REMS / limited-distribution,
controlled-substance, inventory, and pharmacist requirements, and are meant to
be maintained by compliance staff (Section IX).

The controlled-substance rule implements a *simplified, configurable* reading
of 21 CFR Part 1306 transfer provisions. It is illustrative and must be
reviewed by compliance staff before any operational use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import Network, Order, Packaging, Carrier, Site, TempClass

Rule = Callable[[Order, Site, Network], Optional[str]]


# --------------------------------------------------------------------------
# Order x site rules
# --------------------------------------------------------------------------
def rule_licensure(order: Order, site: Site, net: Network) -> Optional[str]:
    patient = net.patients[order.patient_id]
    if patient.state not in site.licensed_states:
        return f"not licensed to ship into {patient.state} (no resident/nonresident license)"
    return None


def rule_payer_network(order: Order, site: Site, net: Network) -> Optional[str]:
    patient = net.patients[order.patient_id]
    if patient.payer not in site.payer_networks:
        return f"not in payer/PBM network {patient.payer}"
    return None


def rule_rems(order: Order, site: Site, net: Network) -> Optional[str]:
    product = net.products[order.ndc]
    if product.rems and order.ndc not in site.rems_certified:
        return f"not REMS-certified for {product.name}"
    return None


def rule_limited_distribution(order: Order, site: Site, net: Network) -> Optional[str]:
    product = net.products[order.ndc]
    if product.limited_distribution and order.ndc not in site.ld_authorized:
        return f"not in manufacturer limited-distribution network for {product.name}"
    return None


def rule_controlled_substance(order: Order, site: Site, net: Network) -> Optional[str]:
    product = net.products[order.ndc]
    sched = product.cs_schedule
    if sched is None:
        return None
    if sched not in site.dea_schedules:
        return f"DEA registration does not cover schedule {roman(sched)}"
    if site.site_id == order.origin_site:
        return None
    # Cross-site fulfillment of a controlled prescription = a transfer.
    shared_db = order.origin_site in site.shares_rx_database
    if sched == 2:
        # Simplified: one-time transfer of an electronic C-II prescription for
        # initial fill; no transfer otherwise.
        if not order.electronic_rx:
            return "C-II paper prescription cannot be transferred"
        if order.rx_previously_transferred:
            return "C-II prescription already transferred once"
        return None
    # Schedules III-V: one-time transfer unless pharmacies share a real-time
    # online database.
    if order.rx_previously_transferred and not shared_db:
        return f"C-{roman(sched)} prescription already transferred (no shared real-time database)"
    return None


def rule_inventory(order: Order, site: Site, net: Network) -> Optional[str]:
    product = net.products[order.ndc]
    qty = site.available_qty(order.ndc, net.today, product.min_remaining_dating_days)
    if qty < order.qty:
        return (f"insufficient valid inventory ({qty} < {order.qty} units non-expired, "
                f"non-recalled, unreserved with >= {product.min_remaining_dating_days} days dating)")
    return None


def rule_packaging_available(order: Order, site: Site, net: Network) -> Optional[str]:
    tc = net.products[order.ndc].temp_class
    if not any(tc in net.packaging[p].temp_classes for p in site.packaging):
        return f"no qualified packaging on site for {tc.value}"
    return None


def rule_pharmacist(order: Order, site: Site, net: Network) -> Optional[str]:
    if not site.pharmacist_available:
        return "no pharmacist available for verification"
    return None


DEFAULT_RULES: Dict[str, Rule] = {
    "LICENSURE": rule_licensure,
    "PAYER_NETWORK": rule_payer_network,
    "REMS": rule_rems,
    "LIMITED_DISTRIBUTION": rule_limited_distribution,
    "CONTROLLED_SUBSTANCE": rule_controlled_substance,
    "INVENTORY": rule_inventory,
    "PACKAGING": rule_packaging_available,
    "PHARMACIST": rule_pharmacist,
}


@dataclass
class EligibilityResult:
    order_id: str
    eligible_sites: List[str] = field(default_factory=list)
    eliminated: Dict[str, List[str]] = field(default_factory=dict)   # site -> ["RULE: reason"]

    @property
    def feasible(self) -> bool:
        return bool(self.eligible_sites)


class EligibilityEngine:
    """Applies every mandatory rule to every (order, site) pair and records reasons."""

    def __init__(self, rules: Optional[Dict[str, Rule]] = None):
        self.rules = dict(rules or DEFAULT_RULES)

    def evaluate(self, order: Order, net: Network) -> EligibilityResult:
        res = EligibilityResult(order.order_id)
        for site in net.sites.values():
            reasons = []
            for code, rule in self.rules.items():
                msg = rule(order, site, net)
                if msg:
                    reasons.append(f"{code}: {msg}")
            if reasons:
                res.eliminated[site.site_id] = reasons
            else:
                res.eligible_sites.append(site.site_id)
        return res

    def evaluate_all(self, orders: List[Order], net: Network) -> Dict[str, EligibilityResult]:
        return {o.order_id: self.evaluate(o, net) for o in orders}

    def is_eligible(self, order: Order, site: Site, net: Network) -> bool:
        return all(rule(order, site, net) is None for rule in self.rules.values())


# --------------------------------------------------------------------------
# Shipment-level checks (packaging qualification and carrier acceptance)
# --------------------------------------------------------------------------
def packaging_qualifies(pkg: Packaging, temp_classes: set) -> bool:
    """Packaging must be qualified for *every* temperature class in the shipment.

    Ambient items may ride in any temperature-controlled box; a cold box cannot
    carry frozen product and vice versa (no mixed cold classes in one box).
    """
    controlled = {t for t in temp_classes if t != TempClass.AMBIENT}
    return controlled <= set(pkg.temp_classes) and (
        bool(controlled) or TempClass.AMBIENT in pkg.temp_classes
    )


def carrier_accepts(carrier: Carrier, pkg: Packaging, has_controlled: bool) -> Optional[str]:
    if pkg.dry_ice and not carrier.accepts_dry_ice:
        return "carrier does not accept dry ice (49 CFR 173.217)"
    if has_controlled and not carrier.accepts_controlled:
        return "carrier does not accept controlled substances"
    return None


def roman(n: int) -> str:
    return {2: "II", 3: "III", 4: "IV", 5: "V"}.get(n, str(n))
