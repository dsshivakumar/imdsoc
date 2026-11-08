"""Unit-consistent monetary objective, Eq. (1).

    c_s = C_ship + C_pack + C_labor + C_xfer + C_inv
          + pi_exc * L_exc + pi_late * L_late + pi_fail * L_fail + rho * e_s

Every term is in dollars. Consequence costs L and the carbon price rho are
policy parameters (Section IX): they embed value judgments and must be set
transparently by accountable staff. Defaults below are illustrative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .models import Network, TempClass
from .plans import COLD_PACK_LABOR_MIN, Plan
from .predictive import RateCard

# kg CO2e per tonne-km by mode. Placeholder values in the spirit of ISO 14083 /
# GLEC; replace with carrier-reported or GLEC default factors.
EMISSION_FACTORS = {"road": 0.45, "air": 1.30, "courier": 0.60}
LAST_MILE_KG = 0.25
CIRCUITY = 1.2


@dataclass
class PolicyParameters:
    carbon_price_per_kg: float = 0.19            # rho, $/kg CO2e
    patient_impact_late: float = 40.0
    patient_impact_late_cold_or_specialty: float = 150.0
    patient_impact_excursion: float = 150.0
    patient_impact_failed: float = 60.0
    failed_handling: float = 25.0
    transfer_fee: float = 3.50                   # admin cost per transferred Rx
    dscsa_record_fee: float = 0.50               # serialized-product traceability record
    inventory_weight: float = 0.5                # C_inv: expected-loss weight on at-risk short-dated value
    inventory_horizon_days: int = 30             # short-dated stock within this many days of unusable is at risk
    pharmacist_premium: float = 1.6              # pharmacist minute vs technician minute


@dataclass
class CostBreakdown:
    ship: float
    pack: float
    labor: float
    xfer: float
    inv: float
    exc: float          # pi_exc * L_exc
    late: float         # pi_late * L_late
    fail: float         # pi_fail * L_fail
    carbon: float       # rho * e_s
    emissions_kg: float
    L_exc: float
    L_late: float
    L_fail: float

    @property
    def total(self) -> float:
        return (self.ship + self.pack + self.labor + self.xfer + self.inv
                + self.exc + self.late + self.fail + self.carbon)

    @property
    def deterministic(self) -> float:
        return self.ship + self.pack + self.labor + self.xfer + self.inv

    def components(self) -> Dict[str, float]:
        return {"shipping": self.ship, "packaging": self.pack, "labor": self.labor,
                "transfer": self.xfer, "inventory": self.inv, "excursion_risk": self.exc,
                "late_risk": self.late, "failed_delivery_risk": self.fail, "carbon": self.carbon}

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d


def emissions_kg(plan: Plan) -> float:
    tkm = plan.weight_kg / 1000.0 * plan.distance_miles * 1.609 * CIRCUITY
    return EMISSION_FACTORS[plan.mode] * tkm + LAST_MILE_KG


class CostModel:
    def __init__(self, net: Network, policy: PolicyParameters | None = None,
                 rate_card: RateCard | None = None):
        self.net = net
        self.policy = policy or PolicyParameters()
        self.rate_card = rate_card or RateCard()
        self._min_expiry = self._network_min_expiry()

    def refresh(self) -> None:
        self._min_expiry = self._network_min_expiry()

    def _network_min_expiry(self) -> Dict[str, int]:
        """Earliest usable expiry for each drug anywhere in the network."""
        out: Dict[str, int] = {}
        for site in self.net.sites.values():
            for lot in site.lots:
                prod = self.net.products[lot.ndc]
                if lot.available(self.net.today, prod.min_remaining_dating_days) > 0:
                    out[lot.ndc] = min(out.get(lot.ndc, 10 ** 9), lot.expiry_day)
        return out

    def product_value(self, plan: Plan, temp_controlled_only: bool = False) -> float:
        v = 0.0
        for ndc, q in plan.qty_by_ndc.items():
            prod = self.net.products[ndc]
            if temp_controlled_only and prod.temp_class == TempClass.AMBIENT:
                continue
            v += prod.unit_cost * q
        return v

    def inventory_cost(self, plan: Plan) -> float:
        """Penalize using fresher stock when shorter-dated usable stock exists elsewhere.

        If the network's shortest-dated usable lot of a drug is close to becoming
        undispensable and this plan would draw from a later-dated lot instead,
        the plan is charged an expected write-off cost proportional to the value
        drawn and to how soon the short-dated lot runs out of dating.
        """
        pol, site = self.policy, self.net.sites[plan.site_id]
        cost = 0.0
        for ndc, q in plan.qty_by_ndc.items():
            prod = self.net.products[ndc]
            lots = site.usable_lots(ndc, self.net.today, prod.min_remaining_dating_days)
            e_min = self._min_expiry.get(ndc)
            if not lots or e_min is None or lots[0].expiry_day <= e_min:
                continue
            days_left = e_min - self.net.today - prod.min_remaining_dating_days
            urgency = min(1.0, max(0.0, 1.0 - days_left / pol.inventory_horizon_days))
            cost += pol.inventory_weight * prod.unit_cost * q * urgency
        return cost

    def evaluate(self, plan: Plan) -> CostBreakdown:
        pol, net = self.policy, self.net
        site = net.sites[plan.site_id]
        ship = self.rate_card.price(plan, net)
        pack = net.packaging[plan.pkg_id].cost
        labor = site.labor_rate_per_min * (
            plan.load["pharmacist_verify_min"] * pol.pharmacist_premium
            + plan.load["tech_fill_min"]
            + COLD_PACK_LABOR_MIN * plan.load["cold_packout"])
        xfer = plan.transfer_orders * (pol.transfer_fee + pol.dscsa_record_fee)
        inv = self.inventory_cost(plan)

        value_tc = self.product_value(plan, temp_controlled_only=True)
        specialty = plan.is_cold or any(net.products[n].limited_distribution or net.products[n].rems
                                        for n in plan.qty_by_ndc)
        L_exc = value_tc + ship + pack + pol.patient_impact_excursion
        L_late = (pol.patient_impact_late_cold_or_specialty if specialty else pol.patient_impact_late)
        L_fail = ship + pack + pol.failed_handling + pol.patient_impact_failed + 0.5 * value_tc
        e = emissions_kg(plan)
        return CostBreakdown(ship, pack, labor, xfer, inv,
                             plan.pi_exc * L_exc, plan.pi_late * L_late, plan.pi_fail * L_fail,
                             pol.carbon_price_per_kg * e, e, L_exc, L_late, L_fail)

    def annotate(self, plans: List[Plan]) -> None:
        for p in plans:
            p.cost = self.evaluate(p)
