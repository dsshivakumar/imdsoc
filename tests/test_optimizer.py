import itertools

import pytest

from imdsoc.chance import chance_filter
from imdsoc.cost import CostModel
from imdsoc.eligibility import EligibilityEngine
from imdsoc.models import Lot, Order
from imdsoc.optimizer import UNASSIGNED_PENALTY, solve_set_partitioning
from imdsoc.plans import PlanGenerator
from imdsoc.reservation import InventoryLedger, ReservationError


def _pipeline(net, suite, orders):
    elig = EligibilityEngine().evaluate_all(orders, net)
    plans = PlanGenerator().generate(orders, elig, net)
    suite.annotate(plans, net, 0.02, 0.10)
    CostModel(net, rate_card=suite.rate_card).annotate(plans)
    return elig, plans


def _tx_patient_with_orders(net, ndcs):
    pt = next(p for p in net.patients.values() if p.state == "TX" and p.usual_site == "AUS")
    return [Order(f"O{k}", pt.patient_id, n, 1, 96.0, "AUS") for k, n in enumerate(ndcs)]


def test_single_order_plans_always_generated_and_consolidation_optional(trained):
    net, suite, _ = trained
    orders = _tx_patient_with_orders(net, ["P01", "P02"])
    _, plans = _pipeline(net, suite, orders)
    sizes = {p.n_orders for p in plans}
    assert sizes == {1, 2}
    for o in orders:
        assert any(p.order_ids == (o.order_id,) for p in plans)


def test_unit_consistent_objective(trained):
    net, suite, _ = trained
    _, plans = _pipeline(net, suite, _tx_patient_with_orders(net, ["P04"]))
    p = plans[0]
    c = p.cost
    assert c.total == pytest.approx(sum(c.components().values()))
    assert c.exc == pytest.approx(p.pi_exc * c.L_exc)
    assert c.late == pytest.approx(p.pi_late * c.L_late)


def test_chance_filter_enforces_quantile_limit(trained):
    net, suite, _ = trained
    _, plans = _pipeline(net, suite, _tx_patient_with_orders(net, ["P04", "P06"]))
    kept, rejected = chance_filter(plans)
    assert all(p.q_1_minus_alpha <= min(p.deadline_hours, p.hold_hours) for p in kept)
    assert all(p.q_1_minus_alpha > min(p.deadline_hours, p.hold_hours) for p in rejected)
    assert all(p.alpha == 0.02 for p in kept if p.is_cold)


def test_milp_assigns_each_order_exactly_once(trained):
    net, suite, _ = trained
    orders = _tx_patient_with_orders(net, ["P01", "P02", "P03"])
    _, plans = _pipeline(net, suite, orders)
    kept, _ = chance_filter(plans)
    res = solve_set_partitioning(kept, orders, net)
    covered = list(itertools.chain.from_iterable(p.order_ids for p in res.selected))
    assert sorted(covered) == sorted(o.order_id for o in orders)
    assert not res.unassigned and res.status == "optimal"
    # MILP is no worse than the cheapest single-order plans taken independently.
    singles = sum(min(p.cost.total for p in kept if p.order_ids == (o.order_id,)) for o in orders)
    assert res.objective <= singles + 1e-6


def test_inventory_constraint_prevents_double_consumption(trained):
    net, suite, _ = trained
    import copy
    net = copy.deepcopy(net)
    for s in net.sites.values():
        s.lots = [l for l in s.lots if l.ndc != "P01"]
    net.sites["DAL"].lots.append(Lot("only", "P01", 30, 500))
    pts = [p for p in net.patients.values() if p.state == "TX"][:2]
    orders = [Order(f"I{k}", pt.patient_id, "P01", 30, 120.0, pt.usual_site) for k, pt in enumerate(pts)]
    _, plans = _pipeline(net, suite, orders)
    res = solve_set_partitioning(chance_filter(plans)[0], orders, net)
    assert len(res.unassigned) == 1          # only one order can draw the 30 units
    assert res.objective < UNASSIGNED_PENALTY


def test_capacity_constraint(trained):
    net, suite, _ = trained
    orders = _tx_patient_with_orders(net, ["P01", "P02", "P03"])
    _, plans = _pipeline(net, suite, orders)
    kept, _ = chance_filter(plans)
    cap = {(sid, r): (0.0 if sid == "AUS" else 1e9) for sid in net.sites
           for r in ("pharmacist_verify_min", "tech_fill_min", "cold_packout")}
    res = solve_set_partitioning(kept, orders, net, capacity=cap)
    assert all(p.site_id != "AUS" for p in res.selected)


def test_reservations_are_atomic_and_expire(net):
    t = [0.0]
    ledger = InventoryLedger(net, clock=lambda: t[0])
    site = net.sites["DAL"]
    site.lots = [Lot("L", "P01", 30, 500)]
    rid = ledger.reserve("A", "DAL", "P01", 30, ttl_s=10)
    with pytest.raises(ReservationError):
        ledger.reserve("B", "DAL", "P01", 1)            # stock is held
    with pytest.raises(ReservationError):
        ledger.reserve("A", "OKC", "P01", 1)            # one active reservation per order
    t[0] = 11.0
    rid2 = ledger.reserve("B", "DAL", "P01", 30)       # expired hold released
    assert ledger.get(rid).state == "expired"
    with pytest.raises(ReservationError):
        ledger.commit(rid)
    ledger.commit(rid2)
    assert site.lots[0].qty == 0 and site.lots[0].reserved == 0
