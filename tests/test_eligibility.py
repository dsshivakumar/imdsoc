from dataclasses import replace

from imdsoc.eligibility import EligibilityEngine, carrier_accepts, packaging_qualifies
from imdsoc.models import Lot, Order, TempClass


def _order(net, ndc, pid=None, origin=None, **kw):
    pid = pid or next(iter(net.patients))
    pt = net.patients[pid]
    return Order("T1", pid, ndc, 1, 72.0, origin or pt.usual_site, **kw)


def _patient_in(net, state):
    return next(p for p in net.patients.values() if p.state == state)


def test_licensure_eliminates_unlicensed_sites(net):
    pt = _patient_in(net, "CO")
    res = EligibilityEngine().evaluate(_order(net, "P01", pt.patient_id), net)
    assert "AUS" in res.eliminated
    assert any(r.startswith("LICENSURE") for r in res.eliminated["AUS"])


def test_rems_and_limited_distribution(net):
    pt = _patient_in(net, "TX")
    res = EligibilityEngine().evaluate(_order(net, "P11", pt.patient_id), net)   # REMS
    assert any(r.startswith("REMS") for r in res.eliminated["AUS"])
    res = EligibilityEngine().evaluate(_order(net, "P07", pt.patient_id), net)   # LD + frozen
    assert any(r.startswith("LIMITED_DISTRIBUTION") for r in res.eliminated["DAL"])
    assert set(res.eligible_sites) <= {"HOU", "DEN"}


def test_controlled_substance_rules(net):
    pt = _patient_in(net, "TX")
    eng = EligibilityEngine()
    # HOU's DEA registration does not cover C-II.
    o = _order(net, "P08", pt.patient_id, origin="AUS")
    assert any("schedule II" in r for r in eng.evaluate(o, net).eliminated["HOU"])
    # Previously transferred C-II cannot be transferred again.
    o2 = _order(net, "P08", pt.patient_id, origin="AUS", rx_previously_transferred=True)
    assert any("already transferred" in r for r in eng.evaluate(o2, net).eliminated.get("DAL", []))
    # C-IV previously transferred: allowed between pharmacies sharing a real-time database.
    o3 = _order(net, "P09", pt.patient_id, origin="AUS", rx_previously_transferred=True)
    cs = eng.rules["CONTROLLED_SUBSTANCE"]
    assert cs(o3, net.sites["DAL"], net) is None
    assert cs(o3, net.sites["OKC"], net) is not None


def test_inventory_excludes_expired_recalled_and_short_dated(net):
    site = net.sites["AUS"]
    site.lots = [Lot("x1", "P04", 100, expiry_day=30),      # short-dated (< 60 days)
                 Lot("x2", "P04", 100, expiry_day=400, recalled=True),
                 Lot("x3", "P04", 3, expiry_day=400)]
    assert site.available_qty("P04", 0, 60) == 3
    pt = _patient_in(net, "TX")
    o = Order("T", pt.patient_id, "P04", 5, 72, "AUS")
    assert EligibilityEngine().rules["INVENTORY"](o, site, net) is not None


def test_packaging_and_carrier_checks(net):
    P = net.packaging
    assert packaging_qualifies(P["COLD-48"], {TempClass.REFRIGERATED, TempClass.AMBIENT})
    assert not packaging_qualifies(P["COLD-48"], {TempClass.FROZEN})
    assert not packaging_qualifies(P["AMB-STD"], {TempClass.CRT})
    assert not packaging_qualifies(P["COLD-48"], {TempClass.REFRIGERATED, TempClass.CRT})
    assert carrier_accepts(net.carriers["CAR-B"], P["FRZ-DRY-48"], False) is not None
    assert carrier_accepts(net.carriers["CAR-A"], P["FRZ-DRY-48"], True) is None


def test_hold_time_degrades_with_heat(net):
    pkg = net.packaging["COLD-48"]
    assert pkg.hold_hours(TempClass.REFRIGERATED, 35) < pkg.hold_hours(TempClass.REFRIGERATED, 22)
    assert pkg.hold_hours(TempClass.AMBIENT, 40) == float("inf")
