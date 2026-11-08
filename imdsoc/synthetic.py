"""Synthetic multi-site pharmacy network and ground-truth outcome simulator.

The paper reports no deployment data (Section VIII). This module creates a
reproducible stand-in so that every component and the full evaluation
protocol can be exercised end to end:

* ``build_network`` - eight dispensing sites (retail, central-fill, mail-order,
  specialty, hospital-affiliated) across six states, with licensure, payer,
  REMS, limited-distribution, DEA and packaging attributes;
* ``OrderStream`` - daily prescription orders with deadlines;
* ``World`` - the *hidden* ground truth that generates transit time,
  temperature excursion and failed delivery. Predictive models never see its
  parameters; they learn from logged outcomes.

All numbers are illustrative. Replace this module with real data adapters
(FHIR, PMS, carrier, IoT) for an actual study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .models import (Carrier, Lot, Network, Order, Packaging, Patient, Product,
                     ServiceLevel, Site, TempClass)
from .plans import Plan

STATES = ("TX", "OK", "NM", "CO", "AR", "LA")
REGION_OF_STATE = {"TX": "south_central", "OK": "south_central", "AR": "south_central",
                   "LA": "south_central", "NM": "mountain", "CO": "mountain"}
CITY_CENTERS = {
    "TX": [(30.27, -97.74), (32.78, -96.80), (29.76, -95.37), (29.42, -98.49), (31.76, -106.49)],
    "OK": [(35.47, -97.52), (36.15, -95.99)],
    "NM": [(35.08, -106.65), (32.31, -106.78)],
    "CO": [(39.74, -104.99), (38.83, -104.82)],
    "AR": [(34.75, -92.29), (36.06, -94.16)],
    "LA": [(29.95, -90.07), (30.45, -91.19)],
}
STATE_BOUNDS = {  # rough lat/lon boxes for rural patient placement
    "TX": (27.0, 34.5, -104.0, -94.5), "OK": (34.0, 36.9, -99.5, -95.0),
    "NM": (32.2, 36.8, -108.5, -103.5), "CO": (37.2, 40.8, -108.5, -102.5),
    "AR": (33.2, 36.4, -94.3, -90.5), "LA": (29.5, 32.9, -93.7, -90.5),
}
PAYERS = ("PBM-A", "PBM-B", "PBM-C", "MEDICAID-TX")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def build_products() -> Dict[str, Product]:
    T = TempClass
    items = [
        Product("P01", "Atorvastatin 20 mg tab", T.AMBIENT, 0.10, weight_kg=0.05),
        Product("P02", "Metformin 500 mg tab", T.AMBIENT, 0.05, weight_kg=0.08),
        Product("P03", "Levothyroxine 50 mcg tab", T.CRT, 0.15, weight_kg=0.04),
        Product("P04", "Insulin glargine pen", T.REFRIGERATED, 25.0, weight_kg=0.05,
                min_remaining_dating_days=60),
        Product("P05", "Biologic SC injection (specialty)", T.REFRIGERATED, 1200.0,
                limited_distribution=True, weight_kg=0.1, min_remaining_dating_days=60),
        Product("P06", "GLP-1 agonist pen", T.REFRIGERATED, 250.0, weight_kg=0.06,
                min_remaining_dating_days=45),
        Product("P07", "Frozen specialty product", T.FROZEN, 800.0, limited_distribution=True,
                weight_kg=0.1, min_remaining_dating_days=30),
        Product("P08", "Oxycodone 5 mg tab (C-II)", T.AMBIENT, 0.40, cs_schedule=2, weight_kg=0.04),
        Product("P09", "Alprazolam 0.5 mg tab (C-IV)", T.AMBIENT, 0.20, cs_schedule=4, weight_kg=0.04),
        Product("P10", "Testosterone cypionate vial (C-III)", T.CRT, 30.0, cs_schedule=3, weight_kg=0.05),
        Product("P11", "Isotretinoin 40 mg cap (REMS)", T.AMBIENT, 6.0, rems=True, weight_kg=0.05),
        Product("P12", "Oral oncology agent (limited distribution)", T.CRT, 400.0,
                limited_distribution=True, weight_kg=0.05),
    ]
    return {p.ndc: p for p in items}


# Order quantity (dispensable units) and default promised-delivery hours.
ORDER_PROFILE = {
    "P01": (30, 120), "P02": (60, 120), "P03": (30, 120), "P04": (5, 72),
    "P05": (2, 72), "P06": (4, 72), "P07": (1, 48), "P08": (20, 72),
    "P09": (30, 96), "P10": (1, 96), "P11": (30, 96), "P12": (30, 72),
}
PRODUCT_MIX = {"P01": 0.16, "P02": 0.15, "P03": 0.11, "P04": 0.11, "P05": 0.05, "P06": 0.10,
               "P07": 0.02, "P08": 0.07, "P09": 0.07, "P10": 0.04, "P11": 0.05, "P12": 0.07}


def build_packaging() -> Dict[str, Packaging]:
    T = TempClass
    items = [
        Packaging("AMB-STD", frozenset({T.AMBIENT}), math.inf, 0.80, 0.15),
        Packaging("CRT-INS", frozenset({T.CRT, T.AMBIENT}), 72.0, 4.00, 0.40),
        Packaging("COLD-24", frozenset({T.REFRIGERATED, T.AMBIENT}), 24.0, 9.00, 1.2),
        Packaging("COLD-48", frozenset({T.REFRIGERATED, T.AMBIENT}), 48.0, 16.0, 1.8),
        Packaging("COLD-72", frozenset({T.REFRIGERATED, T.AMBIENT}), 72.0, 26.0, 2.5),
        Packaging("FRZ-DRY-48", frozenset({T.FROZEN}), 48.0, 30.0, 3.0, dry_ice=True),
        Packaging("FRZ-DRY-96", frozenset({T.FROZEN}), 96.0, 48.0, 5.0, dry_ice=True),
    ]
    return {p.pkg_id: p for p in items}


def build_carriers() -> Dict[str, Carrier]:
    ground = ServiceLevel("ground", "road", 24.0, 7.0, 0.004)
    two_day = ServiceLevel("2day", "air", 40.0, 14.0, 0.006)
    overnight = ServiceLevel("overnight", "air", 18.0, 28.0, 0.010)
    same_day = ServiceLevel("same_day", "courier", 4.0, 14.0, 0.45, max_miles=60.0)
    return {
        "CAR-A": Carrier("CAR-A", (ground, two_day, overnight), True, True, reliability=1.0),
        "CAR-B": Carrier("CAR-B", (ServiceLevel("ground", "road", 24.0, 5.5, 0.0035),
                                   ServiceLevel("overnight", "air", 18.0, 23.0, 0.009)),
                         accepts_dry_ice=False, accepts_controlled=True, reliability=1.3),
        "COURIER": Carrier("COURIER", (same_day,), accepts_dry_ice=False,
                           accepts_controlled=True, reliability=1.1),
    }


def build_sites() -> Dict[str, Site]:
    allp = frozenset({"AMB-STD", "CRT-INS", "COLD-24", "COLD-48"})
    spec = allp | {"COLD-72", "FRZ-DRY-48", "FRZ-DRY-96"}
    cap = lambda v, f, c: {"pharmacist_verify_min": v, "tech_fill_min": f, "cold_packout": c}
    sites = [
        Site("AUS", "Austin retail", "TX", 30.27, -97.74, "south_central",
             frozenset({"TX"}), frozenset({"PBM-A", "PBM-B", "MEDICAID-TX"}),
             frozenset(), frozenset(), frozenset({2, 3, 4, 5}), allp, cap(75, 100, 7)),
        Site("DAL", "Dallas central fill", "TX", 32.78, -96.80, "south_central",
             frozenset({"TX", "OK", "AR", "LA"}), frozenset({"PBM-A", "PBM-B", "PBM-C", "MEDICAID-TX"}),
             frozenset({"P11"}), frozenset(), frozenset({2, 3, 4, 5}), allp | {"COLD-72"},
             cap(160, 215, 14), labor_rate_per_min=1.0),
        Site("HOU", "Houston specialty", "TX", 29.76, -95.37, "south_central",
             frozenset({"TX", "LA", "OK", "AR", "NM", "CO"}), frozenset({"PBM-A", "PBM-C", "MEDICAID-TX"}),
             frozenset({"P11"}), frozenset({"P05", "P07", "P12"}), frozenset({3, 4, 5}), spec,
             cap(100, 135, 12), labor_rate_per_min=1.4),
        Site("SAT", "San Antonio retail", "TX", 29.42, -98.49, "south_central",
             frozenset({"TX"}), frozenset({"PBM-B", "PBM-C", "MEDICAID-TX"}),
             frozenset(), frozenset(), frozenset({2, 3, 4, 5}), allp, cap(70, 95, 6)),
        Site("OKC", "Oklahoma City mail order", "OK", 35.47, -97.52, "south_central",
             frozenset(STATES), frozenset({"PBM-A", "PBM-B", "PBM-C"}),
             frozenset({"P11"}), frozenset({"P12"}), frozenset({3, 4, 5}), allp | {"COLD-72"},
             cap(170, 230, 14), labor_rate_per_min=0.95),
        Site("DEN", "Denver specialty", "CO", 39.74, -104.99, "mountain",
             frozenset({"CO", "NM", "TX", "OK"}), frozenset({"PBM-A", "PBM-B", "PBM-C"}),
             frozenset({"P11"}), frozenset({"P05", "P07"}), frozenset({2, 3, 4, 5}), spec,
             cap(95, 125, 9), labor_rate_per_min=1.35),
        Site("ABQ", "Albuquerque retail", "NM", 35.08, -106.65, "mountain",
             frozenset({"NM", "CO"}), frozenset({"PBM-A", "PBM-C"}),
             frozenset(), frozenset(), frozenset({2, 3, 4, 5}), allp, cap(55, 75, 4)),
        Site("LIT", "Little Rock hospital outpatient", "AR", 34.75, -92.29, "south_central",
             frozenset({"AR", "LA", "OK", "TX"}), frozenset({"PBM-A", "PBM-B", "PBM-C"}),
             frozenset(), frozenset({"P05"}), frozenset({2, 3, 4, 5}), allp | {"COLD-72"},
             cap(70, 95, 6)),
    ]
    # Central fill shares a real-time Rx database with the retail stores it serves.
    sites[1].shares_rx_database = frozenset({"AUS", "SAT"})
    sites[0].shares_rx_database = frozenset({"DAL"})
    sites[3].shares_rx_database = frozenset({"DAL"})
    return {s.site_id: s for s in sites}


STOCKED = {  # which sites stock which products
    "AUS": ["P01", "P02", "P03", "P04", "P06", "P08", "P09", "P10"],
    "DAL": ["P01", "P02", "P03", "P04", "P06", "P08", "P09", "P10", "P11"],
    "HOU": ["P04", "P05", "P06", "P07", "P10", "P11", "P12"],
    "SAT": ["P01", "P02", "P03", "P04", "P06", "P08", "P09"],
    "OKC": ["P01", "P02", "P03", "P04", "P06", "P09", "P10", "P11", "P12"],
    "DEN": ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P11"],
    "ABQ": ["P01", "P02", "P03", "P04", "P06", "P08", "P09"],
    "LIT": ["P01", "P02", "P03", "P04", "P05", "P06", "P08", "P09"],
}


def stock_sites(sites: Dict[str, Site], products: Dict[str, Product], rng: np.random.Generator,
                days_of_demand: float = 10.0, today: int = 0) -> None:
    for sid, site in sites.items():
        site.lots = []
        for ndc in STOCKED[sid]:
            per_order, _ = ORDER_PROFILE[ndc]
            base = per_order * max(2.0, days_of_demand * PRODUCT_MIX[ndc] * 12)
            for k in range(int(rng.integers(2, 4))):
                qty = int(base * rng.uniform(0.4, 1.0))
                # Mix of short-dated and fresh lots so FEFO / write-off matter.
                min_d = products[ndc].min_remaining_dating_days
                expiry = today + int(rng.choice([min_d + rng.integers(3, 25), rng.integers(150, 400)],
                                                p=[0.35, 0.65]))
                site.lots.append(Lot(f"{sid}-{ndc}-L{k}", ndc, qty, expiry))


def build_patients(n: int, sites: Dict[str, Site], rng: np.random.Generator) -> Dict[str, Patient]:
    state_w = np.array([0.55, 0.10, 0.08, 0.09, 0.08, 0.10])
    patients = {}
    for i in range(n):
        st = STATES[rng.choice(len(STATES), p=state_w)]
        if rng.random() < 0.72:
            clat, clon = CITY_CENTERS[st][rng.integers(len(CITY_CENTERS[st]))]
            lat, lon = clat + rng.normal(0, 0.15), clon + rng.normal(0, 0.15)
            ruca = int(rng.choice([1, 1, 1, 2, 3]))
        else:
            la0, la1, lo0, lo1 = STATE_BOUNDS[st]
            lat, lon = rng.uniform(la0, la1), rng.uniform(lo0, lo1)
            ruca = int(rng.choice([4, 5, 7, 8, 10]))
        payer = PAYERS[rng.choice(4, p=[0.35, 0.3, 0.25, 0.10])] if st == "TX" else \
            PAYERS[rng.choice(3, p=[0.4, 0.35, 0.25])]
        # Usual site: a site licensed in the patient's state and in network, nearest first.
        cands = [s for s in sites.values() if st in s.licensed_states and payer in s.payer_networks]
        cands.sort(key=lambda s: (s.lat - lat) ** 2 + (s.lon - lon) ** 2)
        usual = cands[0].site_id if cands else "OKC"
        if cands and rng.random() < 0.25:   # some patients use mail order by habit
            usual = cands[min(len(cands) - 1, int(rng.integers(1, 3)))].site_id
        patients[f"PT{i:05d}"] = Patient(f"PT{i:05d}", st, lat, lon, payer, ruca, usual)
    return patients


def build_network(seed: int = 7, n_patients: int = 600) -> Network:
    rng = np.random.default_rng(seed)
    products = build_products()
    sites = build_sites()
    stock_sites(sites, products, rng)
    patients = build_patients(n_patients, sites, rng)
    net = Network(sites, products, build_packaging(), build_carriers(), patients)
    set_weather(net, rng)
    return net


def set_weather(net: Network, rng: np.random.Generator, heat_shift: float = 0.0,
                severity_shift: float = 0.0) -> None:
    base = {"TX": 33, "OK": 31, "NM": 30, "CO": 24, "AR": 31, "LA": 32}
    net.ambient_forecast_c = {s: float(base[s] + rng.normal(0, 4) + heat_shift) for s in STATES}
    net.weather_severity = {s: float(np.clip(rng.beta(1.2, 6) + severity_shift, 0, 1)) for s in STATES}


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
class OrderStream:
    def __init__(self, net: Network, seed: int = 11, patients_per_day: int = 70):
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.patients_per_day = patients_per_day
        self._ndcs = list(PRODUCT_MIX)
        self._p = np.array([PRODUCT_MIX[n] for n in self._ndcs])
        self._p /= self._p.sum()

    def day(self, day: int, surge: float = 1.0) -> List[Order]:
        rng = self.rng
        pids = list(self.net.patients)
        n = int(self.patients_per_day * surge)
        chosen = rng.choice(len(pids), size=min(n, len(pids)), replace=False)
        orders: List[Order] = []
        for idx in chosen:
            pt = self.net.patients[pids[idx]]
            k = int(rng.choice([1, 2, 3], p=[0.58, 0.30, 0.12]))
            ndcs = rng.choice(len(self._ndcs), size=k, replace=False, p=self._p)
            for j in ndcs:
                ndc = self._ndcs[j]
                qty, deadline = ORDER_PROFILE[ndc]
                if rng.random() < 0.10:
                    deadline = 36.0                       # urgent / therapy start
                cs = self.net.products[ndc].cs_schedule is not None
                orders.append(Order(
                    order_id=f"D{day:03d}-{len(orders):04d}",
                    patient_id=pt.patient_id, ndc=ndc, qty=int(qty), deadline_hours=float(deadline),
                    origin_site=pt.usual_site, day=day,
                    rx_previously_transferred=bool(cs and rng.random() < 0.15),
                    electronic_rx=bool(rng.random() < 0.95),
                ))
        return orders


# ---------------------------------------------------------------------------
# Ground truth (hidden from the decision pipeline)
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    transit_hours: float
    excursion: bool
    failed: bool

    def late(self, deadline: float) -> bool:
        return self.transit_hours > deadline


@dataclass
class Scenario:
    """Stress knobs used by discrete-event simulation (Section VIII, stage 2)."""
    carrier_delay: float = 1.0      # multiplicative transit slowdown
    extra_severity: float = 0.0     # extreme weather
    heat_shift: float = 0.0
    fuel_surcharge: float = 0.12
    demand_surge: float = 1.0


class World:
    def __init__(self, net: Network, seed: int = 99, scenario: Optional[Scenario] = None):
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario or Scenario()

    def transit(self, plan: Plan) -> float:
        rng, sc = self.rng, self.scenario
        carrier = self.net.carriers[plan.carrier_id]
        sev = min(1.0, plan.weather_severity + sc.extra_severity)
        rel = carrier.reliability
        if plan.mode == "courier":
            t = 2.0 + plan.distance_miles / 22.0 + rng.exponential(1.0 * rel) * (1 + sev)
        elif plan.mode == "road":
            zones = plan.distance_miles / 420.0
            t = 20.0 + 22.0 * zones + rng.lognormal(0.0, 0.35 * rel) * 4.0 * (1 + 2 * sev)
            if plan.rural:
                t += rng.choice([0.0, 10.0], p=[0.65, 0.35])
        else:  # air
            t = plan.service_base_hours - 2.0 + rng.gamma(2.0, 1.5 * rel) * (1 + sev)
            if plan.rural:
                t += rng.choice([0.0, 6.0, 24.0], p=[0.6, 0.3, 0.1])
        if rng.random() < 0.25 * sev:      # weather disruption tail
            t += 24.0
        return float(t * sc.carrier_delay)

    def sample(self, plan: Plan) -> Outcome:
        t = self.transit(plan)
        rng = self.rng
        if plan.is_cold or any(tc == TempClass.CRT for tc in plan.temp_classes):
            margin = (t - plan.hold_hours) / 3.0
            p_exc = 0.004 + 0.95 / (1.0 + math.exp(-margin))
            if plan.ambient_c > 36:
                p_exc += 0.01
        else:
            p_exc = 0.0
        sev = min(1.0, plan.weather_severity + self.scenario.extra_severity)
        p_fail = (0.008 + 0.02 * plan.rural + 0.015 * plan.signature + 0.03 * sev
                  - 0.004 * (plan.mode == "courier"))
        return Outcome(t, bool(rng.random() < p_exc), bool(rng.random() < max(0.0, p_fail)))
