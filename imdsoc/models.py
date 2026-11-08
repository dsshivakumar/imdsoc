"""Domain model for IMDSOC (Section III, Table II of the paper).

All entities are plain dataclasses so the decision pipeline can be run
in-process, in tests, or behind a service boundary (Section VII).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


class TempClass(str, Enum):
    AMBIENT = "ambient"
    CRT = "controlled_room_temperature"
    REFRIGERATED = "refrigerated"   # 2-8 C
    FROZEN = "frozen"

    @property
    def is_cold(self) -> bool:
        return self in (TempClass.REFRIGERATED, TempClass.FROZEN)


# Site resources used in capacity constraint (5).
RESOURCES: Tuple[str, ...] = ("pharmacist_verify_min", "tech_fill_min", "cold_packout")


@dataclass(frozen=True)
class Product:
    ndc: str
    name: str
    temp_class: TempClass
    unit_cost: float
    cs_schedule: Optional[int] = None        # 2..5 for DEA schedules, None otherwise
    rems: bool = False                       # requires REMS-certified dispensing site
    limited_distribution: bool = False       # manufacturer-authorized sites only
    serialized: bool = True                  # DSCSA traceability on transfer
    min_remaining_dating_days: int = 30      # minimum shelf life at dispense
    weight_kg: float = 0.2

    @property
    def needs_dry_ice(self) -> bool:
        return self.temp_class == TempClass.FROZEN

    @property
    def needs_signature(self) -> bool:
        return self.cs_schedule is not None or self.rems


@dataclass
class Lot:
    lot_id: str
    ndc: str
    qty: int
    expiry_day: int          # absolute simulation day on which the lot expires
    recalled: bool = False
    reserved: int = 0        # units held by time-limited reservations

    def available(self, today: int, min_dating: int) -> int:
        if self.recalled or self.expiry_day - today < min_dating:
            return 0
        return max(0, self.qty - self.reserved)


@dataclass(frozen=True)
class Packaging:
    pkg_id: str
    temp_classes: FrozenSet[TempClass]
    hold_hours_at_22c: float        # qualified hold time at 22 C ambient
    cost: float
    weight_kg: float
    dry_ice: bool = False

    def hold_hours(self, temp_class: TempClass, ambient_c: float) -> float:
        """Qualified hold time H_p under a forecast ambient profile.

        Ambient-class products have no hold limit. For temperature-controlled
        classes, hold time degrades linearly (2%/C) with ambient temperature above
        22 C and improves modestly below it (bounded), which mirrors the shape
        of typical packaging qualification curves. Replace with the vendor's
        qualification data in production.
        """
        if temp_class == TempClass.AMBIENT:
            return math.inf
        factor = 1.0 - 0.02 * (ambient_c - 22.0)
        return self.hold_hours_at_22c * min(1.2, max(0.35, factor))


@dataclass(frozen=True)
class ServiceLevel:
    name: str                    # e.g. "ground", "2day", "overnight", "same_day"
    mode: str                    # "road", "air", "courier"
    base_hours: float
    base_rate: float
    per_mile: float
    max_miles: float = math.inf  # e.g. courier radius


@dataclass(frozen=True)
class Carrier:
    carrier_id: str
    services: Tuple[ServiceLevel, ...]
    accepts_dry_ice: bool = True
    accepts_controlled: bool = True
    reliability: float = 1.0     # >1 means slower / more variable (ground truth only)


@dataclass
class Site:
    site_id: str
    name: str
    state: str
    lat: float
    lon: float
    region: str
    licensed_states: FrozenSet[str]          # resident + nonresident licenses
    payer_networks: FrozenSet[str]
    rems_certified: FrozenSet[str]           # NDCs with REMS certification
    ld_authorized: FrozenSet[str]            # NDCs with limited-distribution authorization
    dea_schedules: FrozenSet[int]            # DEA registration covers these schedules
    packaging: FrozenSet[str]                # qualified packaging available on site
    capacity: Dict[str, float]               # per planning window, keyed by RESOURCES
    labor_rate_per_min: float = 1.2
    pharmacist_available: bool = True
    shares_rx_database: FrozenSet[str] = frozenset()  # sites sharing a real-time Rx DB
    lots: List[Lot] = field(default_factory=list)

    def usable_lots(self, ndc: str, today: int, min_dating: int) -> List[Lot]:
        """Lots usable for dispensing, first-expiry-first-out order."""
        lots = [l for l in self.lots if l.ndc == ndc and l.available(today, min_dating) > 0]
        return sorted(lots, key=lambda l: l.expiry_day)

    def available_qty(self, ndc: str, today: int, min_dating: int) -> int:
        return sum(l.available(today, min_dating) for l in self.usable_lots(ndc, today, min_dating))


@dataclass(frozen=True)
class Patient:
    patient_id: str
    state: str
    lat: float
    lon: float
    payer: str
    ruca: int                 # Rural-Urban Commuting Area primary code 1..10
    usual_site: str
    zip3: str = ""

    @property
    def rural(self) -> bool:
        # RUCA 1-3 metropolitan; 4-10 micropolitan / small town / rural.
        return self.ruca >= 4


@dataclass(frozen=True)
class Order:
    order_id: str
    patient_id: str
    ndc: str
    qty: int
    deadline_hours: float               # D: hours from planning time to promised delivery
    origin_site: str                    # site that received the prescription
    day: int = 0
    rx_previously_transferred: bool = False
    electronic_rx: bool = True


@dataclass
class Network:
    """Everything the decision pipeline needs about the world at planning time."""
    sites: Dict[str, Site]
    products: Dict[str, Product]
    packaging: Dict[str, Packaging]
    carriers: Dict[str, Carrier]
    patients: Dict[str, Patient]
    today: int = 0
    ambient_forecast_c: Dict[str, float] = field(default_factory=dict)   # by patient state
    weather_severity: Dict[str, float] = field(default_factory=dict)     # 0..1 by patient state

    def ambient(self, state: str) -> float:
        return self.ambient_forecast_c.get(state, 22.0)

    def severity(self, state: str) -> float:
        return self.weather_severity.get(state, 0.0)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
