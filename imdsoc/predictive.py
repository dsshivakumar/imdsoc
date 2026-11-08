"""Predictive models (Section VI, Table III).

* Transit time  - quantile gradient boosting [Koenker & Bassett; LightGBM-style]
  gives Q_{1-alpha}(T_s) for the chance constraint (6) and, by inverting the
  predicted quantile curve, pi_late = P(T_s > D_s).
* Excursion risk - boosted classifier with isotonic calibration -> pi_exc.
* Failed delivery - boosted classifier with isotonic calibration -> pi_fail.
* Shipping cost - deterministic rate card plus surcharges -> C_ship.

Probabilities are calibrated on held-out folds because they are multiplied by
dollar consequence costs in Eq. (1). Models only supply inputs; they never
decide feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .models import Network
from .plans import Plan

QUANTILES: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99)
MODE_CODE = {"road": 0, "air": 1, "courier": 2}


def plan_features(plans: Sequence[Plan], net: Network) -> np.ndarray:
    carriers = sorted(net.carriers)
    rows = []
    for p in plans:
        rows.append([
            p.distance_miles,
            p.service_base_hours,
            MODE_CODE[p.mode],
            carriers.index(p.carrier_id),
            float(p.rural),
            p.weather_severity,
            p.ambient_c,
            float(p.is_cold),
            min(p.hold_hours, 500.0),
            float(p.signature),
            float(p.n_orders),
        ])
    return np.asarray(rows, dtype=float)


FEATURE_NAMES = ["distance_miles", "service_base_hours", "mode", "carrier", "rural",
                 "weather_severity", "ambient_c", "is_cold", "hold_hours", "signature", "n_orders"]


# ---------------------------------------------------------------------------
# Shipping cost: rate cards plus surcharge model
# ---------------------------------------------------------------------------
@dataclass
class RateCard:
    fuel_surcharge: float = 0.12
    dry_ice_fee: float = 12.0
    signature_fee: float = 6.0
    rural_das_fee: float = 4.5          # delivery-area surcharge
    per_kg_over_1: float = 1.5

    def price(self, plan: Plan, net: Network) -> float:
        svc = next(s for s in net.carriers[plan.carrier_id].services if s.name == plan.service)
        base = svc.base_rate + svc.per_mile * plan.distance_miles
        base += self.per_kg_over_1 * max(0.0, plan.weight_kg - 1.0)
        if net.packaging[plan.pkg_id].dry_ice:
            base += self.dry_ice_fee
        if plan.signature:
            base += self.signature_fee
        if plan.rural:
            base += self.rural_das_fee
        return base * (1.0 + self.fuel_surcharge)


# ---------------------------------------------------------------------------
# Transit quantiles
# ---------------------------------------------------------------------------
class TransitQuantileModel:
    def __init__(self, quantiles: Sequence[float] = QUANTILES, random_state: int = 0):
        self.quantiles = tuple(sorted(quantiles))
        self.models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250,
                                                        learning_rate=0.08, random_state=random_state)
                       for q in self.quantiles}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransitQuantileModel":
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return array (n, len(quantiles)), monotone in quantile (rearranged)."""
        Q = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        return np.maximum.accumulate(np.maximum(Q, 0.5), axis=1)

    def quantile_for(self, Q: np.ndarray, level: float) -> np.ndarray:
        """Q_level by interpolation on the predicted quantile curve (conservative upward)."""
        qs = np.array(self.quantiles)
        if level >= qs[-1]:
            return Q[:, -1]
        idx = int(np.searchsorted(qs, level))
        if qs[idx] == level:
            return Q[:, idx]
        lo, hi = qs[idx - 1], qs[idx]
        w = (level - lo) / (hi - lo)
        return Q[:, idx - 1] * (1 - w) + Q[:, idx] * w

    def prob_exceed(self, Q: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        """P(T > threshold) from the predicted quantile curve (piecewise-linear CDF)."""
        qs = np.array(self.quantiles)
        out = np.empty(len(Q))
        for i in range(len(Q)):
            cdf = np.interp(threshold[i], Q[i], qs, left=0.0, right=qs[-1])
            if threshold[i] > Q[i, -1]:
                # tail beyond the top quantile: geometric decay
                cdf = 1.0 - (1.0 - qs[-1]) * np.exp(-(threshold[i] - Q[i, -1]) / max(1.0, Q[i, -1] - Q[i, -2]))
            out[i] = 1.0 - cdf
        return np.clip(out, 1e-4, 1.0)


class CalibratedRisk:
    def __init__(self, random_state: int = 0):
        self.model = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=random_state),
            method="isotonic", cv=3)
        self.base_rate: float = 0.0
        self.constant: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedRisk":
        self.base_rate = float(np.mean(y))
        if y.sum() < 5 or y.sum() > len(y) - 5:
            self.constant = self.base_rate
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return np.clip(self.model.predict_proba(X)[:, 1], 1e-4, 1.0)


@dataclass
class PredictorSuite:
    """Bundle of the Table III models. ``disabled`` supports ablations."""
    transit: TransitQuantileModel = field(default_factory=TransitQuantileModel)
    excursion: CalibratedRisk = field(default_factory=CalibratedRisk)
    failed: CalibratedRisk = field(default_factory=CalibratedRisk)
    rate_card: RateCard = field(default_factory=RateCard)
    disabled: frozenset = frozenset()       # subset of {"transit", "excursion", "failed"}
    fitted: bool = False

    def fit(self, plans: List[Plan], transit: np.ndarray, excursion: np.ndarray,
            failed: np.ndarray, net: Network) -> "PredictorSuite":
        X = plan_features(plans, net)
        self.transit.fit(X, transit)
        cold_mask = np.array([p.hold_hours < 1e6 for p in plans])
        self.excursion.fit(X[cold_mask], excursion[cold_mask])
        self.failed.fit(X, failed)
        self.fitted = True
        return self

    def annotate(self, plans: List[Plan], net: Network, alpha_cold: float, alpha_other: float,
                 point_estimate: bool = False) -> None:
        """Fill quantiles, Q_{1-alpha}, pi_late, pi_exc, pi_fail on each plan in place."""
        if not plans:
            return
        X = plan_features(plans, net)
        if "transit" in self.disabled:
            # Ablation: replace the learned model by the service's nominal hours.
            base = X[:, 1]
            Q = np.column_stack([base * (0.8 + 0.5 * q) for q in self.transit.quantiles])
        else:
            Q = self.transit.predict(X)
        median = self.transit.quantile_for(Q, 0.5)
        deadlines = np.array([p.deadline_hours for p in plans])
        pi_late = self.transit.prob_exceed(Q, deadlines)
        holds = np.array([p.hold_hours for p in plans])
        temp_controlled = np.isfinite(holds)
        pi_exc = np.zeros(len(plans))
        if temp_controlled.any():
            if "excursion" in self.disabled:
                pi_exc[temp_controlled] = self.excursion.base_rate
            else:
                pi_exc[temp_controlled] = self.excursion.predict(X[temp_controlled])
        pi_fail = (np.full(len(plans), self.failed.base_rate) if "failed" in self.disabled
                   else self.failed.predict(X))
        for i, p in enumerate(plans):
            p.alpha = alpha_cold if p.is_cold else alpha_other
            p.quantiles = {q: float(Q[i, k]) for k, q in enumerate(self.transit.quantiles)}
            if point_estimate:
                p.q_1_minus_alpha = float(median[i])
            else:
                p.q_1_minus_alpha = float(self.transit.quantile_for(Q[i:i + 1], 1.0 - p.alpha)[0])
            p.pi_late = float(pi_late[i])
            p.pi_exc = float(pi_exc[i])
            p.pi_fail = float(pi_fail[i])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)
