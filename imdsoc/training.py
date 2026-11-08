"""Training the Table III models from logged shipments, with a temporal split.

Shipments logged on earlier days train the models; later days test them
(Section VIII, stage 1: "models trained on an earlier period and tested on a
later one"). The synthetic log uses randomized plan choices so every
site/carrier/service combination is observed; real logs are confounded by the
incumbent policy, which is one reason the protocol adds simulation and
shadow mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .eligibility import EligibilityEngine
from .models import Network
from .plans import Plan, PlanGenerator
from .predictive import (PredictorSuite, expected_calibration_error, plan_features)
from .synthetic import OrderStream, World, set_weather


@dataclass
class ShipmentLog:
    plans: List[Plan]
    transit: np.ndarray
    excursion: np.ndarray
    failed: np.ndarray
    day: np.ndarray


def log_history(net: Network, days: int, seed: int = 3, plans_per_day: int = 700,
                patients_per_day: int = 70) -> ShipmentLog:
    rng = np.random.default_rng(seed)
    stream = OrderStream(net, seed=seed + 1, patients_per_day=patients_per_day)
    world = World(net, seed=seed + 2)
    engine = EligibilityEngine()
    gen = PlanGenerator(max_consolidation=3)
    plans, T, E, F, D = [], [], [], [], []
    for d in range(-days, 0):
        set_weather(net, rng)
        orders = stream.day(d)
        elig = engine.evaluate_all(orders, net)
        day_plans = gen.generate(orders, elig, net)
        if not day_plans:
            continue
        pick = rng.choice(len(day_plans), size=min(plans_per_day, len(day_plans)), replace=False)
        for k in pick:
            p = day_plans[k]
            out = world.sample(p)
            plans.append(p); T.append(out.transit_hours); E.append(out.excursion)
            F.append(out.failed); D.append(d)
    return ShipmentLog(plans, np.array(T), np.array(E, dtype=int), np.array(F, dtype=int), np.array(D))


def fit_predictors(log: ShipmentLog, net: Network, test_fraction: float = 0.25
                   ) -> Tuple[PredictorSuite, Dict[str, float]]:
    """Temporal split: fit on earliest days, report held-out diagnostics on the latest."""
    cut = np.quantile(log.day, 1.0 - test_fraction)
    tr, te = log.day < cut, log.day >= cut
    sel = lambda m: [p for p, k in zip(log.plans, m) if k]
    suite = PredictorSuite().fit(sel(tr), log.transit[tr], log.excursion[tr], log.failed[tr], net)
    diag = evaluate_predictors(suite, sel(te), log.transit[te], log.excursion[te], log.failed[te], net)
    # Refit on all history for deployment.
    suite = PredictorSuite().fit(log.plans, log.transit, log.excursion, log.failed, net)
    return suite, diag


def evaluate_predictors(suite: PredictorSuite, plans: List[Plan], transit: np.ndarray,
                        exc: np.ndarray, fail: np.ndarray, net: Network) -> Dict[str, float]:
    X = plan_features(plans, net)
    Q = suite.transit.predict(X)
    out: Dict[str, float] = {"n_test": float(len(plans))}
    for k, q in enumerate(suite.transit.quantiles):
        if q in (0.5, 0.9, 0.98):
            out[f"coverage_q{int(q * 100)}"] = float(np.mean(transit <= Q[:, k]))
            r = transit - Q[:, k]
            out[f"pinball_q{int(q * 100)}"] = float(np.mean(np.maximum(q * r, (q - 1) * r)))
    holds = np.array([p.hold_hours for p in plans])
    m = np.isfinite(holds)
    if m.any():
        pe = suite.excursion.predict(X[m])
        out["excursion_brier"] = float(np.mean((pe - exc[m]) ** 2))
        out["excursion_ece"] = expected_calibration_error(exc[m], pe)
        out["excursion_rate"] = float(exc[m].mean())
    pf = suite.failed.predict(X)
    out["failed_brier"] = float(np.mean((pf - fail) ** 2))
    out["failed_ece"] = expected_calibration_error(fail, pf)
    return out
