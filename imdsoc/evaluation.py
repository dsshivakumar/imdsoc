"""Evaluation protocol (Section VIII).

Stage 1  Replay: each strategy is run day by day on the same order stream.
         Deterministic costs (rate cards, packaging, labor) are computed
         exactly; transit, excursion and failure outcomes come from the
         ground-truth simulator, using common random numbers per patient and
         route so paired comparisons are fair. Models were trained on an
         earlier period (see ``training.py``).
Stage 2  Discrete-event simulation of site workload turns workload into
         missed carrier pickups, and stress scenarios vary carrier delay,
         extreme weather, heat, fuel cost and demand.
Stage 3  Shadow mode: ``shadow_mode`` logs IMDSOC recommendations next to
         the incumbent decision without executing them.

Metrics: cost per prescription, on-time rate, excursion rate, consolidation
rate, expired-inventory write-off, site-utilization balance, miles and
emissions, review/override rate, solve time and optimality gap, and
constraint violations (must be zero). Paired per-order differences get
bootstrap confidence intervals; fairness compares rural vs urban (RUCA);
ablations remove consolidation, the chance constraint, and each model.
"""
from __future__ import annotations

import copy
import zlib
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .chance import chance_filter
from .cost import CostModel
from .eligibility import EligibilityEngine, packaging_qualifies
from .models import RESOURCES, Lot, Network, Order, TempClass
from .orchestrator import IMDSOCConfig, Orchestrator
from .plans import Plan
from .predictive import PredictorSuite
from .simulation import simulate_day
from .synthetic import (ORDER_PROFILE, PRODUCT_MIX, STOCKED, OrderStream, Scenario, World,
                        set_weather)


def _h(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


# ---------------------------------------------------------------------------
# Inventory dynamics shared by every strategy
# ---------------------------------------------------------------------------
def replenish(net: Network, day: int, patients_per_day: int) -> None:
    """Deterministic order-up-to replenishment, identical for all strategies."""
    for sid, ndcs in STOCKED.items():
        site = net.sites[sid]
        for ndc in ndcs:
            prod = net.products[ndc]
            per_order, _ = ORDER_PROFILE[ndc]
            n_sites = sum(ndc in v for v in STOCKED.values())
            daily = patients_per_day * 1.54 * PRODUCT_MIX[ndc] * per_order / n_sites
            avail = site.available_qty(ndc, day, prod.min_remaining_dating_days)
            if avail < 2.0 * daily:
                qty = int(np.ceil(6.0 * daily - avail))
                expiry = day + 150 + _h(sid, ndc, day) % 200
                site.lots.append(Lot(f"{sid}-{ndc}-R{day}", ndc, qty, expiry))


def write_off_expiring(net: Network, day: int) -> float:
    """Remove lots that can no longer be dispensed tomorrow; return value written off."""
    value = 0.0
    for site in net.sites.values():
        keep = []
        for lot in site.lots:
            prod = net.products[lot.ndc]
            if lot.qty > 0 and lot.expiry_day - (day + 1) < prod.min_remaining_dating_days:
                value += lot.qty * prod.unit_cost
            elif lot.qty > 0:
                keep.append(lot)
        site.lots = keep
    return value


def consume_fefo(net: Network, plan: Plan) -> None:
    site = net.sites[plan.site_id]
    for ndc, q in plan.qty_by_ndc.items():
        prod = net.products[ndc]
        for lot in site.usable_lots(ndc, net.today, prod.min_remaining_dating_days):
            take = min(q, lot.available(net.today, prod.min_remaining_dating_days))
            lot.qty -= take
            q -= take
            if q == 0:
                break


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------
@dataclass
class DayDecision:
    plans: List[Plan]
    escalated: Dict[str, List[str]]
    review: set = field(default_factory=set)
    fallback: set = field(default_factory=set)      # escalated, then handled by staff
    solve_seconds: float = 0.0
    gap: float = 0.0
    n_plans: int = 0


StrategyFn = Callable[[Network, PredictorSuite, List[Order]], DayDecision]


def imdsoc_strategy(config: Optional[IMDSOCConfig] = None, disabled: frozenset = frozenset(),
                    human_fallback: bool = True) -> StrategyFn:
    def run(net: Network, suite: PredictorSuite, orders: List[Order]) -> DayDecision:
        s = copy.copy(suite)
        s.disabled = disabled
        orc = Orchestrator(net, s, config)
        res = orc.run(orders, execute=True)
        esc = {o: d.reasons for o, d in res.decisions.items() if d.status == "escalated"}
        review = {o for o, d in res.decisions.items() if d.status == "review"}
        plans = [p for p in res.selected_plans if not any(o in esc for o in p.order_ids)]
        fallback: set = set()
        if human_fallback and esc:
            # Escalated orders are resolved by operations staff. We model staff as
            # following the incumbent SOP (fixed site) on remaining capacity, so the
            # comparison does not silently drop the hardest orders.
            cap = {(sid, r): st.capacity[r] for sid, st in net.sites.items() for r in RESOURCES}
            for p in plans:
                for r, l in p.load.items():
                    cap[(p.site_id, r)] -= l
            fb = BASELINES["fixed_site"](net, s).run([o for o in orders if o.order_id in esc], cap)
            for p in fb.selected:
                consume_fefo(net, p)
                fallback |= set(p.order_ids)
            plans += fb.selected
            esc = {o: r for o, r in esc.items() if o not in fallback}
        return DayDecision(plans, esc, review=review, fallback=fallback,
                           solve_seconds=sum(r.solve_seconds for r in res.solves),
                           gap=max([r.gap for r in res.solves] or [0.0]), n_plans=res.n_plans_after_chance)
    run.executes_inventory = True
    return run


def baseline_strategy(name: str) -> StrategyFn:
    cls = BASELINES[name]

    def run(net: Network, suite: PredictorSuite, orders: List[Order]) -> DayDecision:
        r = cls(net, suite).run(orders)
        for p in r.selected:
            consume_fefo(net, p)
        return DayDecision(r.selected, r.escalated)
    run.executes_inventory = True
    return run


def check_violations(plans: Sequence[Plan], orders_by_id: Dict[str, Order], net: Network,
                     engine: EligibilityEngine) -> int:
    """Re-verify every mandatory constraint on the chosen plans (must be zero)."""
    v = 0
    for p in plans:
        site = net.sites[p.site_id]
        for oid in p.order_ids:
            o = orders_by_id[oid]
            for code, rule in engine.rules.items():
                if code == "INVENTORY":
                    continue        # inventory is checked jointly below (before commit)
                if rule(o, site, net):
                    v += 1
        if not packaging_qualifies(net.packaging[p.pkg_id], set(p.temp_classes)):
            v += 1
    return v


# ---------------------------------------------------------------------------
# Replay + DES
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSetup:
    net: Network                          # state at the start of the test period
    suite: PredictorSuite
    days: int = 10
    patients_per_day: int = 70
    order_seed: int = 2024
    outcome_seed: int = 77
    scenario: Scenario = field(default_factory=Scenario)
    use_des: bool = True

    def __post_init__(self):
        stream = OrderStream(self.net, seed=self.order_seed, patients_per_day=self.patients_per_day)
        wrng = np.random.default_rng(self.order_seed + 1)
        self.orders: Dict[int, List[Order]] = {}
        self.weather: Dict[int, Tuple[Dict, Dict]] = {}
        probe = copy.copy(self.net)
        for d in range(self.days):
            set_weather(probe, wrng, self.scenario.heat_shift, 0.0)
            self.weather[d] = (dict(probe.ambient_forecast_c), dict(probe.weather_severity))
            self.orders[d] = stream.day(d, surge=self.scenario.demand_surge)


def run_strategy(name: str, fn: StrategyFn, setup: ExperimentSetup) -> Tuple[pd.DataFrame, pd.DataFrame]:
    net = copy.deepcopy(setup.net)
    sc = setup.scenario
    suite = copy.copy(setup.suite)
    suite.rate_card = replace(setup.suite.rate_card, fuel_surcharge=sc.fuel_surcharge)
    engine = EligibilityEngine()
    rows, day_rows = [], []
    for d in range(setup.days):
        net.today = d
        net.ambient_forecast_c, net.weather_severity = (dict(x) for x in setup.weather[d])
        replenish(net, d, setup.patients_per_day)
        orders = setup.orders[d]
        by_id = {o.order_id: o for o in orders}
        dd = fn(net, suite, orders)
        violations = check_violations(dd.plans, by_id, net, engine)
        des = simulate_day(dd.plans, net, np.random.default_rng(_h(setup.outcome_seed, d, "des")),
                           sc.demand_surge) if setup.use_des else {}
        # Realized outcomes. The true world sees the scenario's extra severity.
        real_sev = {k: min(1.0, v + sc.extra_severity) for k, v in net.weather_severity.items()}
        cm = CostModel(net, rate_card=suite.rate_card)
        for p in dd.plans:
            w = World(net, scenario=replace(sc, extra_severity=0.0))
            w.rng = np.random.default_rng(_h(setup.outcome_seed, d, p.patient_id, p.site_id,
                                             p.carrier_id, p.service))
            p_true = copy.copy(p)
            p_true.weather_severity = real_sev.get(net.patients[p.patient_id].state, p.weather_severity)
            out = w.sample(p_true)
            missed = des[p.site_id].missed_pickup.get(p.plan_id, False) if des else False
            transit = out.transit_hours + (24.0 if missed else 0.0)
            # A shipment that misses pickup waits in qualified on-site storage, so the
            # extra day affects lateness but not in-box excursion.
            c = p.cost if p.cost is not None else cm.evaluate(p)
            n = p.n_orders
            for oid in p.order_ids:
                o = by_id[oid]
                prod = net.products[o.ndc]
                late = transit > o.deadline_hours
                temp_ctrl = prod.temp_class != TempClass.AMBIENT
                exc = bool(out.excursion and temp_ctrl)
                # Cash cost: C_inv is an internal shadow price for inventory age, so it is
                # excluded here; its effect is measured directly as expired write-off.
                realized = (c.ship + c.pack + c.labor + c.xfer + c.carbon) / n
                realized += (c.L_exc / n) * exc + (c.L_late / n) * late + (c.L_fail / n) * out.failed
                pt = net.patients[o.patient_id]
                rows.append(dict(
                    strategy=name, day=d, order_id=oid, ndc=o.ndc, site=p.site_id,
                    carrier=p.carrier_id, service=p.service, pkg=p.pkg_id,
                    rural=pt.rural, ruca=pt.ruca, temp_controlled=temp_ctrl, cold=prod.temp_class.is_cold,
                    consolidated=n > 1, expected_cost=c.total / n, realized_cost=realized,
                    ship_pack_cost=(c.ship + c.pack) / n, deterministic_cost=c.deterministic / n,
                    on_time=not late, excursion=exc, failed=out.failed, missed_pickup=missed,
                    miles=p.distance_miles / n, emissions_kg=c.emissions_kg / n,
                    transfer=o.origin_site != p.site_id, review=oid in dd.review,
                    escalated_fallback=oid in dd.fallback, served=True))
        for oid, reasons in dd.escalated.items():
            o = by_id[oid]
            rows.append(dict(strategy=name, day=d, order_id=oid, ndc=o.ndc, served=False,
                             rural=net.patients[o.patient_id].rural, reason=reasons[0] if reasons else ""))
        util = [net_load / net.sites[s].capacity["pharmacist_verify_min"]
                for s, net_load in _site_loads(dd.plans, net).items()]
        backlog = [r.backlog for r in des.values()] if des else [0]
        written = write_off_expiring(net, d)
        day_rows.append(dict(strategy=name, day=d, write_off=written, violations=violations,
                             utilization_cv=float(np.std(util) / max(1e-9, np.mean(util))),
                             backlog_var=float(np.var(backlog)), backlog_total=int(np.sum(backlog)),
                             solve_seconds=dd.solve_seconds, gap=dd.gap, n_plans=dd.n_plans))
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def _site_loads(plans: Sequence[Plan], net: Network) -> Dict[str, float]:
    out = {s: 0.0 for s in net.sites}
    for p in plans:
        out[p.site_id] += p.load["pharmacist_verify_min"]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def summarize(orders: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in orders.groupby("strategy", sort=False):
        s = g[g.served]
        dd = days[days.strategy == name]
        tc = s[s.temp_controlled]
        out.append(dict(
            strategy=name,
            served_rate=len(s) / len(g),
            cost_per_rx=s.realized_cost.mean(),
            expected_cost_per_rx=s.expected_cost.mean(),
            ship_pack_per_rx=s.ship_pack_cost.mean(),
            on_time_rate=s.on_time.mean(),
            excursion_rate=tc.excursion.mean() if len(tc) else 0.0,
            failed_rate=s.failed.mean(),
            consolidation_rate=s.consolidated.mean(),
            cross_site_rate=s.transfer.mean(),
            write_off_value=dd.write_off.sum(),
            utilization_cv=dd.utilization_cv.mean(),
            backlog_variance=dd.backlog_var.mean(),
            missed_pickup_rate=s.missed_pickup.mean(),
            miles_per_rx=s.miles.mean(),
            kg_co2e_per_rx=s.emissions_kg.mean(),
            review_rate=s.review.mean(),
            escalation_rate=(s.escalated_fallback.mean() * len(s) + (len(g) - len(s))) / len(g),
            solve_seconds=dd.solve_seconds.mean(),
            max_gap=dd.gap.max(),
            violations=int(dd.violations.sum()),
        ))
    return pd.DataFrame(out).set_index("strategy")


def paired_bootstrap(orders: pd.DataFrame, a: str, b: str, col: str, n_boot: int = 2000,
                     seed: int = 0) -> Dict[str, float]:
    """Mean per-order difference a - b on orders served by both, with 95% bootstrap CI."""
    A = orders[(orders.strategy == a) & orders.served].set_index("order_id")[col].astype(float)
    B = orders[(orders.strategy == b) & orders.served].set_index("order_id")[col].astype(float)
    common = A.index.intersection(B.index)
    d = (A.loc[common] - B.loc[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"a": a, "b": b, "metric": col, "n": len(d), "mean_diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi)}


def fairness(orders: pd.DataFrame) -> pd.DataFrame:
    s = orders[orders.served]
    t = s.groupby(["strategy", "rural"]).agg(on_time=("on_time", "mean"), cost=("realized_cost", "mean"),
                                             n=("order_id", "count")).unstack("rural")
    t.columns = [f"{m}_{'rural' if r else 'urban'}" for m, r in t.columns]
    t["on_time_gap_rural_minus_urban"] = t["on_time_rural"] - t["on_time_urban"]
    t["cost_gap_rural_minus_urban"] = t["cost_rural"] - t["cost_urban"]
    return t


def shadow_mode(net: Network, suite: PredictorSuite, orders: List[Order], incumbent: str = "fixed_site"
                ) -> pd.DataFrame:
    """Stage 3: log IMDSOC recommendations alongside incumbent decisions; execute nothing."""
    shadow_net = copy.deepcopy(net)
    rec = Orchestrator(shadow_net, suite).run(orders, execute=False)
    inc = BASELINES[incumbent](copy.deepcopy(net), suite).run(orders)
    inc_by = {p.order_ids[0]: p for p in inc.selected}
    rows = []
    for o in orders:
        d = rec.decisions.get(o.order_id)
        ip = inc_by.get(o.order_id)
        rows.append(dict(order_id=o.order_id, imdsoc_status=d.status if d else "none",
                         imdsoc_site=d.plan.site_id if d and d.plan else None,
                         imdsoc_service=d.plan.service if d and d.plan else None,
                         imdsoc_expected_cost=(d.plan.cost.total / d.plan.n_orders) if d and d.plan else None,
                         incumbent_site=ip.site_id if ip else None,
                         incumbent_service=ip.service if ip else None,
                         incumbent_expected_cost=ip.cost.total if ip else None,
                         agree_site=bool(d and d.plan and ip and d.plan.site_id == ip.site_id),
                         explanation=d.explanation.headline if d and d.explanation else
                         ("; ".join(d.reasons[:1]) if d else "")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full protocol
# ---------------------------------------------------------------------------
ABLATIONS = {
    "imdsoc": (IMDSOCConfig(), frozenset()),
    "abl_no_consolidation": (IMDSOCConfig(consolidate=False), frozenset()),
    "abl_point_estimate": (IMDSOCConfig(chance_constraint=False), frozenset()),
    "abl_no_transit_model": (IMDSOCConfig(), frozenset({"transit"})),
    "abl_no_excursion_model": (IMDSOCConfig(), frozenset({"excursion"})),
    "abl_no_failed_model": (IMDSOCConfig(), frozenset({"failed"})),
}


def run_protocol(setup: ExperimentSetup, strategies: Optional[Sequence[str]] = None,
                 ablations: bool = True, progress: Callable[[str], None] = lambda s: None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs: Dict[str, StrategyFn] = {"imdsoc": imdsoc_strategy(*ABLATIONS["imdsoc"])}
    for b in (strategies or BASELINES):
        runs[b] = baseline_strategy(b)
    if ablations:
        for k, (cfg, dis) in ABLATIONS.items():
            if k != "imdsoc":
                runs[k] = imdsoc_strategy(cfg, dis)
    all_o, all_d = [], []
    for name, fn in runs.items():
        progress(name)
        o, d = run_strategy(name, fn, setup)
        all_o.append(o); all_d.append(d)
    return pd.concat(all_o, ignore_index=True), pd.concat(all_d, ignore_index=True)


def test_hypotheses(orders: pd.DataFrame, summary: pd.DataFrame, non_inferiority: float = 0.01
                    ) -> List[Dict]:
    res = []
    for base in ("fixed_site", "nearest_eligible"):
        if base not in summary.index:
            continue
        c = paired_bootstrap(orders, "imdsoc", base, "realized_cost")
        t = paired_bootstrap(orders, "imdsoc", base, "on_time")
        ok = c["ci_high"] < 0 and t["ci_low"] > -non_inferiority
        res.append(dict(hypothesis="H1", comparison=f"imdsoc vs {base}",
                        evidence=f"cost diff ${c['mean_diff']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}]; "
                                 f"on-time diff {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]",
                        supported=bool(ok)))
    if "abl_point_estimate" in summary.index:
        a, b = summary.loc["imdsoc"], summary.loc["abl_point_estimate"]
        e = paired_bootstrap(orders[orders.temp_controlled.fillna(False).astype(bool)],
                             "imdsoc", "abl_point_estimate", "excursion")
        res.append(dict(hypothesis="H2", comparison="chance-constrained vs point-estimate plans",
                        evidence=f"excursion rate {a.excursion_rate:.4f} vs {b.excursion_rate:.4f}; "
                                 f"paired diff {e['mean_diff']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]",
                        supported=bool(e["ci_high"] < 0)))
    if "abl_no_consolidation" in summary.index:
        a, b = summary.loc["imdsoc"], summary.loc["abl_no_consolidation"]
        e = paired_bootstrap(orders, "imdsoc", "abl_no_consolidation", "ship_pack_cost")
        res.append(dict(hypothesis="H3", comparison="consolidation vs none",
                        evidence=f"shipping+packaging per Rx ${a.ship_pack_per_rx:.2f} vs ${b.ship_pack_per_rx:.2f}; "
                                 f"paired diff {e['mean_diff']:+.2f} [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}]",
                        supported=bool(e["ci_high"] < 0)))
    if "fixed_site" in summary.index:
        a, b = summary.loc["imdsoc"], summary.loc["fixed_site"]
        res.append(dict(hypothesis="H4", comparison="cross-site vs fixed-site",
                        evidence=f"write-off ${a.write_off_value:,.0f} vs ${b.write_off_value:,.0f}; "
                                 f"backlog variance {a.backlog_variance:.2f} vs {b.backlog_variance:.2f}; "
                                 f"utilization CV {a.utilization_cv:.2f} vs {b.utilization_cv:.2f}",
                        supported=bool(a.write_off_value <= b.write_off_value
                                       and a.backlog_variance <= b.backlog_variance)))
    res.append(dict(hypothesis="H5", comparison="explanations vs none (pharmacist override rate)",
                    evidence="requires a user study with pharmacists under institutional review; "
                             "not testable in simulation", supported=None))
    return res


SCENARIOS = {
    "baseline": Scenario(),
    "carrier_delay_+30%": Scenario(carrier_delay=1.3),
    "extreme_weather": Scenario(extra_severity=0.4),
    "heat_wave_+6C": Scenario(heat_shift=6.0),
    "fuel_surcharge_25%": Scenario(fuel_surcharge=0.25),
    "demand_surge_+40%": Scenario(demand_surge=1.4),
}


def stress_test(net: Network, suite: PredictorSuite, days: int = 5,
                strategies: Sequence[str] = ("imdsoc", "fixed_site", "nearest_eligible"),
                scenarios: Optional[Dict[str, Scenario]] = None,
                progress: Callable[[str], None] = lambda s: None) -> pd.DataFrame:
    """Stage 2 sensitivity: rerun strategies under each stress scenario."""
    rows = []
    for sc_name, sc in (scenarios or SCENARIOS).items():
        setup = ExperimentSetup(net, suite, days=days, scenario=sc)
        for st in strategies:
            progress(f"{sc_name}/{st}")
            fn = imdsoc_strategy(*ABLATIONS["imdsoc"]) if st == "imdsoc" else baseline_strategy(st)
            o, d = run_strategy(st, fn, setup)
            summ = summarize(o, d).iloc[0]
            rows.append(dict(scenario=sc_name, strategy=st, cost_per_rx=summ.cost_per_rx,
                             on_time_rate=summ.on_time_rate, excursion_rate=summ.excursion_rate,
                             missed_pickup_rate=summ.missed_pickup_rate,
                             escalation_rate=summ.escalation_rate))
    return pd.DataFrame(rows)
