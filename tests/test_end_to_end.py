import copy

from imdsoc.evaluation import (ExperimentSetup, baseline_strategy, imdsoc_strategy, run_strategy,
                               shadow_mode, summarize)
from imdsoc.orchestrator import Orchestrator
from imdsoc.synthetic import OrderStream


def test_predictor_quantiles_are_calibrated(trained):
    _, _, diag = trained
    assert 0.83 <= diag["coverage_q90"] <= 0.96
    assert diag["coverage_q98"] >= 0.90      # small training set in tests


def test_orchestrator_end_to_end(trained):
    net, suite, _ = trained
    net = copy.deepcopy(net)
    orders = OrderStream(net, seed=123, patients_per_day=30).day(0)
    orc = Orchestrator(net, suite)
    res = orc.run(orders)
    assert set(res.decisions) == {o.order_id for o in orders}
    for d in res.decisions.values():
        if d.status in ("assigned", "review"):
            assert d.plan is not None and d.reservation_ids
            assert d.explanation.headline
            assert d.plan.q_1_minus_alpha <= d.plan.binding_limit       # Eq. (6)
            o = next(o for o in orders if o.order_id == d.order_id)
            site = net.sites[d.plan.site_id]
            for code, rule in orc.eligibility.rules.items():
                if code != "INVENTORY":          # stock was consumed on commit
                    assert rule(o, site, net) is None, (code, d.order_id)
        else:
            assert d.reasons
    assert orc.audit_log


def test_exception_path_resolves_without_failed_site(trained):
    net, suite, _ = trained
    net = copy.deepcopy(net)
    orders = OrderStream(net, seed=5, patients_per_day=15).day(0)
    res = Orchestrator(net, suite).resolve_exception(orders, down_sites={"OKC"}, note="cold-room alarm")
    assert all(d.plan.site_id != "OKC" for d in res.decisions.values() if d.plan)


def test_replay_has_zero_violations(trained):
    net, suite, _ = trained
    setup = ExperimentSetup(net, suite, days=1, patients_per_day=25)
    o1, d1 = run_strategy("imdsoc", imdsoc_strategy(), setup)
    o2, d2 = run_strategy("fixed_site", baseline_strategy("fixed_site"), setup)
    import pandas as pd
    s = summarize(pd.concat([o1, o2]), pd.concat([d1, d2]))
    assert (s.violations == 0).all()
    assert s.loc["imdsoc", "served_rate"] > 0.9


def test_shadow_mode_does_not_mutate_inventory(trained):
    net, suite, _ = trained
    before = sum(l.qty for s in net.sites.values() for l in s.lots)
    df = shadow_mode(net, suite, OrderStream(net, seed=8, patients_per_day=10).day(0))
    after = sum(l.qty for s in net.sites.values() for l in s.lots)
    assert before == after and len(df) > 0
