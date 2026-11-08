"""Run the Section VIII evaluation protocol end to end and write a report.

    python scripts/run_evaluation.py --days 10 --stress-days 5

Outputs (in --out):
  predictor_diagnostics.json   held-out coverage, pinball loss, Brier, ECE
  orders.csv / days.csv        per-order and per-day records for every strategy
  summary.csv                  Section VIII metrics per strategy / ablation
  paired_bootstrap.csv         paired per-order differences with 95% CIs
  fairness.csv                 rural vs urban (RUCA) on-time and cost
  hypotheses.json              H1-H5 verdicts
  stress.csv                   discrete-event simulation stress scenarios
  shadow_mode.csv              stage-3 shadow log (IMDSOC vs incumbent)
  REPORT.md                    human-readable summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from imdsoc.evaluation import (ExperimentSetup, fairness, paired_bootstrap, run_protocol,  # noqa: E402
                               shadow_mode, stress_test, summarize, test_hypotheses)
from imdsoc.synthetic import OrderStream, build_network  # noqa: E402
from imdsoc.training import fit_predictors, log_history  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--stress-days", type=int, default=5)
    ap.add_argument("--history-days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    net = build_network(seed=args.seed)
    log(f"Stage 1: training models on {args.history_days} days of history (temporal split)")
    suite, diag = fit_predictors(log_history(net, days=args.history_days), net)
    json.dump(diag, open(os.path.join(args.out, "predictor_diagnostics.json"), "w"), indent=2)
    net.today = 0

    log(f"Stage 1+2: replay with discrete-event workload simulation, {args.days} test days")
    setup = ExperimentSetup(net, suite, days=args.days)
    orders, days = run_protocol(setup, progress=lambda n: log(f"  running {n}"))
    orders.to_csv(os.path.join(args.out, "orders.csv"), index=False)
    days.to_csv(os.path.join(args.out, "days.csv"), index=False)
    summary = summarize(orders, days)
    summary.to_csv(os.path.join(args.out, "summary.csv"))

    pairs = [paired_bootstrap(orders, "imdsoc", b, m)
             for b in summary.index if b != "imdsoc"
             for m in ("realized_cost", "on_time")]
    pd.DataFrame(pairs).to_csv(os.path.join(args.out, "paired_bootstrap.csv"), index=False)
    fair = fairness(orders)
    fair.to_csv(os.path.join(args.out, "fairness.csv"))
    hyps = test_hypotheses(orders, summary)
    json.dump(hyps, open(os.path.join(args.out, "hypotheses.json"), "w"), indent=2)

    stress = None
    if not args.no_stress:
        log(f"Stage 2: stress scenarios, {args.stress_days} days each")
        stress = stress_test(net, suite, days=args.stress_days, progress=lambda n: log(f"  {n}"))
        stress.to_csv(os.path.join(args.out, "stress.csv"), index=False)

    log("Stage 3: shadow mode (recommendations logged, nothing executed)")
    shadow = shadow_mode(net, suite, OrderStream(net, seed=9001).day(0))
    shadow.to_csv(os.path.join(args.out, "shadow_mode.csv"), index=False)

    # ---- report -----------------------------------------------------------
    key = ["served_rate", "escalation_rate", "cost_per_rx", "on_time_rate", "excursion_rate",
           "failed_rate", "consolidation_rate", "cross_site_rate", "write_off_value",
           "utilization_cv", "backlog_variance", "missed_pickup_rate", "miles_per_rx",
           "kg_co2e_per_rx", "review_rate", "solve_seconds", "max_gap", "violations"]
    rep = [f"# IMDSOC evaluation report\n",
           f"Synthetic network, {args.days} test days, {orders.order_id.nunique()} unique orders, "
           f"seed {args.seed}. All results are from simulation, not deployment.\n",
           "## Predictor diagnostics (held-out later period)\n",
           "```json\n" + json.dumps({k: round(v, 4) for k, v in diag.items()}, indent=2) + "\n```\n",
           "## Metrics by strategy\n", md_table(summary[key].T), "\n",
           "## Hypotheses\n"]
    for h in hyps:
        verdict = {True: "supported", False: "not supported", None: "not testable here"}[h["supported"]]
        rep.append(f"- **{h['hypothesis']}** ({h['comparison']}): {verdict}. {h['evidence']}")
    rep += ["\n## Paired per-order differences (IMDSOC minus strategy, 95% bootstrap CI)\n",
            md_table(pd.DataFrame(pairs).set_index("b")[["metric", "n", "mean_diff", "ci_low", "ci_high"]]),
            "\n## Rural vs urban (RUCA >= 4 is rural)\n",
            md_table(fair[["on_time_urban", "on_time_rural", "on_time_gap_rural_minus_urban",
                           "cost_urban", "cost_rural", "cost_gap_rural_minus_urban"]])]
    if stress is not None:
        rep += ["\n## Stress scenarios (discrete-event simulation)\n",
                md_table(stress.set_index("scenario"))]
    agree = shadow.agree_site.mean()
    rep += [f"\n## Shadow mode\n\n{len(shadow)} orders logged; IMDSOC agreed with the incumbent "
            f"site on {agree:.1%}. See shadow_mode.csv.\n"]
    open(os.path.join(args.out, "REPORT.md"), "w").write("\n".join(rep))
    log(f"Done. Report: {os.path.join(args.out, 'REPORT.md')}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[key].T.round(3))
    for h in hyps:
        print(h["hypothesis"], h["supported"], "-", h["evidence"])


if __name__ == "__main__":
    main()
