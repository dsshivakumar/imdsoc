"""Plan one window of orders and print grounded explanations.

    python scripts/run_demo.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from imdsoc.orchestrator import Orchestrator  # noqa: E402
from imdsoc.synthetic import OrderStream, build_network  # noqa: E402
from imdsoc.training import fit_predictors, log_history  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", type=int, default=6, help="explanations to print")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = build_network(seed=args.seed)
    print(f"Training predictive models on {args.history_days} days of logged shipments ...")
    suite, diag = fit_predictors(log_history(net, days=args.history_days), net)
    print("Held-out diagnostics:", {k: round(v, 3) for k, v in diag.items()})

    net.today = 0
    orders = OrderStream(net, seed=500).day(0)
    orc = Orchestrator(net, suite)
    res = orc.run(orders)
    print(f"\n{len(orders)} orders | {res.n_plans_generated} plans generated | "
          f"{res.n_plans_after_chance} pass chance constraint | {res.seconds:.2f}s")
    for s in res.solves:
        print(f"  MILP: {s.n_orders} orders, {s.n_plans} plans, status={s.status}, "
              f"gap={s.gap:.2%}, {s.solve_seconds * 1000:.0f} ms")
    print("Decision status:", dict(Counter(d.status for d in res.decisions.values())))

    shown = 0
    for d in sorted(res.decisions.values(), key=lambda d: (d.plan is None, d.order_id)):
        if shown >= args.show:
            break
        print(f"\n[{d.order_id}] {d.status.upper()}")
        print("  " + (d.explanation.narrative if d.explanation else "; ".join(d.reasons[:3])))
        shown += 1
    esc = [d for d in res.decisions.values() if d.status == "escalated"]
    if esc:
        print(f"\n[{esc[0].order_id}] ESCALATED\n  " + "\n  ".join(esc[0].reasons[:4]))

    path = os.path.join(args.out, "audit_log.jsonl")
    orc.write_audit_log(path)
    print(f"\nAudit log written to {path}")


if __name__ == "__main__":
    main()
