# IMDSOC — Regulation-Aware, Cost-Optimized Pharmacy Fulfillment

Reference implementation of the paper **"A Reference Architecture for Regulation-Aware,
Cost-Optimized Pharmacy Fulfillment"** (Devalokam Subrayappa, Bundela, Potla, Basu,
Annamalai Sekar, Prasad).

IMDSOC (Intelligent Medication Dispensing and Shipping Optimization Center) decides, for every
prescription in a multi-site pharmacy network, **which site dispenses it and how it ships**:
site, carrier, service level, packaging, and consolidation of a patient's orders. The central
design rule is that **mandatory requirements and monetary trade-offs are handled in separate
stages**:

1. **Deterministic eligibility** removes sites that are ineligible on legal, clinical or
   operational grounds (licensure, payer network, REMS, limited distribution, controlled-substance
   transfer, inventory validity, packaging, pharmacist availability) and records why.
2. **Chance constraints** keep only shipment plans whose predicted transit-time quantile fits
   inside both the delivery deadline and the packaging hold time.
3. A **set-partitioning MILP** with an objective expressed entirely in dollars chooses the plans.
4. Every recommendation is **explained** by its eliminating rules and cost components, and is
   **gated** (assign / human review / escalate).

Learning components only supply probabilities and quantiles; they never decide feasibility.

```
orders ─► eligibility rules ─► plan generation ─► predict (quantiles, π_exc, π_late, π_fail)
        (reasons recorded)    (single + consolidated)            │
                                                                  ▼
 audit log ◄─ reserve & dispatch ◄─ gate ◄─ explain ◄─ MILP ◄─ chance filter (Eq. 6)
                                     │                 (Eqs. 2–5, per region)
                                     └─► human review / escalation
```

## Paper-to-code map

| Paper | What it is | Code |
|---|---|---|
| §III, Table II | Products, sites, carriers, packaging, orders | `imdsoc/models.py` |
| §IV "Eligibility then ranking" | Deterministic rules with recorded reasons | `imdsoc/eligibility.py` |
| §V.A | Shipment plans `s = (O_s, j_s, k_s, m_s, p_s)`, consolidation | `imdsoc/plans.py` |
| §V.B, Eq. (1) | Unit-consistent dollar objective | `imdsoc/cost.py` |
| §V.C, Eq. (6) | Chance constraint `Q̂₁₋α(T_s) ≤ min{D_s, H_ps}` | `imdsoc/chance.py` |
| §V.C, Eqs. (2)–(5) | Set-partitioning MILP (HiGHS via SciPy) | `imdsoc/optimizer.py` |
| §V.D, Algorithm 1 | Per-region solve, escalation, re-solve on exceptions | `imdsoc/orchestrator.py` |
| §IV, Fig. 3 | Atomic, TTL-limited inventory reservations | `imdsoc/reservation.py` |
| §VI, Table III | Quantile transit model, calibrated excursion / failed-delivery models, rate card | `imdsoc/predictive.py`, `imdsoc/training.py` |
| §VI | Grounded explanations; pluggable (LLM/RAG) narrator that stays advisory | `imdsoc/explain.py` |
| Table I | Baseline strategies | `imdsoc/baselines.py` |
| §VIII | Replay, discrete-event simulation, shadow mode, metrics, bootstrap CIs, RUCA fairness, ablations, H1–H5 | `imdsoc/evaluation.py`, `imdsoc/simulation.py` |
| — | Synthetic network and hidden ground truth (no deployment data exists) | `imdsoc/synthetic.py` |

## Quick start

Requires Python 3.10+.

```bash
git clone <this repo> && cd imdsoc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_demo.py            # one planning window with explanations + audit log
python scripts/run_evaluation.py      # full Section VIII protocol -> results/REPORT.md
pytest -q                             # 18 tests
```

`run_evaluation.py` options: `--days` (test days, default 10), `--stress-days`, `--history-days`,
`--seed`, `--no-stress`, `--out`.

### Example explanation (from `run_demo.py`)

> Site DEN: $13.94 lower expected cost per order than site OKC, of which $22.23 is lower shipping.
> Ship from DEN via CAR-A ground in COLD-72; expected cost $530.60. Transit 98th percentile 47.4 h
> <= limit 70.8 h (deadline / packaging hold). Ineligible sites -> AUS: LICENSURE; DAL: LICENSURE;
> LIT: LICENSURE; SAT: LICENSURE. 34 plan(s) removed by the transit/hold-time chance constraint.
> Escalated: changes patient's usual dispensing pharmacy (notification/consent).

## Using it as a library

```python
from imdsoc import Orchestrator, IMDSOCConfig, PolicyParameters
from imdsoc.synthetic import build_network, OrderStream
from imdsoc.training import log_history, fit_predictors

net = build_network()                                   # or build a Network from your own data
suite, diagnostics = fit_predictors(log_history(net, days=20), net)

orc = Orchestrator(net, suite,
                   config=IMDSOCConfig(alpha_cold=0.02, alpha_other=0.10),
                   policy=PolicyParameters(carbon_price_per_kg=0.19))
result = orc.run(OrderStream(net).day(0))
for d in result.decisions.values():
    print(d.order_id, d.status, d.explanation.headline if d.explanation else d.reasons)
orc.write_audit_log("audit_log.jsonl")
```

Rules are a plain dict of functions, so compliance staff can add or replace them:

```python
from imdsoc.eligibility import DEFAULT_RULES
rules = dict(DEFAULT_RULES, STATE_SPECIFIC=lambda order, site, net: None)  # return a reason to eliminate
Orchestrator(net, suite, rules=rules)
```

## Results on the synthetic network

`python scripts/run_evaluation.py --days 14 --stress-days 5` (seed 7, 1,490 orders; full output in
[`results/REPORT.md`](results/REPORT.md)). **These are simulation results on synthetic data, not
evidence about any real pharmacy network.** The magnitudes depend heavily on the synthetic
assumptions, in particular the product values and the excursion model.

| Metric | IMDSOC | Fixed site | Nearest eligible | Lowest ship cost | Rules only |
|---|---|---|---|---|---|
| Cost per Rx (cash + realized consequences) | $55 | $225 | $232 | $460 | $379 |
| On-time rate | 98.4% | 98.8% | 98.3% | 94.0% | 92.7% |
| Excursion rate (temperature-controlled) | 1.9% | 8.0% | 9.5% | 54.0% | 38.3% |
| Consolidation rate | 43% | 0% | 0% | 0% | 0% |
| Expired write-off (14 days) | $1,090 | $2,597 | $5,110 | $5,110 | $97 |
| Escalation rate (no plan / chance-infeasible) | 5.2% | 1.3% | 1.3% | 1.3% | 1.3% |
| Mandatory-constraint violations | 0 | 0 | 0 | 0 | 0 |
| MILP solve time per day | 0.05 s | – | – | – | – |

Hypotheses (95% paired bootstrap CIs, 1 pp non-inferiority margin for on-time):

- **H1** cost lower at no loss in on-time: *supported* vs nearest-eligible; *not supported* vs
  fixed-site — cost is lower, but the on-time CI (−1.1 pp to +0.3 pp) crosses the margin.
- **H2** chance-constrained vs point-estimate plans lower excursion: *not supported* (1.89% vs
  2.03%, CI includes 0). Because Eq. (1) already prices excursion risk, the point-estimate variant
  avoids most risky plans anyway.
- **H3** consolidation lowers shipping and packaging cost: *supported* (−$2.77 per Rx).
- **H4** cross-site fulfillment lowers write-off and backlog variance vs fixed site: *supported*.
- **H5** explanations reduce pharmacist override rate: needs a user study; not testable here.

Fairness finding worth noting for the paper: IMDSOC's rural on-time rate is 2.0 pp below urban,
versus 0.6 pp for fixed-site. Rural cost is not higher. Under the paper's rule that fairness is a
release criterion, this configuration would need tuning (e.g., a higher late-delivery consequence
cost for rural destinations) before release.

## Design notes and simplifications

- **Controlled-substance transfer rules** implement a simplified, configurable reading of
  21 CFR Part 1306 (C-II: one-time transfer of an electronic prescription; C-III–V: one transfer
  unless the pharmacies share a real-time database). State law varies. Rules must be maintained by
  compliance staff; nothing here is legal advice.
- **Infeasible or unservable orders** are absorbed by a heavily penalized "unassigned" variable,
  so the MILP stays feasible and those orders are escalated with reasons instead of dropped. In
  the evaluation, escalated orders are handled by staff following the fixed-site SOP so they are
  not excluded from the comparison.
- **Regions** are solved sequentially, each consuming the inventory and capacity it uses.
  `IMDSOCConfig.capacity_headroom` (default 0.85) leaves a buffer below `Cap_jr`.
- **C_inv** charges an expected write-off when a plan draws later-dated stock while short-dated
  stock elsewhere is close to losing dating. It is an internal shadow price, so the evaluation
  reports cash cost without it and measures write-off directly.
- **Emission factors** (`cost.EMISSION_FACTORS`) are placeholders; use ISO 14083 / GLEC factors.
- **Excursion model**: the paper lists a survival model [Cox]; this implementation uses a
  calibrated boosted classifier, the alternative named in Table III.
- **SHAP and the knowledge graph** are not included; the explanation layer is built on rule
  reasons and Eq. (1) components. `Narrator` is the hook for a RAG service (e.g., Amazon Bedrock).
- **Training logs** in the synthetic world use randomized plan choices. Real logs are confounded
  by the incumbent policy, which is why the protocol adds simulation and shadow mode.
- **Demand / workload forecasting** (temporal fusion transformer) is out of scope; capacities are
  given.

## Mapping to the AWS reference implementation (§VII)

| Module | AWS service in the paper |
|---|---|
| `orchestrator.Orchestrator` | AWS Step Functions (workflow), API Gateway (intake) |
| `eligibility`, `plans`, `explain` | Containers on Amazon EKS |
| `optimizer` | Solver container on AWS Fargate |
| `predictive` | Amazon SageMaker AI endpoints |
| `reservation.InventoryLedger` | Aurora PostgreSQL conditional writes with TTL |
| Rules data (licenses, payers, REMS) | Amazon Neptune graph |
| `explain.Narrator` (LLM option) | Amazon Bedrock with knowledge bases and guardrails |
| `orchestrator.write_audit_log` | Immutable audit store (CloudTrail / S3 Object Lock) |

The code is cloud-neutral; the table only shows where each piece would run.

## Repository layout

```
imdsoc/        library (see map above)
scripts/       run_demo.py, run_evaluation.py
tests/         pytest suite
results/       report and CSVs from the run above (orders.csv is regenerated, not committed)
```

## Citation

```
S. K. Devalokam Subrayappa, P. Bundela, S. S. C. Potla, S. Basu, M. Annamalai Sekar, A. Prasad,
"A Reference Architecture for Regulation-Aware, Cost-Optimized Pharmacy Fulfillment."
```

## License

MIT — see [LICENSE](LICENSE).
