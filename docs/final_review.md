# Final structured self-review

These scores are a strict self-assessment of committed and measured evidence, not a third-party
endorsement. Synthetic-data results do not establish bank-production effectiveness.

## Quality gate 1 — fraud ML reviewer

**Pass with limitations.** The implementation uses globally ordered, emit-before-update
features; chronological train/tune/calibration/threshold/test windows; seven-day label
availability; multiple meaningful baselines; four imbalance strategies; PR-AUC-first model
comparison; post-fit isotonic calibration; and separate June business-policy optimization.
The sealed test is reporting-only and a regression test enforces this. The principal limitation
is Sparkov's synthetic generator signal; no fairness or real investigator-queue study is
possible.

## Quality gate 2 — ML platform reviewer

**Pass locally; container status follows public CI.** Offline/online replay tests assert feature
parity and retries are idempotent. A missing Redis state path marks history unavailable and
enforces at least review. PostgreSQL combines prediction persistence and Kafka outbox creation
in one transaction. Corrupt events are sanitized to a DLQ. Model metadata pins feature/data
hashes, thresholds, calibration, cutoff, and version. Delayed labels and human-gated
champion/challenger requests are implemented. The reference Redis engine uses a global lock;
real scale requires partitioned state processing.

## Quality gate 3 — hiring manager

**Pass.** The first page presents the business objective, measured test evidence, architecture,
limits, plots, and reproducible startup. Code, tests, generated artifacts, model card, monitoring
guide, security notes, and interview answers agree on the same implementation. No raw data,
credentials, model binary, bank impact, or unmeasured scale claim is published.

## Scores

| Category | Score | Evidence and deduction |
|---|---:|---|
| Machine Learning | 9/10 | Six learned/baseline approaches, Optuna, calibration, chronological evaluation; synthetic-only evidence prevents 10. |
| Feature Engineering | 9/10 | Shared historical velocity, behavior, merchant, geography, label-delay state with parity/leakage tests; lacks device/network features. |
| Fraud/Risk Understanding | 9/10 | Three-way constrained policy, simulated loss, delayed labels, queue volume, explicit false-positive economics. |
| Software Engineering | 9/10 | Typed contracts, strict MyPy, validation, idempotency, transactional persistence, unit/integration tests and coverage gate. |
| MLOps | 8/10 | MLflow lineage/registry, drift, retraining request, dashboards, CI; no real cloud deployment or long-running scheduler. |
| Real-Time Systems | 8/10 | Kafka KRaft, Redis, PostgreSQL outbox, DLQ and retries; single-node reference and no distributed load result. |
| Documentation | 9/10 | Architecture, data contract, model card, monitoring, security, interview guide, generated reports. |
| GitHub Portfolio Quality | 9/10 | Recruiter-first README, plots, CI, Compose, measured claims; screenshots/container evidence depend on public CI. |
| Interview Defensibility | 9/10 | Decisions, constraints, limitations, leakage boundaries, and exact metrics are committed and internally consistent. |

## Remaining honest gaps

- Sparkov has no authentic device, IP, channel, authentication, or country signals.
- The local latency run is in-process with dependencies inactive; it is not a production load test.
- Redis state serialization and a global lock prioritize exact demo parity over horizontal scale.
- No real fraud-operations team validated the simulated costs or promotion constraints.
- Cloud infrastructure, disaster recovery, PCI scope, adversarial testing, fairness analysis, and
  a shadow/canary deployment are deliberately outside the measured repository scope.
