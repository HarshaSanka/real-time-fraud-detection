# Monitoring and delayed labels

Prometheus records request counts by low-cardinality source/decision/status, latency
histograms, and probability histograms. Transaction IDs, customer IDs, merchant IDs, and
amounts are intentionally excluded from metric labels and logs.

The Streamlit dashboard reads persisted predictions for operational volume and transaction
explanations. Grafana provides API request, latency, and score panels. PostgreSQL joins label
events to the original prediction and model version.

## Drift

- Data drift is a change in `P(X)`. Numeric features use population stability index; categorical
  features use Jensen–Shannon divergence.
- Concept drift is a change in `P(Y|X)`. It can be assessed only after labels arrive and meet
  minimum support of 200 outcomes and 20 frauds.
- Drift is a retraining signal, not automatic proof of model failure and not an auto-promotion
  command.

The current history-based rates are cumulative and naturally mature over time. Their PSI is
reported separately from exogenous transaction features so normal label accumulation is not
misrepresented as a new fraud regime.

## Failure runbook

- Model invalid/missing: readiness and scoring fail 503; liveness remains 200.
- Redis unavailable: score with missing-history flag and enforce at least REVIEW.
- Kafka unavailable: database transaction succeeds and unpublished outbox rows retry later.
- Invalid/corrupt Kafka event: sanitized metadata enters the DLQ and input offset commits.
- PostgreSQL unavailable: do not return an un-auditable authorization result.
