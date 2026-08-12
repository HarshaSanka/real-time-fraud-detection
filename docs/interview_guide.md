# Interview guide

1. **Why is accuracy weak?** With 0.521% fraud, an all-legitimate predictor exceeds 99%
   accuracy while catching nothing. I evaluate ranking, calibration, missed fraud value, and
   review/block capacity.
2. **Why PR-AUC?** It concentrates on positive-class retrieval under imbalance and shows the
   precision cost of increasing fraud recall.
3. **Class imbalance?** I measured unweighted, class-weighted, undersampled, and randomly
   oversampled training. Validation and test distributions remain untouched.
4. **Why not SMOTE?** Interpolation can invent invalid combinations of category, location,
   entity history, and velocity. That is harder to defend than weighted loss or sampling real
   rows.
5. **Leakage prevention?** Events are sorted globally; the engine emits features before state
   update; label-derived rates update seven days later. Tests assert both.
6. **Why chronological splits?** Deployment predicts future behavior from past behavior.
   Random splits can leak customer/merchant regimes and understate drift.
7. **Threshold selection?** I use a separate June window to choose APPROVE/REVIEW/BLOCK
   thresholds under explicit simulated loss and queue constraints—not 0.5.
8. **False-positive cost?** It includes manual review, decline friction, lost conversion, and
   trust. The demo uses transparent $5 review and $25 false-block assumptions.
9. **False-negative cost?** The demo uses transaction amount as the missed-fraud loss proxy;
   real loss would include recovery, interchange, operations, and customer impact.
10. **Why boosted trees?** Mixed nonlinear amount, velocity, geography, category, and entity
    history interactions are a strong tabular fit; logistic remains the interpretable baseline.
11. **Real-time features?** Redis holds ordered customer/merchant state and transaction
    snapshots. The event is added after feature extraction.
12. **Why Kafka?** It decouples sources, scoring, alerts, labels, and replay while providing
    durable partitions and consumer recovery.
13. **Why Redis?** Low-latency recent state and retry-safe snapshots. The demo's global lock is
    explicitly not the scaled design.
14. **Delayed labels?** Label events join to prediction IDs in PostgreSQL, update performance
    for the model version, then update confirmed-history state.
15. **Drift?** PSI/JS detect `P(X)` movement; delayed labeled metrics detect `P(Y|X)` changes.
16. **Retraining?** Drift or performance gates open a request. A challenger is evaluated and
    waits for human approval.
17. **Champion/challenger?** The champion stays production-addressable while candidates are
    compared on PR-AUC, calibration, cost, and review/block volume.
18. **Millions of transactions?** Partition Kafka and state by customer, run stateless model
    replicas, use a compiled model format, shard storage, and load test. This repo does not
    claim that scale.
19. **Lower latency?** Preload bundles, avoid per-request allocation, batch where permitted,
    use Redis pipelines/Lua, profile encoders, and consider Treelite/ONNX after parity checks.
20. **Real bank changes?** Add tokenized card/device/network/authorization features, real label
    semantics and economics, model-risk validation, fairness analysis, shadow/canary rollout,
    PCI controls, disaster recovery, and investigator feedback loops.

## Resume bullets

- Built a real-time fraud detection platform with Python, Kafka, Redis, FastAPI, PostgreSQL,
  MLflow, Prometheus, and Grafana, processing 1.85M synthetic Sparkov transactions through a
  leakage-safe behavioral and velocity feature pipeline.
- Trained and calibrated logistic regression, random forest, XGBoost, LightGBM, and CatBoost
  models using chronological validation and Optuna, achieving 0.931 PR-AUC with LightGBM on a
  525,661-transaction sealed test.
- Optimized APPROVE/REVIEW/BLOCK thresholds against simulated loss and capacity constraints;
  the governed random-forest champion captured 97.61% of fraud dollars at 0.19% review and
  0.40% block volume on the reporting-only test.
- Implemented seven-day delayed labels, idempotent prediction logging, transactional Kafka
  outbox delivery, drift monitoring, and human-gated champion/challenger promotion.
