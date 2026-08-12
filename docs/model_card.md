# Model card

## Intended use

Portfolio demonstration of calibrated synthetic transaction-risk scoring and governed
three-way decisions. It is not suitable for real card authorization, customer adverse action,
or financial loss estimation.

## Training protocol

- Train: calendar year 2019.
- Tuning comparison: January–March 2020.
- Calibration: April–May 2020.
- Threshold and promotion: June 2020.
- Sealed reporting test: July–December 2020.

Optuna tunes XGBoost and LightGBM against validation PR-AUC. Candidate models refit on data
available through March. Isotonic calibration is fit only on April–May; Platt scaling is the
declared low-support fallback. Business thresholds are fit only on June.

## Promotion policy

A learned challenger must satisfy the June capacity/capture constraints, improve pre-test
validation PR-AUC by at least 2% relative to logistic regression, and improve June simulated
expected cost by at least 2%. Calibration quality is reported separately on the sealed test
and cannot select a model. A failed challenger remains a candidate.
Retraining produces a request and never changes the champion automatically.

## Limitations

Sparkov is synthetic, generator patterns may be easier to learn than adaptive real fraud,
and the source lacks device/network/card-present signals. Fraud-history PSI also reflects
normal state maturation as labels accumulate. Fairness conclusions cannot be drawn because
sensitive fields are excluded and no legitimate adverse-impact study is possible here.

## Measured results

The full sealed test contains 525,661 transactions and 2,012 frauds. Random forest is the
champion because it produced the lowest June pre-test policy cost among accepted challengers.
On the reporting-only test it achieved 0.9097 PR-AUC, 92.79% recall, 60.19% precision, 0.000848
Brier score, 0.000385 expected calibration error, 0.19% review volume, 0.40% block volume,
97.61% fraud-dollar capture, and $39,357.60 simulated cost.

LightGBM recorded the highest sealed-test PR-AUC (0.9313); XGBoost recorded 95.68% recall and
98.03% fraud-dollar capture. Neither changes the pre-test champion decision. Sparkov generator
patterns likely make these results substantially easier than production fraud.

Segment results by supported merchant category, amount bucket, hour bucket, and synthetic
customer state are generated in `reports/segment_performance.csv`. Precision, recall, and
PR-AUC are withheld for segments with fewer than two frauds; the `minimum_support_met` flag
requires at least 1,000 transactions and 10 frauds before operational interpretation.
