# Business problem and simulated risk policy

Fraud authorization is a risk-allocation problem. A probability estimate is useful only when
it is combined with fraud loss, customer friction, review capacity, and delayed outcomes.

This project maps calibrated probability to three actions:

- `APPROVE`: accept the transaction and incur its amount if it later proves fraudulent.
- `REVIEW`: pay a simulated $5 review cost and capture 80% of fraudulent value.
- `BLOCK`: stop fraud value but assign a simulated $25 friction cost to a legitimate block.

The June 2020 policy window selects two thresholds subject to review volume at or below 5%,
block volume at or below 1%, and a target of at least 80% fraud-dollar capture. If no threshold
pair satisfies all three, the report says so and chooses the lowest-cost pair satisfying the
capacity limits. These values are portfolio assumptions, not bank economics.

Accuracy is unsuitable because more than 99% of Sparkov transactions are legitimate. The
repository prioritizes PR-AUC, calibrated probabilities, recall, fraud dollars captured,
expected cost, and operational volume.
