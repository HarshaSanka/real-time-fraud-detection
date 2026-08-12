# Fraud feature and decision strategy

The feature set combines authorization-time context with strictly historical behavior:

- Cyclical hour/weekday, amount, log amount, home-to-merchant distance, and city population.
- Customer 7/30-day average amounts and amount-to-baseline ratio.
- Transaction counts over 5 minutes, 30 minutes, 1 hour, and 24 hours.
- One-hour amount and unique-merchant velocity.
- Time and distance from the previous merchant, travel speed, and impossible travel.
- Merchant 24-hour volume and 30-day average amount.
- Smoothed confirmed fraud rates with seven-day label availability.
- Customer/merchant cold-start and online-history-unavailable flags.

SMOTE is deliberately excluded. Interpolating encoded categories, geographic positions,
temporal velocity, and entity history can synthesize transactions that no customer made and
states that could not coexist. The repository measures class weighting, deterministic
undersampling, and random oversampling using training data only.

Rules provide an interpretable baseline. Logistic regression tests a linear scorecard-like
approach. Random forest, XGBoost, LightGBM, and CatBoost compare nonlinear tabular learners.
Anomaly detection is omitted because labels are available and adding an unsupervised model
without a demonstrated gain would increase complexity without strengthening the system.
