# Data dictionary and privacy boundary

## Dataset

The [Sparkov Kaggle dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) is
CC0 synthetic data generated with the public Sparkov simulator. Planning and runtime checks
verify 1,852,394 rows, 9,651 frauds (0.521%), 999 customers, 693 merchants, 14 categories,
no missing source values, and no duplicate transaction IDs from 2019-01-01 through 2020-12-31.

The generator includes merchant/category, timestamp, amount, home and merchant coordinates,
city population, and a fraud label. It does not include a real transaction channel, payment
method, device, country, authorization response, chargeback workflow, or production context.

## Public event fields

`TransactionEventV1` accepts pseudonymous transaction, customer, and merchant IDs; aware UTC
timestamp; positive amount; USD; one of the 14 known categories or `other`; merchant
coordinates; and optional customer home coordinates, state, and city population.

## Excluded source fields

Synthetic card number, first/last name, gender, occupation, street, city, ZIP, and raw birth
date never enter the processed Parquet or model contract. `cc_num` and merchant string are
salted SHA-256 pseudonyms. A real card platform would use tokenization in a PCI-controlled
zone and expose only surrogate IDs to ML systems.

## Feature availability

All aggregates are computed immediately before the current event is appended. Labels enter
customer and merchant fraud-history aggregates only seven days after their transaction. The
test suite verifies both invariants.
