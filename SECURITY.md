# Security policy and data handling

This repository accepts no PAN, CVV, banking credential, name, street address, or raw birth
date in its public API. Source Sparkov personal fields are synthetic but are still treated as
sensitive and remain in ignored raw storage. Processed IDs are deterministic salted hashes.

Application logs and Prometheus labels must never include transaction/customer/merchant IDs
or amounts. Kafka dead letters retain only schema version, transaction ID when parseable,
error type, and payload byte length. Secrets belong in environment variables or a production
secret manager; `.env` is ignored.

A real deployment would place tokenization and PCI data access outside this service, use TLS
and mutual authentication for Kafka/Redis/PostgreSQL, encrypt data and backups, rotate salts
and credentials, apply least-privilege service identities, enforce retention, audit model and
policy changes, and require independent security and model-risk review.

Report vulnerabilities privately through the repository owner rather than opening an issue
containing an exploit or credential.
