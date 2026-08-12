# Public CI verification

[GitHub Actions run #3](https://github.com/HarshaSanka/real-time-fraud-detection/actions/runs/31650537957)
completed successfully for commit `408247f` on 2026-08-12.

## Quality job

- Python 3.12 installation, Ruff formatting/linting, and strict MyPy passed.
- Unit/integration tests passed the configured 80% coverage gate.
- The executed notebook and sanitized feature/model smoke workflow passed.
- The dependency vulnerability audit passed.

## Container and streaming job

- The runtime image built and the Compose configuration validated.
- PostgreSQL, Redis, Kafka KRaft, topic initialization, API, scoring consumer, outbox publisher,
  and delayed-label consumer started successfully.
- API readiness succeeded with the CI model bundle loaded.
- The replay producer published 20 events; assertions verified at least 20 persisted predictions,
  published outbox rows, and Redis feature state.

This is a functional reference-flow verification on a GitHub-hosted runner, not a throughput,
resilience, or production-scale benchmark.
