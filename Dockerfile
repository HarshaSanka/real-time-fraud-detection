# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip wheel --wheel-dir /wheels '.[all]'

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FRAUD_PROJECT_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 fraud
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY configs ./configs
COPY dashboards ./dashboards
COPY monitoring ./monitoring
COPY scripts ./scripts
RUN mkdir -p data/raw data/processed artifacts predictions reports && chown -R fraud:fraud /app
USER fraud
EXPOSE 8000 8501
CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
