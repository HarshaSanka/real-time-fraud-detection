"""Local in-process HTTP benchmark; reports environment and dependencies honestly."""

from __future__ import annotations

import json
import platform
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx
import numpy as np

from fraud_detection.api.main import app
from fraud_detection.utils.config import project_root


def payload(index: int) -> dict[str, object]:
    return {
        "transaction_id": f"bench_{index}_{uuid.uuid4().hex[:8]}",
        "customer_id": f"cust_bench_{index % 50}",
        "merchant_id": f"merchant_bench_{index % 20}",
        "timestamp": datetime.now(UTC).isoformat(),
        "amount": float(20 + index % 200),
        "currency": "USD",
        "merchant_category": "shopping_net",
        "merchant_latitude": 41.8781,
        "merchant_longitude": -87.6298,
        "customer_home_latitude": 41.881,
        "customer_home_longitude": -87.62,
        "customer_state": "IL",
        "city_population": 2_746_388,
    }


def main() -> None:
    transport = httpx.ASGITransport(app=app)
    results: list[dict[str, object]] = []
    for concurrency in [1, 4, 16]:
        latencies: list[float] = []
        errors = 0
        count = 200

        def request(index: int, latency_values: list[float] = latencies) -> None:
            nonlocal errors

            async def send() -> None:
                nonlocal errors
                started = time.perf_counter()
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post("/score", json=payload(index))
                latency_values.append((time.perf_counter() - started) * 1000)
                errors += int(response.status_code != 200)

            import asyncio

            asyncio.run(send())

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(request, range(count)))
        elapsed = time.perf_counter() - started
        results.append(
            {
                "concurrency": concurrency,
                "requests": count,
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "throughput_requests_per_second": count / elapsed,
                "errors": errors,
            }
        )
    report = {
        "benchmark_type": "in-process FastAPI ASGI",
        "dependencies": {"redis": False, "postgresql": False, "kafka": False},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "results": results,
        "limitations": "This is not a distributed or production load test.",
    }
    output = project_root() / "reports/inference_benchmark.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
