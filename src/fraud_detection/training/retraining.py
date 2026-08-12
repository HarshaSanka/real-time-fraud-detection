"""Controlled champion/challenger workflow; promotion is never automatic."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraud_detection.utils.config import project_root


def create_retraining_request(
    trigger: str,
    evidence: dict[str, Any],
    candidate_version: str | None = None,
) -> Path:
    payload = {
        "status": "awaiting_human_approval",
        "trigger": trigger,
        "candidate_version": candidate_version,
        "created_at": datetime.now(UTC).isoformat(),
        "evidence": evidence,
        "automatic_promotion": False,
    }
    output = project_root() / "reports/retraining_request.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
