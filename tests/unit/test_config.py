from pathlib import Path

from pytest import MonkeyPatch

from fraud_detection.utils.config import project_root


def test_project_root_honors_runtime_override(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRAUD_PROJECT_ROOT", str(tmp_path))

    assert project_root() == tmp_path.resolve()
