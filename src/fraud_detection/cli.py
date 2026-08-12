"""Operator CLI for the complete local workflow."""

from __future__ import annotations

import json

import typer

from fraud_detection.data.eda import generate_eda
from fraud_detection.data.ingestion import download_dataset, verify_source_files
from fraud_detection.data.preprocessing import process_raw_data
from fraud_detection.data.validation import validate_raw_data
from fraud_detection.explainability.shap_explainer import generate_shap_artifacts
from fraud_detection.features.pipeline import build_features
from fraud_detection.monitoring.workflow import run_drift_monitor
from fraud_detection.streaming.consumer import run_consumer
from fraud_detection.streaming.labels import run_label_consumer
from fraud_detection.streaming.outbox import publish_pending
from fraud_detection.streaming.producer import replay
from fraud_detection.training.train import run_benchmark
from fraud_detection.utils.config import load_settings
from fraud_detection.utils.logging import configure_logging

app = typer.Typer(help="Real-time fraud detection platform")
data_app = typer.Typer(help="Download, validate, process, and explore Sparkov")
features_app = typer.Typer(help="Build leakage-safe fraud features")
train_app = typer.Typer(help="Benchmark and govern fraud models")
stream_app = typer.Typer(help="Produce and consume Kafka events")
monitor_app = typer.Typer(help="Run drift and performance monitoring")
demo_app = typer.Typer(help="Bootstrap a sanitized local demonstration")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(stream_app, name="stream")
app.add_typer(monitor_app, name="monitor")
app.add_typer(demo_app, name="demo")


def _show(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, default=str))


@data_app.command("download")
def data_download(force: bool = False) -> None:
    _show([str(path) for path in download_dataset(load_settings("portfolio"), force)])


@data_app.command("validate")
def data_validate() -> None:
    settings = load_settings("portfolio")
    verify_source_files(settings)
    _show(validate_raw_data(settings))


@data_app.command("process")
def data_process(profile: str = "portfolio") -> None:
    settings = load_settings(profile)
    _show({"output": str(process_raw_data(settings))})


@data_app.command("eda")
def data_eda(profile: str = "portfolio") -> None:
    _show(generate_eda(load_settings(profile)))


@features_app.command("build")
def features_build(profile: str = "portfolio") -> None:
    _show({"output": str(build_features(load_settings(profile)))})


@train_app.command("benchmark")
def train_benchmark(profile: str = "portfolio") -> None:
    _show(run_benchmark(load_settings(profile)))


@train_app.command("explain")
def train_explain(profile: str = "portfolio") -> None:
    _show(generate_shap_artifacts(load_settings(profile)))


@stream_app.command("replay")
def stream_replay(profile: str = "demo", limit: int = 1000, rate: float = 50.0) -> None:
    _show({"published": replay(load_settings(profile), limit, rate)})


@stream_app.command("consume")
def stream_consume(profile: str = "demo") -> None:
    run_consumer(load_settings(profile))


@stream_app.command("outbox")
def stream_outbox(profile: str = "demo", once: bool = False) -> None:
    _show({"published": publish_pending(load_settings(profile), once)})


@stream_app.command("labels")
def stream_labels(profile: str = "demo") -> None:
    run_label_consumer(load_settings(profile))


@monitor_app.command("run")
def monitor_run(profile: str = "portfolio") -> None:
    _show(run_drift_monitor(load_settings(profile)))


@demo_app.command("bootstrap")
def demo_bootstrap() -> None:
    settings = load_settings("demo")
    download_dataset(settings)
    validate_raw_data(load_settings("portfolio"))
    process_raw_data(settings)
    build_features(settings)
    _show(run_benchmark(settings))


if __name__ == "__main__":
    configure_logging()
    app()
