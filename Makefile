.PHONY: install quality test integration download validate process features eda train api dashboard mlflow monitor demo benchmark clean

install:
	python -m pip install -e '.[all]'

quality:
	ruff format --check src tests dashboards scripts
	ruff check src tests dashboards scripts
	mypy src

test:
	pytest -m 'not integration and not full_data and not container'

integration:
	pytest -m integration

download:
	fraud-detect data download

validate:
	fraud-detect data validate

process:
	fraud-detect data process

features:
	fraud-detect features build

eda:
	fraud-detect data eda

train:
	fraud-detect train benchmark --profile portfolio

api:
	uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboards/app.py

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts/mlflow --port 5000

monitor:
	fraud-detect monitor run

demo:
	fraud-detect demo bootstrap

benchmark:
	python scripts/benchmark_api.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist
