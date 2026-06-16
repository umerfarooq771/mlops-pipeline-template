.PHONY: setup data train evaluate test app api clean

setup:
	pip install -r requirements.txt

data:
	python data/generate_synthetic_data.py

train: data
	python src/training/train.py

evaluate: train
	python src/evaluation/evaluate.py

test:
	pytest tests/ -v --tb=short

app: train
	streamlit run app.py

api: train
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

pipeline: data train evaluate
	@echo "Full pipeline complete."

docker-build:
	docker build -t mlops-pipeline .

docker-run:
	docker run -p 8000:8000 mlops-pipeline

clean:
	rm -rf models/ mlruns/ docs/evaluation/ docs/monitoring/
	find . -type d -name __pycache__ -exec rm -rf {} +
