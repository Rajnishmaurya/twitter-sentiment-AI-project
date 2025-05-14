Twitter Sentiment Analysis – End-to-End ML Application

This project is a production-grade machine learning pipeline for classifying sentiments from tweets using a Support Vector Machine (SVM) classifier with TF-IDF features. It integrates modern MLOps practices such as Docker, Prometheus-Grafana monitoring, MLflow tracking, DVC pipelines, and drift detection.

---

## 📁 Project Structure

```
twitter-sentiment-AI-project/
│
├── data/                   # Raw & processed data
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/               # Data ingestion and preprocessing scripts
│   ├── models/             # Model training logic
│   ├── deployment/         # FastAPI app with drift monitoring
│   ├── drift/              # Drift detection and PushGateway integration
│   ├── monitoring/         # Prometheus and Grafana setup
│   └── evaluation/         # Model/unit tests
│
├── logs/                  # Inference logs and recent predictions
├── mlruns/                # MLflow tracking data
├── dvc.yaml               # DVC pipeline definition
├── docker-compose.yml     # Docker service configuration
├── prometheus.yml         # Prometheus scrape targets
└── README.md              # This file
```

---

## Features

### Functional

* Predict tweet sentiment (positive/negative)
* TF-IDF vectorizer + SVM model
* FastAPI frontend UI
* Drift detection using cosine distance

### MLOps

* **MLflow** for experiment tracking
* **DVC** for data and pipeline management
* **Prometheus** for real-time metric scraping
* **Grafana** dashboards for visualization
* **PushGateway** for pushing custom drift metrics
* **Docker Compose** for container orchestration

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Rajnishmaurya/twitter-sentiment-AI-project.git
cd twitter-sentiment-AI-project
```

### 2. Create and Activate Environment

```bash
conda create -n twitter-sentiment python=3.9 -y
conda activate twitter-sentiment
pip install -r requirements.txt
```

### 3. Run Data Pipeline

```bash
dvc repro
dvc dag
```

### 4. Run Locally (without Docker)

```bash
uvicorn src.deployment.api:app --reload
```

Visit: [http://localhost:8000](http://localhost:8000)

### 5. Run MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

Visit: [http://localhost:5000](http://localhost:5000)

---

##  Docker Deployment

### Build and Launch All Services

```bash
docker-compose up --build -d
```

### Containers Started:

* `fastapi`: model inference + drift detection
* `prometheus`: collects app metrics
* `grafana`: dashboards
* `pushgateway`: receives metrics from drift logic
* `node-exporter`: system-level metrics

---

## Monitoring

### Prometheus UI:

[http://localhost:9090](http://localhost:9090)

### Grafana UI:

[http://localhost:3000](http://localhost:3000)
Login: `admin` / `admin`

### Useful Metrics:

* `http_requests_total`
* `http_request_duration_seconds`
* `data_drift_detected`
* `process_resident_memory_bytes`
* `node_cpu_seconds_total`

---

## Drift Detection

Implemented using cosine similarity on vectorized text (recent vs. training).

* Runs every 20 seconds
* Logs in console and Prometheus (via PushGateway)
* Triggers `data_drift_detected = 1` if drift > 0.3

---

##  Testing

### Run Unit Tests

```bash
pytest tests/
```

### Files:

* `test_model.py`: Ensures SVM predicts correctly
* `test_data_pipeline.py`: Validates text cleaning logic

---

##  Documentation Checklist

### 1. Architecture Diagram

* Includes Docker, FastAPI, MLflow, DVC, and Monitoring blocks

### 2. High Level Design (HLD)

* Loose coupling between API & ML model
* Modular structure with clear separation

### 3. Low Level Design (LLD)

* `/` → Home form
* `/predict` → POST form text → predicts sentiment

### 4. Test Plan

* Unit tests for preprocessing and model
* Acceptance: 95%+ accuracy

### 5. User Manual

Visit `localhost:8000`

* Enter tweet → click predict → get label
* All predictions logged + used for drift

---

##  Deliverables

* Dockerized containers: app, monitoring stack
* MLflow artifacts
* DVC pipeline with DAG
* PushGateway alerts
* Grafana Dashboards
* Drift logic (custom metrics)
* README, HLD, LLD, and test report

---


---

*This project demonstrates the full MLOps lifecycle from data ingestion to production deployment and monitoring.*
