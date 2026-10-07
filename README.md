# 🚀 MLOps Screen Time Prediction Pipeline with Apache Airflow

**End-to-end Machine Learning + MLOps portfolio project using Python, Scikit-learn, Apache Airflow and Docker.**

This project predicts **mobile app usage in minutes** and turns the complete machine-learning workflow into a reproducible and automated MLOps pipeline.

The goal was not only to train a model in a Jupyter Notebook, but to move from an experimental notebook workflow to a modular training pipeline that can be orchestrated, executed and monitored with **Apache Airflow**.

---

## Project at a Glance

This project demonstrates how a typical Machine Learning workflow can evolve into an MLOps-oriented architecture:

```text
Raw Data
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Date-Aware Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Model + Metrics + Predictions
   ↓
Apache Airflow Orchestration
   ↓
Dockerized Execution
```

### What was implemented

- Data loading and validation
- Feature engineering
- Date-aware train/test splitting
- Reproducible preprocessing with `ColumnTransformer`
- `MinMaxScaler` for numerical features
- `OneHotEncoder` for categorical features
- Random Forest regression
- Baseline comparison
- MAE, RMSE and R² evaluation
- Model artifact persistence with `joblib`
- Metrics and prediction export
- Reusable Python training pipeline
- Apache Airflow DAG with TaskFlow API
- Docker Compose environment for Airflow
- Successful end-to-end DAG execution

---

## Project Objective

The Machine Learning task is to predict:

**`Usage (minutes)`**

based on mobile application behavior such as:

- application name
- number of notifications
- number of times the application was opened
- calendar-based features derived from the date

The broader engineering objective is to demonstrate how a Machine Learning workflow can be made:

**reproducible · modular · automated · observable**

---

## Why this is an MLOps Project

A Machine Learning notebook is useful for experimentation, but a real workflow should also be repeatable and automated.

In this project, the notebook logic was transformed into reusable Python functions and then orchestrated with Apache Airflow.

The Airflow workflow contains four tasks:

```text
validate_raw_data
        ↓
prepare_features
        ↓
train_model
        ↓
report_metrics
```

This means that data validation, feature preparation, model training and metric reporting can run as one controlled workflow instead of being executed manually cell by cell.

---

## Key Engineering Decisions

### 1. Data Validation

Before model training, the pipeline checks:

- required columns
- missing values
- duplicate rows
- negative numeric values
- repeated `Date + App` combinations

This prevents the model from silently training on invalid input data.

### 2. Feature Engineering

The following features are created:

- `DayOfWeek`
- `DayOfMonth`
- `IsWeekend`
- `Notifications_x_TimesOpened`

These features add calendar and behavioral information to the model.

### 3. Date-Aware Train/Test Split

The dataset contains repeated `Date + App` combinations.

For this reason, the project deliberately avoids a naive row-based:

```python
shift(1)
```

as a "previous-day usage" feature, because the previous row is not guaranteed to represent the previous calendar day for the same application.

Instead, the project uses a grouped split by date so that the same date does not appear in both training and test data.

### 4. Reproducible Preprocessing

Preprocessing is implemented with:

```text
ColumnTransformer
├── Numerical Features → MinMaxScaler
└── Categorical Features → OneHotEncoder
```

The preprocessing and Random Forest model are combined in one Scikit-learn `Pipeline`.

This ensures that the same transformations are applied consistently during training and future prediction.

---

## Dataset

The dataset contains **200 observations**.

| Column | Description |
|---|---|
| `Date` | Observation date |
| `App` | Mobile application |
| `Usage (minutes)` | Target variable |
| `Notifications` | Number of notifications |
| `Times Opened` | Number of times the application was opened |

---

## Model

**Algorithm:** Random Forest Regressor

Random Forest was selected because it is:

- effective for small tabular datasets
- able to model nonlinear relationships
- robust and easy to interpret as a baseline
- well suited for demonstrating an end-to-end MLOps workflow

The focus of this project is not aggressive hyperparameter optimization, but the engineering of a reproducible ML training workflow.

---

## Model Results

The validated pipeline produced:

| Metric | Result |
|---|---:|
| Training rows | 160 |
| Test rows | 40 |
| MAE | ~16.84 min |
| RMSE | ~21.48 min |
| R² | ~0.373 |

The Random Forest also performed better than a simple mean-prediction baseline in terms of MAE.

Because the dataset is small and covers a limited period, the model results should be interpreted as a technical demonstration rather than a production-grade forecasting benchmark.

---

## Model Artifacts

The pipeline automatically creates:

```text
artifacts/
├── screentime_random_forest.joblib
├── metrics.json
├── test_predictions.csv
└── screentime_features.csv
```

### What these files represent

- **`.joblib`** → trained preprocessing + model pipeline
- **`metrics.json`** → evaluation metrics and model metadata
- **`test_predictions.csv`** → actual vs. predicted test results
- **`screentime_features.csv`** → processed feature dataset

This allows the result of each training run to be persisted outside the notebook.

---

## Apache Airflow Orchestration

The Airflow DAG uses the modern TaskFlow API:

```python
from airflow.sdk import dag, task
```

The workflow is scheduled daily for demonstration purposes:

```text
validate_raw_data
        ↓
prepare_features
        ↓
train_model
        ↓
report_metrics
```

The full pipeline was successfully executed in Apache Airflow with all four tasks completing successfully.

---

## Dockerized Airflow Environment

Apache Airflow is executed locally using Docker Compose.

Docker provides an isolated environment containing the Airflow services required for orchestration, including:

- Airflow API Server
- Scheduler
- DAG Processor
- Worker
- Triggerer
- PostgreSQL
- Redis

This makes the environment reproducible and avoids relying on a machine-specific Airflow installation.

---

## Repository Structure

A clean GitHub version of the project can be organized as:

```text
MLOps_Screentime_Prediction/
│
├── MLOps_Screentime_Prediction.ipynb
├── train_pipeline.py
├── screentime_mlops_dag.py
├── screentime_analysis.csv
│
├── artifacts/
│   ├── metrics.json
│   ├── test_predictions.csv
│   └── screentime_random_forest.joblib
│
├── docs/
│   ├── From_Notebook_to_MLOps_Pipeline_Practical_Guide.pdf
│   └── MLOps_From_Notebook_to_Pipeline_Guide.md
│
├── images/
│   └── airflow_pipeline_success.png
│
├── .airflowignore
├── .gitignore
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

Runtime folders such as `logs/`, `config/`, `plugins/`, `.ipynb_checkpoints/` and `__pycache__/` should not be committed to GitHub.

---

## Run the Notebook

Install the Machine Learning dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
MLOps_Screentime_Prediction.ipynb
```

The notebook contains:

- data understanding
- exploratory data analysis
- feature engineering
- preprocessing
- baseline model
- Random Forest model
- evaluation
- feature importance
- artifact creation

---

## Run the Training Pipeline Without Airflow

From the project folder:

```bash
python -c "from train_pipeline import train_and_save; print(train_and_save('screentime_analysis.csv', 'artifacts'))"
```

This runs:

```text
Load Data
   ↓
Validate Data
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Save Artifacts
```

---

## Run the Airflow Pipeline

Start the Docker environment:

```bash
docker compose up airflow-init
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

Open Airflow:

```text
http://localhost:8080
```

Default local demo credentials:

```text
Username: airflow
Password: airflow
```

Activate and trigger:

```text
screentime_mlops_pipeline
```

The expected successful workflow is:

```text
validate_raw_data ✓
prepare_features  ✓
train_model       ✓
report_metrics    ✓
```

Stop the environment when finished:

```bash
docker compose down
```

---

## Practical Lessons from the Project

This project also demonstrates several real engineering issues that appear when moving from a notebook to an orchestrated environment.

Examples include:

- Docker volume paths
- Airflow DAG discovery
- Jupyter `.ipynb_checkpoints`
- `.airflowignore`
- recursive DAG directory scanning
- dependency availability inside containers

These issues were diagnosed using Airflow task logs and Docker service logs.

This troubleshooting process is an important part of practical MLOps work: the goal is not only to write model code, but also to make the workflow execute reliably in its target environment.

---

## Technical Case Study

A detailed step-by-step explanation of the complete implementation is available as a separate technical guide.

**From Notebook to MLOps Pipeline**  
*A Practical Technical Case Study with Apache Airflow and Docker*

The guide explains:

- why a notebook alone is not an MLOps system
- how the ML pipeline was modularized
- how artifacts are created
- how Airflow orchestrates the workflow
- why Docker is used
- how the complete system fits together
- lessons learned during implementation

See:

```text
docs/From_Notebook_to_MLOps_Pipeline_Practical_Guide.pdf
docs/MLOps_From_Notebook_to_Pipeline_Guide.md
```

---

## Limitations

This project is intentionally a compact MLOps prototype.

Current limitations:

- only 200 observations
- limited historical time coverage
- no live data ingestion
- no production inference API
- no model registry
- no automated CI/CD pipeline
- no drift monitoring
- no cloud deployment

The project should therefore be viewed as a **production-oriented MLOps prototype**, not a complete production platform.

---

## Future Improvements

A Version 2 could extend the project with:

- MLflow experiment tracking
- MLflow Model Registry
- FastAPI prediction service
- GitHub Actions CI/CD
- automated tests with `pytest`
- data drift monitoring
- model drift monitoring
- cloud deployment
- larger historical dataset
- automated upstream data ingestion

---

## Skills Demonstrated

**Machine Learning**

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Random Forest` · `Feature Engineering` · `Model Evaluation`

**ML Engineering**

`ColumnTransformer` · `Pipeline` · `OneHotEncoder` · `MinMaxScaler` · `joblib` · `Model Artifacts`

**MLOps**

`Apache Airflow` · `TaskFlow API` · `DAG Orchestration` · `Docker` · `Docker Compose` · `Workflow Automation`

**Engineering Practices**

`Modular Python` · `Reproducible Training` · `Data Validation` · `Logging` · `Troubleshooting` · `Git/GitHub`

---

## Portfolio Summary

This project demonstrates the transition from an experimental Machine Learning notebook to a reproducible MLOps training workflow.

The key achievement is not only the prediction model itself, but the complete workflow around it:

**data validation → feature engineering → preprocessing → model training → evaluation → artifact management → Airflow orchestration → Docker execution**

This reflects the core idea of practical MLOps: making Machine Learning workflows repeatable, maintainable and automatable.

---

## Acknowledgement

The project was inspired by Aman Kharwal's article:

**MLOps Pipeline using Apache Airflow**

https://amanxai.com/2025/01/20/mlops-pipeline-using-apache-airflow/

The implementation was redesigned and extended with additional data validation, feature engineering, date-aware splitting, reusable preprocessing, model artifacts, Apache Airflow 3 TaskFlow API and Docker-based orchestration.
