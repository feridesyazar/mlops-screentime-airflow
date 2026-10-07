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

During execution, the pipeline creates the following artifacts:

```text
artifacts/
├── screentime_random_forest.joblib
├── metrics.json
├── test_predictions.csv
└── screentime_features.csv
```

### What these files represent

- **`screentime_random_forest.joblib`** → trained preprocessing + model pipeline
- **`metrics.json`** → evaluation metrics and model metadata
- **`test_predictions.csv`** → actual vs. predicted test results
- **`screentime_features.csv`** → processed feature dataset

The repository includes `metrics.json` and `test_predictions.csv` as example outputs from a completed training run.

The trained model and processed feature dataset can be regenerated by executing the training pipeline.

This allows the results of a training run to be persisted outside the notebook while keeping the repository lightweight.

---

## Apache Airflow Orchestration

The Airflow DAG uses the modern TaskFlow API:

```python
from airflow.sdk import dag, task
```

The workflow is scheduled daily for demonstration purposes and can also be triggered manually:

```text
validate_raw_data
        ↓
prepare_features
        ↓
train_model
        ↓
report_metrics
```

Each task has a clear responsibility:

- `validate_raw_data` checks the incoming dataset
- `prepare_features` performs feature engineering and prepares the data
- `train_model` executes the reusable Machine Learning training pipeline
- `report_metrics` reports the final model performance

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

The Docker environment also allows the complete MLOps workflow to be started, stopped and recreated consistently.

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

The notebook is intended for exploration and understanding of the Machine Learning workflow.

The reusable production-oriented training logic is implemented separately in `train_pipeline.py`.

---

## Run the Training Pipeline Without Airflow

The Machine Learning pipeline can also be executed independently of Airflow.

From the project folder:

```bash
python -c "from train_pipeline import train_and_save; print(train_and_save('screentime_analysis.csv', 'artifacts'))"
```

This executes the complete training workflow:

```text
Load Data
   ↓
Validate Data
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Build Preprocessing Pipeline
   ↓
Train Random Forest
   ↓
Evaluate Model
   ↓
Save Artifacts
```

This separation is important because the Machine Learning logic does not depend on Airflow.

Airflow orchestrates the process, while `train_pipeline.py` contains the reusable Machine Learning functionality.

---

## Run the Airflow Pipeline

Initialize the Docker environment:

```bash
docker compose up airflow-init
```

Start the Airflow services:

```bash
docker compose up -d
```

Check the container status:

```bash
docker compose ps
```

The Airflow services should report a healthy status.

Open the Airflow interface:

```text
http://localhost:8080
```

Default local demo credentials:

```text
Username: airflow
Password: airflow
```

In the Airflow interface, activate and trigger:

```text
screentime_mlops_pipeline
```

The expected successful workflow is:

```text
validate_raw_data ✓
        ↓
prepare_features  ✓
        ↓
train_model       ✓
        ↓
report_metrics    ✓
```

When all four tasks are green, the complete MLOps training workflow has executed successfully.

Stop the Docker environment when finished:

```bash
docker compose down
```

---

## Practical Lessons from the Project

Moving from a Jupyter Notebook to an orchestrated environment introduced several practical engineering challenges.

Examples included:

- Docker volume paths
- Airflow DAG discovery
- Jupyter `.ipynb_checkpoints`
- `.airflowignore`
- recursive DAG directory scanning
- Python dependency availability inside containers
- validating container and service health
- diagnosing task failures through Airflow logs

These issues were diagnosed using Airflow task logs and Docker service logs.

One example was Airflow detecting a Jupyter checkpoint file as an additional DAG. This caused the pipeline to use an incorrect project path.

Another issue was recursive DAG directory scanning when Airflow attempted to scan its own runtime log directories.

These issues were resolved by configuring `.airflowignore` correctly and excluding runtime directories from DAG discovery.

This troubleshooting process is an important part of practical MLOps work: the goal is not only to write Machine Learning code, but also to make the workflow execute reliably in its target environment.

---

## 📘 Technical Case Study

A detailed step-by-step explanation of the complete project is also available as a technical guide.

### [From Notebook to MLOps Pipeline – Practical Technical Case Study](./From_Notebook_to_MLOps_Pipeline_Practical_Guide.pdf)

The guide explains:

- how a notebook-based Machine Learning project evolves into an MLOps workflow
- why reusable Python modules are important
- how data validation is integrated into the pipeline
- how feature engineering and preprocessing are organized
- how model artifacts are created
- how Apache Airflow orchestrates the workflow
- why Docker is used
- how the components work together
- practical implementation challenges and troubleshooting lessons
- how the project can be explained in an MLOps engineering context

The guide is designed to make the complete architecture understandable even for readers who are new to MLOps.

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
- no data drift monitoring
- no model drift monitoring
- no cloud deployment

The project should therefore be viewed as a **production-oriented MLOps prototype**, not a complete production platform.

The primary objective is to demonstrate the engineering transition from a notebook-based Machine Learning workflow to a reproducible and automated training pipeline.

---

## Future Improvements

A future version could extend the project with:

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
- scheduled retraining based on new data
- model versioning and promotion workflows

These additions would extend the current training pipeline toward a more complete production MLOps platform.

---

## Skills Demonstrated

### Machine Learning

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Random Forest` · `Feature Engineering` · `Model Evaluation`

### ML Engineering

`ColumnTransformer` · `Pipeline` · `OneHotEncoder` · `MinMaxScaler` · `joblib` · `Model Artifacts` · `Reusable Training Logic`

### MLOps

`Apache Airflow` · `TaskFlow API` · `DAG Orchestration` · `Docker` · `Docker Compose` · `Workflow Automation`

### Engineering Practices

`Modular Python` · `Reproducible Training` · `Data Validation` · `Logging` · `Troubleshooting` · `Environment Isolation` · `Git/GitHub`

---

## Portfolio Summary

This project demonstrates the transition from an experimental Machine Learning notebook to a reproducible MLOps training workflow.

The key achievement is not only the prediction model itself, but the complete engineering workflow around it:

**data validation → feature engineering → preprocessing → model training → evaluation → artifact management → Airflow orchestration → Docker execution**

The notebook provides an interactive environment for understanding and evaluating the model.

The reusable Python training pipeline separates the Machine Learning logic from the notebook.

Apache Airflow orchestrates the individual workflow stages.

Docker provides the reproducible execution environment required to run the complete system.

Together, these components demonstrate the core idea of practical MLOps:

**making Machine Learning workflows reproducible, modular, observable, maintainable and automatable.**
