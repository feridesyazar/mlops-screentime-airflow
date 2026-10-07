from __future__ import annotations

import json
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


# =========================================================
# PROJECT PATHS
# =========================================================

# screentime_mlops_dag.py is located directly
# inside the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent

# Allow Python / Airflow to find train_pipeline.py
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT MACHINE LEARNING PIPELINE
# =========================================================

from train_pipeline import (
    engineer_features,
    load_data,
    train_and_save,
    validate_data,
)


# =========================================================
# FILE PATHS
# =========================================================

RAW_PATH = PROJECT_ROOT / "screentime_analysis.csv"

PROCESSED_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "screentime_features.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


# =========================================================
# AIRFLOW DAG
# =========================================================

@dag(
    dag_id="screentime_mlops_pipeline",
    schedule="@daily",
    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz="UTC",
    ),
    catchup=False,
    tags=[
        "mlops",
        "machine-learning",
        "screentime",
    ],
)
def screentime_mlops_pipeline():

    # -----------------------------------------------------
    # TASK 1 - DATA VALIDATION
    # -----------------------------------------------------

    @task
    def validate_raw_data() -> str:

        df = load_data(RAW_PATH)

        report = validate_data(df)

        print("Validation report:")
        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        return str(RAW_PATH)


    # -----------------------------------------------------
    # TASK 2 - FEATURE ENGINEERING
    # -----------------------------------------------------

    @task
    def prepare_features(
        raw_path: str,
    ) -> str:

        df = load_data(raw_path)

        prepared = engineer_features(df)

        PROCESSED_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        prepared.to_csv(
            PROCESSED_PATH,
            index=False,
        )

        print(
            f"Prepared data saved to: "
            f"{PROCESSED_PATH}"
        )

        return str(PROCESSED_PATH)


    # -----------------------------------------------------
    # TASK 3 - MODEL TRAINING
    # -----------------------------------------------------

    @task
    def train_model(
        raw_path: str,
        processed_path: str,
    ) -> dict:

        metadata = train_and_save(
            raw_path=raw_path,
            artifact_dir=ARTIFACT_DIR,
            processed_path=processed_path,
        )

        print("Training completed.")

        return metadata


    # -----------------------------------------------------
    # TASK 4 - MODEL METRICS
    # -----------------------------------------------------

    @task
    def report_metrics(
        metadata: dict,
    ) -> None:

        metrics = metadata["metrics"]

        print(
            f"MAE={metrics['MAE']:.2f} | "
            f"RMSE={metrics['RMSE']:.2f} | "
            f"R2={metrics['R2']:.3f}"
        )

        print(
            f"Artifacts saved in: "
            f"{ARTIFACT_DIR}"
        )


    # =====================================================
    # PIPELINE ORDER
    # =====================================================

    raw_path = validate_raw_data()

    processed_path = prepare_features(
        raw_path
    )

    metadata = train_model(
        raw_path,
        processed_path,
    )

    report_metrics(metadata)


# =========================================================
# CREATE DAG
# =========================================================

screentime_mlops_pipeline()