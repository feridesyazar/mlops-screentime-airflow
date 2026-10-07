# 🚀 MLOps Screen Time Prediction Pipeline with Apache Airflow

**End-to-end Machine Learning + MLOps portfolio project using Python, Scikit-learn, Apache Airflow and Docker.**

This project predicts **mobile app usage in minutes** and transforms the complete Machine Learning workflow into a reproducible and automated MLOps pipeline.

The goal was not only to train a model in a Jupyter Notebook, but to move from an experimental notebook workflow to a modular training pipeline that can be executed, orchestrated and monitored with **Apache Airflow**.

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
