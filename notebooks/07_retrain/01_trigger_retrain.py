# Databricks notebook source
# MAGIC %md
# MAGIC # 07 / Retrain trigger
# MAGIC
# MAGIC A thin pass-through that calls the training + validation + deploy notebooks again,
# MAGIC marking the new MLflow run with `triggered_by = drift_alert`. The orchestrating job
# MAGIC wires this in via a `condition_task` that only runs if `06_monitor/02_drift_check`
# MAGIC exits with `retrain_required`.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# Run training again — tagging this batch as drift-triggered
with mlflow.start_run(run_name="drift_triggered_retrain") as run:
    mlflow.set_tag("triggered_by", "drift_alert")
    dbutils.notebook.run("../03_train/01_train_credit_risk", timeout_seconds=3600)
    dbutils.notebook.run("../03_train/02_train_fraud_detection", timeout_seconds=3600)
    dbutils.notebook.run("../04_test/01_validate_and_promote", timeout_seconds=1800)
    dbutils.notebook.run("../05_deploy/01_deploy_endpoints", timeout_seconds=1800)

dbutils.notebook.exit("retrain_complete")
