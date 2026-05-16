# Databricks notebook source
# MAGIC %md
# MAGIC # 03b / Train fraud-detection model
# MAGIC
# MAGIC Mirrors the credit notebook but trains on `gold_training_fraud` with `is_fraud` as the label.
# MAGIC Class imbalance is heavier here, so we use `class_weight` via sample weights.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_registry_uri("databricks-uc")
user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_path = f"/Users/{user_email}/{CFG['fraud_experiment_name']}"
mlflow.set_experiment(experiment_path)
print(f"experiment: {experiment_path}")

# COMMAND ----------

pdf: pd.DataFrame = spark.table(table("training_fraud")).toPandas()
print(f"rows: {len(pdf)}, positives: {pdf['label'].sum()}")

drop = ["application_id", "customer_id", "application_date", "label"]
y = pdf["label"].astype(int)
X = pdf.drop(columns=drop)

num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

pipe = Pipeline([
    ("pre", pre),
    ("clf", HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.06)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Up-weight the minority class
pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
sample_weight = np.where(y_train == 1, pos_weight, 1.0)

# COMMAND ----------

with mlflow.start_run(run_name="fraud_baseline") as run:
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate_train": float(y_train.mean()),
        "pos_class_weight": float(pos_weight),
    }
    mlflow.log_metrics(metrics)
    mlflow.log_params({"max_iter": 300, "max_depth": 5, "learning_rate": 0.06})

    sig = infer_signature(X_train.head(5), pipe.predict_proba(X_train.head(5)))
    model_name = f"{CATALOG}.{SCHEMA}.{CFG['fraud_model_name']}"
    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="model",
        signature=sig,
        registered_model_name=model_name,
        input_example=X_train.head(2),
    )
    print(f"metrics: {metrics}")

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{model_name}'")
new_version = sorted(versions, key=lambda v: int(v.version))[-1].version
client.set_registered_model_alias(model_name, "candidate", new_version)
print(f"alias @candidate -> {model_name} v{new_version}")

# COMMAND ----------

dbutils.notebook.exit(f"fraud_train_ok:{new_version}")
