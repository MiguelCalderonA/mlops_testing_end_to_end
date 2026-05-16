# Databricks notebook source
# MAGIC %md
# MAGIC # 03a / Train credit-risk model
# MAGIC
# MAGIC - Reads `gold_training_credit`
# MAGIC - Trains a LightGBM-style gradient-boosted classifier (sklearn `HistGradientBoostingClassifier`)
# MAGIC   so no extra cluster libraries are needed.
# MAGIC - Logs to MLflow, registers the model into Unity Catalog,
# MAGIC   and sets the `@candidate` alias on the new version.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_path = f"/Users/{user_email}/{CFG['credit_experiment_name']}"
mlflow.set_experiment(experiment_path)
print(f"experiment: {experiment_path}")

# COMMAND ----------

pdf: pd.DataFrame = spark.table(table("training_credit")).toPandas()
print(f"training rows: {len(pdf)}, positives: {pdf['label'].sum()}")

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
    ("clf", HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# COMMAND ----------

with mlflow.start_run(run_name="credit_risk_baseline") as run:
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate_train": float(y_train.mean()),
    }
    mlflow.log_metrics(metrics)
    mlflow.log_params({"max_iter": 200, "max_depth": 6, "learning_rate": 0.08})

    sig = infer_signature(X_train.head(5), pipe.predict_proba(X_train.head(5)))
    model_name = f"{CATALOG}.{SCHEMA}.{CFG['credit_model_name']}"
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
# Pick the latest version of the registered model — that's the one we just logged
versions = client.search_model_versions(f"name='{model_name}'")
new_version = sorted(versions, key=lambda v: int(v.version))[-1].version
client.set_registered_model_alias(model_name, "candidate", new_version)
print(f"alias @candidate -> {model_name} v{new_version}")

# COMMAND ----------

dbutils.notebook.exit(f"credit_train_ok:{new_version}")
