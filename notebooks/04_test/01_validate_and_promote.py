# Databricks notebook source
# MAGIC %md
# MAGIC # 04 / Automated model validation + champion promotion
# MAGIC
# MAGIC For each model (credit and fraud):
# MAGIC
# MAGIC 1. Load the **@candidate** version
# MAGIC 2. Load a held-out validation slice from the training table
# MAGIC 3. Run a battery of automated checks:
# MAGIC    - AUC ≥ threshold
# MAGIC    - Average precision ≥ threshold
# MAGIC    - No NaN predictions, signature matches input
# MAGIC    - Performance on a "fairness slice" (age group) is not catastrophically worse
# MAGIC 4. If the candidate beats the current `@champion` (or there is none), move the
# MAGIC    `@champion` alias to the candidate.
# MAGIC
# MAGIC The decision is logged as an MLflow tag on the candidate version.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

CREDIT = f"{CATALOG}.{SCHEMA}.{CFG['credit_model_name']}"
FRAUD = f"{CATALOG}.{SCHEMA}.{CFG['fraud_model_name']}"

# TUNE THESE per business risk tolerance. Demo defaults are deliberately
# permissive so the pipeline goes green on 1000 synthetic rows. In production:
#   - derive auc_min from a backtest on real data
#   - add fairness gates (PSI / TPR delta on protected segments)
#   - add a calibration check (e.g., Brier score below a ceiling)
#   - require the candidate to beat the current champion by a margin, not just by epsilon
THRESHOLDS = {
    "credit": {"auc_min": 0.65, "avg_precision_min": 0.30},
    "fraud":  {"auc_min": 0.65, "avg_precision_min": 0.10},
}

# COMMAND ----------

def _load_validation(table_key: str) -> tuple[pd.DataFrame, pd.Series]:
    pdf = spark.table(table(table_key)).toPandas()
    pdf = pdf.sample(frac=1.0, random_state=99).reset_index(drop=True)
    n_val = max(int(len(pdf) * 0.25), 50)
    val = pdf.tail(n_val).copy()
    y = val.pop("label").astype(int)
    drop_cols = ["application_id", "customer_id", "application_date"]
    return val.drop(columns=drop_cols), y


def _validate(model_name: str, table_key: str, thresholds: dict) -> dict:
    cand = client.get_model_version_by_alias(model_name, "candidate")
    cand_uri = f"models:/{model_name}@candidate"
    print(f"loading {model_name} @candidate (v{cand.version})")
    model = mlflow.sklearn.load_model(cand_uri)

    X_val, y_val = _load_validation(table_key)
    proba = model.predict_proba(X_val)[:, 1]

    auc = float(roc_auc_score(y_val, proba))
    ap = float(average_precision_score(y_val, proba))
    nan_share = float(np.isnan(proba).mean())

    # Fairness-style slice: under-35 vs 35+
    age = X_val["age"].fillna(X_val["age"].median())
    under = age < 35
    auc_under = float(roc_auc_score(y_val[under], proba[under])) if under.sum() > 10 and y_val[under].nunique() > 1 else None
    auc_over = float(roc_auc_score(y_val[~under], proba[~under])) if (~under).sum() > 10 and y_val[~under].nunique() > 1 else None

    checks = {
        "auc_ok": auc >= thresholds["auc_min"],
        "ap_ok": ap >= thresholds["avg_precision_min"],
        "nan_ok": nan_share == 0.0,
        "fairness_ok": (
            auc_under is None or auc_over is None
            or abs(auc_under - auc_over) < 0.15
        ),
    }
    passed = all(checks.values())

    result = {
        "version": cand.version,
        "auc": auc,
        "avg_precision": ap,
        "auc_under_35": auc_under,
        "auc_over_35": auc_over,
        "nan_share": nan_share,
        "checks": checks,
        "passed": passed,
    }
    # Tag the version with the outcome
    for k, v in {"validation_auc": auc, "validation_ap": ap, "validation_passed": str(passed)}.items():
        client.set_model_version_tag(model_name, cand.version, k, str(v))
    return result


def _promote_if_better(model_name: str, result: dict, metric: str = "auc") -> str:
    if not result["passed"]:
        print(f"   validation FAILED — not promoting v{result['version']}")
        return "rejected"
    cand_version = result["version"]
    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
        champ_auc = float(client.get_model_version(model_name, champ.version).tags.get("validation_auc", "0"))
        if result[metric] > champ_auc:
            client.set_registered_model_alias(model_name, "champion", cand_version)
            print(f"   promoted v{cand_version} (auc={result[metric]:.3f}) over v{champ.version} (auc={champ_auc:.3f})")
            return "promoted"
        else:
            print(f"   v{cand_version} (auc={result[metric]:.3f}) did not beat v{champ.version} (auc={champ_auc:.3f})")
            return "challenger"
    except Exception:
        # No champion yet -> first promotion
        client.set_registered_model_alias(model_name, "champion", cand_version)
        print(f"   first champion: v{cand_version}")
        return "first_champion"


# COMMAND ----------

credit_result = _validate(CREDIT, "training_credit", THRESHOLDS["credit"])
credit_outcome = _promote_if_better(CREDIT, credit_result)

fraud_result = _validate(FRAUD, "training_fraud", THRESHOLDS["fraud"])
fraud_outcome = _promote_if_better(FRAUD, fraud_result)

# COMMAND ----------

print({"credit": credit_result, "fraud": fraud_result})
print({"credit_outcome": credit_outcome, "fraud_outcome": fraud_outcome})

# COMMAND ----------

dbutils.notebook.exit(f"credit:{credit_outcome},fraud:{fraud_outcome}")
