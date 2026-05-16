# Databricks notebook source
# MAGIC %md
# MAGIC # 05b / Smoke-test deployed endpoints
# MAGIC
# MAGIC Send a few rows of recent data through each serving endpoint to confirm it returns 200s.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

import json
import time
import pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

def _await_ready(endpoint_name: str, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ep = w.serving_endpoints.get(endpoint_name)
        state = ep.state.config_update if ep.state else None
        ready = ep.state and ep.state.ready and ep.state.ready.value == "READY"
        print(f"{endpoint_name}: ready={ready}, config_update={state}")
        if ready:
            return
        time.sleep(20)
    raise RuntimeError(f"endpoint {endpoint_name} not ready after {timeout_s}s")


def _score(endpoint_name: str, df: pd.DataFrame) -> None:
    # Records format is a list-of-dicts; the SDK accepts it natively without a typed wrapper.
    records = df.to_dict(orient="records")
    resp = w.serving_endpoints.query(name=endpoint_name, dataframe_records=records)
    print(f"{endpoint_name}: predictions[:3] = {resp.predictions[:3]}")


# COMMAND ----------

credit_sample = (spark.table(table("training_credit")).limit(5).toPandas()
                 .drop(columns=["application_id", "customer_id", "application_date", "label"]))
fraud_sample = (spark.table(table("training_fraud")).limit(5).toPandas()
                .drop(columns=["application_id", "customer_id", "application_date", "label"]))

for ep in [CFG["credit_endpoint_name"], CFG["fraud_endpoint_name"]]:
    _await_ready(ep)

_score(CFG["credit_endpoint_name"], credit_sample)
_score(CFG["fraud_endpoint_name"], fraud_sample)

# COMMAND ----------

dbutils.notebook.exit("smoke_ok")
