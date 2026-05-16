# Databricks notebook source
# MAGIC %md
# MAGIC # 06 / Lakehouse Monitoring — drift + quality
# MAGIC
# MAGIC Creates a **snapshot monitor** on each training table to track distribution drift
# MAGIC over time. In a real deployment we would point an inference-log monitor at the
# MAGIC inference table emitted by Model Serving + AI Gateway; the wiring is the same.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorSnapshot

w = WorkspaceClient()

# COMMAND ----------

def create_or_update_snapshot_monitor(table_name: str) -> None:
    try:
        existing = w.quality_monitors.get(table_name=table_name)
        print(f"monitor exists for {table_name}: status={existing.status}")
        return
    except Exception:
        pass

    print(f"creating snapshot monitor for {table_name}")
    me = w.current_user.me().user_name
    safe_id = table_name.replace(".", "_")
    try:
        w.quality_monitors.create(
            table_name=table_name,
            assets_dir=f"/Workspace/Users/{me}/lakehouse_monitor_{safe_id}",
            output_schema_name=f"{CATALOG}.{SCHEMA}",
            snapshot=MonitorSnapshot(),
        )
    except Exception as e:
        # Workspaces have a quota on monitor count; treat as a non-fatal warning so
        # the pipeline doesn't block. In production, alert and clean up old monitors.
        msg = str(e)
        if "exceeds the number of limit" in msg or "ResourceExhausted" in msg or "Timed out" in msg:
            print(f"   WARN: could not create monitor for {table_name}: {msg}")
            return
        raise

# COMMAND ----------

create_or_update_snapshot_monitor(table("training_credit"))
create_or_update_snapshot_monitor(table("training_fraud"))

# COMMAND ----------

dbutils.notebook.exit("monitor_ok")
