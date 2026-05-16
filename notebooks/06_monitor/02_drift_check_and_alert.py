# Databricks notebook source
# MAGIC %md
# MAGIC # 06b / Drift check — decide whether to retrain
# MAGIC
# MAGIC Reads the latest Lakehouse Monitor profile metrics and compares the current window
# MAGIC to the baseline. If any of the watched feature columns has a Jensen-Shannon distance
# MAGIC above a threshold, this notebook exits with the literal value `retrain_required`,
# MAGIC which the orchestrating Job uses as a conditional branch (`if/else task`).

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

dbutils.widgets.text("js_threshold", "0.20", "JS distance threshold for drift")
TH = float(dbutils.widgets.get("js_threshold"))

# COMMAND ----------

# Lakehouse Monitoring writes `<table>_profile_metrics` and `<table>_drift_metrics` tables.
def latest_drift(monitored_table: str) -> float:
    drift_table = f"{monitored_table}_drift_metrics"
    try:
        df = spark.sql(f"""
            SELECT MAX(js_distance) AS max_js
            FROM {drift_table}
            WHERE granularity = '1 day'
              AND window.end = (SELECT MAX(window.end) FROM {drift_table})
        """)
        return float(df.collect()[0]["max_js"] or 0.0)
    except Exception as e:
        print(f"   drift table not yet available ({e}); returning 0.0")
        return 0.0


credit_drift = latest_drift(table("training_credit"))
fraud_drift = latest_drift(table("training_fraud"))
print(f"credit max JS = {credit_drift:.3f}, fraud max JS = {fraud_drift:.3f}, threshold = {TH}")

# COMMAND ----------

needs_retrain = credit_drift > TH or fraud_drift > TH
result = "retrain_required" if needs_retrain else "no_action"

# Expose result both as a task value (read by the condition_task) and as the exit value
dbutils.jobs.taskValues.set(key="result", value=result)
dbutils.notebook.exit(result)
