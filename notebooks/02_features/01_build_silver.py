# Databricks notebook source
# MAGIC %md
# MAGIC # 02a / Build Silver tables
# MAGIC
# MAGIC Lightweight cleanup, deduplication, and typing on top of bronze.
# MAGIC Each silver table is a 1:1 normalized version of its bronze counterpart.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

def upsert_silver(source_key: str, target_key: str, dedup_keys: list[str]) -> None:
    src = spark.table(table(source_key))
    # de-duplicate keeping the latest record by ingestion order — for demo simplicity we drop dupes
    cleaned = src.dropDuplicates(dedup_keys)
    (cleaned.write.format("delta")
        .option("overwriteSchema", "true")
        .mode("overwrite")
        .saveAsTable(table(target_key)))
    print(f"{table(target_key):60s} rows={cleaned.count()}")

# COMMAND ----------

upsert_silver("raw_applications", "silver_applications", ["application_id"])
upsert_silver("raw_kyc",          "silver_kyc",          ["customer_id"])
upsert_silver("raw_bureau",       "silver_bureau",       ["application_id"])
upsert_silver("raw_transactions", "silver_transactions", ["customer_id"])
upsert_silver("raw_fraud_events", "silver_fraud_events", ["event_id"])

# COMMAND ----------

dbutils.notebook.exit("silver_ok")
