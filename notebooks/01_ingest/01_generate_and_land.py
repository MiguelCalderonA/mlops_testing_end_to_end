# Databricks notebook source
# MAGIC %md
# MAGIC # 01 / Generate synthetic data and land into Bronze
# MAGIC
# MAGIC Six tables for the Lending Fintech, 1000 rows each:
# MAGIC
# MAGIC | Table | Description |
# MAGIC |---|---|
# MAGIC | `raw_applications` | Loan applications |
# MAGIC | `raw_kyc_identity` | Customer / KYC identity |
# MAGIC | `raw_credit_bureau` | Bureau attributes |
# MAGIC | `raw_internal_transactions` | Internal banking / repayment summary |
# MAGIC | `raw_fraud_events` | Fraud / verification signals |
# MAGIC | `raw_historical_outcomes` | Labels (default / fraud) |
# MAGIC
# MAGIC Re-runnable: writes use `mode("overwrite")` and Delta tables.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

import sys
sys.path.insert(0, f"{REPO_ROOT}/scripts")
from generate_synthetic_data import generate_all  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("rows", str(CFG["synthetic_rows"]), "Synthetic rows per table")
dbutils.widgets.text("seed", str(CFG["random_seed"]), "Random seed")
rows = int(dbutils.widgets.get("rows"))
seed = int(dbutils.widgets.get("seed"))
print(f"Generating {rows} rows per table (seed={seed})")

tables = generate_all(rows=rows, seed=seed)

# COMMAND ----------

mapping = {
    "kyc_identity": "raw_kyc",
    "applications": "raw_applications",
    "credit_bureau": "raw_bureau",
    "internal_transactions": "raw_transactions",
    "fraud_events": "raw_fraud_events",
    "historical_outcomes": "raw_outcomes",
}

for source_name, cfg_key in mapping.items():
    pdf = tables[source_name]
    sdf = spark.createDataFrame(pdf)
    target = table(cfg_key)
    (sdf.write.format("delta")
        .option("overwriteSchema", "true")
        .mode("overwrite")
        .saveAsTable(target))
    print(f"wrote {pdf.shape[0]} rows -> {target}")

# COMMAND ----------

# Quick sanity check
for cfg_key in mapping.values():
    n = spark.table(table(cfg_key)).count()
    print(f"{table(cfg_key):60s} rows={n}")

# COMMAND ----------

dbutils.notebook.exit("ingest_ok")
