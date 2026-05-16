# Databricks notebook source
# MAGIC %md
# MAGIC # 00 / Bootstrap Unity Catalog
# MAGIC
# MAGIC Idempotent: creates the catalog, schema and a managed volume used to land synthetic
# MAGIC data. Safe to re-run on any workspace.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# Try CREATE CATALOG IF NOT EXISTS first — this is the common, friendly path on
# workspaces whose metastore has a default managed-storage root configured. On
# locked-down workspaces (no default storage root, or the user lacks managed-storage
# privileges), this SQL fails. In that case, fall back to verifying the catalog
# exists — the customer is expected to have pre-created it via Catalog Explorer
# or `databricks catalogs create <name> --storage-root s3://...`.
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    print(f"Catalog {CATALOG} created (or already existed) via SQL.")
except Exception as e:
    print(f"CREATE CATALOG failed ({e!s:.200}); falling back to existence check.")
    catalog_exists = spark.sql(f"SHOW CATALOGS LIKE '{CATALOG}'").count() > 0
    if not catalog_exists:
        raise RuntimeError(
            f"Catalog '{CATALOG}' does not exist and CREATE CATALOG was rejected. "
            f"Create it via the Catalog Explorer or "
            f"`databricks catalogs create {CATALOG} --storage-root <s3://...>`, then re-run."
        )
    print(f"Catalog {CATALOG} verified to exist.")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

print(f"Verified {CATALOG}.{SCHEMA} schema and {VOLUME} volume.")

# COMMAND ----------

dbutils.notebook.exit("bootstrap_ok")
