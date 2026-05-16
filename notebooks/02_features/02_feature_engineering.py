# Databricks notebook source
# MAGIC %md
# MAGIC # 02b / Feature engineering
# MAGIC
# MAGIC Joins silver tables and creates engineered features. Two training tables are emitted:
# MAGIC
# MAGIC - `gold_training_credit` — joined features + `defaulted` label
# MAGIC - `gold_training_fraud` — joined features + `is_fraud` label
# MAGIC
# MAGIC We also register two **UC Feature Tables** (Databricks Feature Engineering in Unity Catalog)
# MAGIC so the model serving endpoint can do online feature lookups by `customer_id` or
# MAGIC `application_id`.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when

# COMMAND ----------

# MAGIC %md ## Build the joined gold view

# COMMAND ----------

apps = spark.table(table("silver_applications"))
kyc = spark.table(table("silver_kyc"))
# Drop duplicate join keys from secondary tables so the join doesn't produce ambiguous columns
bureau = spark.table(table("silver_bureau")).drop("customer_id", "bureau_pull_date")
txn = spark.table(table("silver_transactions"))
fraud = spark.table(table("silver_fraud_events")).drop("customer_id", "event_date", "event_id")
outcomes = (spark.table(table("raw_outcomes"))
            .drop("customer_id", "observed_at", "outcome_id", "p_default_true", "p_fraud_true"))

joined = (apps
    .join(kyc, "customer_id", "left")
    .join(bureau, "application_id", "left")
    .join(txn, "customer_id", "left")
    .join(fraud, "application_id", "left")
    .join(outcomes, "application_id", "left")
)

# COMMAND ----------

# MAGIC %md ## Engineered features

# COMMAND ----------

features = (joined
    .withColumn("dti_ratio", F.col("total_debt") / F.greatest(F.col("declared_income"), F.lit(1.0)))
    .withColumn("loan_to_income", F.col("requested_amount") / F.greatest(F.col("declared_income"), F.lit(1.0)))
    .withColumn("payment_to_income",
                (F.col("requested_amount") / F.col("term_months")) / F.greatest(F.col("declared_income") / 12.0, F.lit(1.0)))
    .withColumn("net_monthly_cashflow", F.col("avg_monthly_deposits") - F.col("avg_monthly_withdrawals"))
    .withColumn("repayment_quality",
                F.col("on_time_repayments") / F.greatest(F.col("on_time_repayments") + F.col("late_repayments"), F.lit(1)))
    .withColumn("verification_score",
                (F.col("id_verified").cast("int")
                 + F.col("address_verified").cast("int")
                 + F.col("phone_verified").cast("int")
                 + F.col("device_fingerprint_match").cast("int")
                 + F.col("ip_country_match").cast("int")) / 5.0)
    .withColumn("is_thin_file", F.when(F.col("months_credit_history") < 24, 1).otherwise(0))
)

# COMMAND ----------

# MAGIC %md ## Training tables

# COMMAND ----------

base_cols = [
    "application_id", "customer_id", "application_date",
    "requested_amount", "term_months", "purpose", "channel", "declared_income",
    "age", "country", "city", "years_at_address", "employment_status", "income_band",
    "id_verified", "address_verified", "kyc_risk_score",
    "bureau_score", "credit_utilization", "inquiries_6m", "tradelines_open",
    "delinquencies_24m", "months_credit_history", "has_bankruptcy", "total_debt",
    "months_active", "avg_monthly_deposits", "avg_monthly_withdrawals", "avg_balance",
    "nsf_count_12m", "on_time_repayments", "late_repayments",
    "device_fingerprint_match", "ip_country_match", "velocity_24h", "email_age_days",
    "phone_verified", "watchlist_hit", "synthetic_id_score",
    "dti_ratio", "loan_to_income", "payment_to_income", "net_monthly_cashflow",
    "repayment_quality", "verification_score", "is_thin_file",
]

# Credit training set — only labeled, non-fraud, decisioned rows
credit_train = (features
    .filter("outcome_status in ('paid_or_current', 'defaulted')")
    .select(*base_cols, F.col("defaulted").alias("label"))
)
(credit_train.write.format("delta")
    .option("overwriteSchema", "true")
    .mode("overwrite")
    .saveAsTable(table("training_credit")))
print(f"credit training rows: {credit_train.count()}")

# Fraud training set — all decisioned rows
fraud_train = (features
    .filter("outcome_status != 'in_progress'")
    .select(*base_cols, F.col("is_fraud").alias("label"))
)
(fraud_train.write.format("delta")
    .option("overwriteSchema", "true")
    .mode("overwrite")
    .saveAsTable(table("training_fraud")))
print(f"fraud training rows: {fraud_train.count()}")

# COMMAND ----------

# MAGIC %md ## Register UC Feature Tables for online lookup at serving time

# COMMAND ----------

try:
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()

    # Application-level features (keyed by application_id)
    app_feature_cols = [c for c in base_cols if c not in ("customer_id", "application_date")]
    app_features = features.select(*app_feature_cols).dropDuplicates(["application_id"])

    fe_app_table = table("feature_application")
    try:
        fe.create_table(
            name=fe_app_table,
            primary_keys=["application_id"],
            df=app_features,
            description="Application-level features for credit + fraud models",
        )
        print(f"created feature table {fe_app_table}")
    except Exception as e:
        # Already exists — write instead
        print(f"feature table exists, writing: {e}")
        fe.write_table(name=fe_app_table, df=app_features, mode="merge")

    print("feature tables registered")
except ImportError as e:
    print(f"skipping UC feature table registration (databricks-feature-engineering not on cluster): {e}")

# COMMAND ----------

dbutils.notebook.exit("features_ok")
