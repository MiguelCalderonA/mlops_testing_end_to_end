"""Validate conf/config.yaml is well-formed."""
from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"


def test_config_loads():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in [
        "catalog", "schema", "volume", "tables",
        "credit_model_name", "fraud_model_name",
        "credit_endpoint_name", "fraud_endpoint_name",
        "synthetic_rows", "random_seed",
    ]:
        assert key in cfg, f"missing key {key}"


def test_table_keys_referenced_by_notebooks_exist():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    expected = {
        "raw_applications", "raw_kyc", "raw_bureau",
        "raw_transactions", "raw_fraud_events", "raw_outcomes",
        "silver_applications", "silver_kyc", "silver_bureau",
        "silver_transactions", "silver_fraud_events",
        "feature_application", "feature_customer", "feature_bureau",
        "training_credit", "training_fraud",
        "inference_credit", "inference_fraud",
    }
    assert expected.issubset(cfg["tables"].keys())
