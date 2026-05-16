"""Unit tests for the synthetic data generator. Run with `pytest`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_synthetic_data import generate_all  # noqa: E402


def test_row_counts() -> None:
    tables = generate_all(rows=200, seed=7)
    assert len(tables) == 6
    for name, df in tables.items():
        assert len(df) == 200, f"{name} has wrong row count"


def test_referential_integrity() -> None:
    tables = generate_all(rows=500, seed=11)
    customer_ids = set(tables["kyc_identity"]["customer_id"])
    application_ids = set(tables["applications"]["application_id"])

    assert set(tables["applications"]["customer_id"]).issubset(customer_ids)
    assert set(tables["credit_bureau"]["application_id"]) == application_ids
    assert set(tables["fraud_events"]["application_id"]) == application_ids
    assert set(tables["historical_outcomes"]["application_id"]) == application_ids


def test_labels_have_signal() -> None:
    """Smoke check that the generator produces both positive and negative labels."""
    tables = generate_all(rows=1000, seed=42)
    out = tables["historical_outcomes"]
    assert out["defaulted"].sum() > 5
    assert out["is_fraud"].sum() > 2
    assert out["defaulted"].sum() < 0.6 * len(out)


def test_no_unexpected_nulls() -> None:
    tables = generate_all(rows=200, seed=3)
    for name, df in tables.items():
        # Allow nulls only in outcomes for in-progress rows — none expected with synthetic gen
        assert df.notna().all().all(), f"unexpected nulls in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
