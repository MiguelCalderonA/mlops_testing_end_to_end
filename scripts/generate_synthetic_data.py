"""
Synthetic data generator for the Lending Fintech MLOps demo.

Produces 6 CSV files with N rows each (default: 1000). The data is correlated:
KYC -> applications -> bureau / transactions / fraud -> outcomes. This means a
real model will actually find signal.

Usage (local):
    python scripts/generate_synthetic_data.py --out ./data --rows 1000

Usage (on Databricks, e.g. from a notebook):
    from scripts.generate_synthetic_data import generate_all
    tables = generate_all(rows=1000)
    for name, df in tables.items():
        spark.createDataFrame(df).write.mode("overwrite").saveAsTable(...)
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_kyc(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    first = ["Alex", "Maria", "Carlos", "Sofia", "James", "Linda", "Wei", "Amir",
             "Priya", "Diego", "Anna", "Yuki", "Omar", "Chen", "Olivia", "Lucas"]
    last = ["Garcia", "Smith", "Lopez", "Chen", "Patel", "Khan", "Brown", "Nguyen",
            "Lee", "Kim", "Martinez", "Rossi", "Johnson", "Silva", "Kumar", "Davis"]
    cities = ["Austin", "Madrid", "Mexico City", "Berlin", "Paris", "Mumbai",
              "Sao Paulo", "London", "Tokyo", "Toronto", "Buenos Aires", "Seoul"]
    countries = ["US", "ES", "MX", "DE", "FR", "IN", "BR", "GB", "JP", "CA", "AR", "KR"]

    customer_id = [f"C{1000000 + i}" for i in range(rows)]
    age = rng.integers(21, 75, size=rows)
    years_address = rng.integers(0, 30, size=rows)
    employment_status = rng.choice(
        ["employed", "self_employed", "unemployed", "retired", "student"],
        size=rows, p=[0.62, 0.18, 0.06, 0.10, 0.04],
    )
    income_band = rng.choice(["<25k", "25-50k", "50-100k", "100-200k", ">200k"],
                             size=rows, p=[0.15, 0.30, 0.35, 0.15, 0.05])
    id_verified = rng.choice([True, False], size=rows, p=[0.92, 0.08])
    address_verified = rng.choice([True, False], size=rows, p=[0.88, 0.12])
    kyc_risk_score = np.clip(rng.normal(30, 15, size=rows), 0, 100).round(1)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "first_name": rng.choice(first, size=rows),
        "last_name": rng.choice(last, size=rows),
        "age": age,
        "country": rng.choice(countries, size=rows),
        "city": rng.choice(cities, size=rows),
        "years_at_address": years_address,
        "employment_status": employment_status,
        "income_band": income_band,
        "id_verified": id_verified,
        "address_verified": address_verified,
        "kyc_risk_score": kyc_risk_score,
        "kyc_created_at": _random_dates(rng, rows, days_back=900),
    })
    return df


def gen_applications(kyc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = len(kyc)
    application_id = [f"A{2000000 + i}" for i in range(rows)]
    requested_amount = np.round(rng.lognormal(mean=8.5, sigma=0.6, size=rows), 2)
    requested_amount = np.clip(requested_amount, 500, 75000)
    term_months = rng.choice([6, 12, 18, 24, 36, 48, 60], size=rows,
                             p=[0.05, 0.20, 0.15, 0.30, 0.20, 0.07, 0.03])
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "auto", "small_business", "education", "other"],
        size=rows, p=[0.35, 0.18, 0.10, 0.12, 0.10, 0.08, 0.07],
    )
    channel = rng.choice(["web", "mobile_app", "broker", "branch"], size=rows,
                        p=[0.45, 0.40, 0.10, 0.05])
    # Declared income loosely correlated with KYC income_band
    band_map = {"<25k": 18000, "25-50k": 35000, "50-100k": 72000,
                "100-200k": 140000, ">200k": 260000}
    base_income = kyc["income_band"].map(band_map).to_numpy()
    declared_income = np.round(base_income * rng.normal(1.0, 0.18, size=rows), 0)
    declared_income = np.clip(declared_income, 8000, 600000)

    application_date = _random_dates(rng, rows, days_back=540)

    df = pd.DataFrame({
        "application_id": application_id,
        "customer_id": kyc["customer_id"].to_numpy(),
        "application_date": application_date,
        "requested_amount": requested_amount,
        "term_months": term_months,
        "purpose": purpose,
        "channel": channel,
        "declared_income": declared_income,
    })
    return df


def gen_bureau(apps: pd.DataFrame, kyc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = len(apps)
    # FICO-style score correlated inversely with kyc_risk_score
    kyc_score = kyc.set_index("customer_id").loc[apps["customer_id"], "kyc_risk_score"].to_numpy()
    base = 720 - (kyc_score - 30) * 2.2
    bureau_score = np.clip(base + rng.normal(0, 35, size=rows), 300, 850).astype(int)

    utilization = np.clip(rng.beta(2.0, 4.5, size=rows), 0, 1).round(3)
    inquiries_6m = rng.poisson(1.6, size=rows)
    tradelines_open = rng.poisson(5.5, size=rows)
    delinquencies_24m = rng.poisson(0.4, size=rows)
    months_credit_history = rng.integers(6, 360, size=rows)
    has_bankruptcy = rng.choice([0, 1], size=rows, p=[0.97, 0.03])
    total_debt = np.round(rng.lognormal(9.2, 0.9, size=rows), 0)

    df = pd.DataFrame({
        "application_id": apps["application_id"].to_numpy(),
        "customer_id": apps["customer_id"].to_numpy(),
        "bureau_pull_date": apps["application_date"].to_numpy(),
        "bureau_score": bureau_score,
        "credit_utilization": utilization,
        "inquiries_6m": inquiries_6m,
        "tradelines_open": tradelines_open,
        "delinquencies_24m": delinquencies_24m,
        "months_credit_history": months_credit_history,
        "has_bankruptcy": has_bankruptcy,
        "total_debt": total_debt,
    })
    return df


def gen_transactions(kyc: pd.DataFrame, apps: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    One aggregated row per customer summarizing internal banking history.
    (Real systems store per-transaction rows; we keep one row to fit the 1000-row target.)
    """
    rows = len(kyc)
    avg_monthly_deposits = np.round(np.maximum(rng.normal(3500, 1800, size=rows), 0), 2)
    avg_monthly_withdrawals = np.round(np.maximum(rng.normal(3100, 1500, size=rows), 0), 2)
    nsf_count_12m = rng.poisson(0.6, size=rows)
    on_time_repayments = rng.integers(0, 24, size=rows)
    late_repayments = rng.poisson(1.1, size=rows)
    months_active = rng.integers(1, 60, size=rows)
    avg_balance = np.round(np.maximum(rng.normal(2200, 1700, size=rows), 0), 2)

    df = pd.DataFrame({
        "customer_id": kyc["customer_id"].to_numpy(),
        "as_of_date": _random_dates(rng, rows, days_back=30),
        "months_active": months_active,
        "avg_monthly_deposits": avg_monthly_deposits,
        "avg_monthly_withdrawals": avg_monthly_withdrawals,
        "avg_balance": avg_balance,
        "nsf_count_12m": nsf_count_12m,
        "on_time_repayments": on_time_repayments,
        "late_repayments": late_repayments,
    })
    return df


def gen_fraud_events(apps: pd.DataFrame, kyc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = len(apps)
    device_match = rng.choice([True, False], size=rows, p=[0.85, 0.15])
    ip_country_match = rng.choice([True, False], size=rows, p=[0.90, 0.10])
    velocity_24h = rng.poisson(0.3, size=rows)
    email_age_days = rng.integers(1, 4000, size=rows)
    phone_verified = rng.choice([True, False], size=rows, p=[0.93, 0.07])
    watchlist_hit = rng.choice([0, 1], size=rows, p=[0.985, 0.015])
    synthetic_id_score = np.clip(rng.beta(1.2, 8.0, size=rows), 0, 1).round(3)

    df = pd.DataFrame({
        "event_id": [f"E{3000000 + i}" for i in range(rows)],
        "application_id": apps["application_id"].to_numpy(),
        "customer_id": apps["customer_id"].to_numpy(),
        "event_date": apps["application_date"].to_numpy(),
        "device_fingerprint_match": device_match,
        "ip_country_match": ip_country_match,
        "velocity_24h": velocity_24h,
        "email_age_days": email_age_days,
        "phone_verified": phone_verified,
        "watchlist_hit": watchlist_hit,
        "synthetic_id_score": synthetic_id_score,
    })
    return df


def gen_outcomes(apps: pd.DataFrame, bureau: pd.DataFrame, fraud: pd.DataFrame,
                 txn: pd.DataFrame, kyc: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate labels with realistic signal:
      - default risk increases with low bureau score, high utilization, delinquencies, NSFs
      - fraud increases with watchlist_hit, low email age, ip mismatch, synthetic_id_score
    """
    rows = len(apps)

    # Default probability — logistic function of risk features
    b = bureau.set_index("application_id").loc[apps["application_id"]]
    t = txn.set_index("customer_id").loc[apps["customer_id"]]
    risk_logit = (
        -2.0
        + (700 - b["bureau_score"].to_numpy()) * 0.012
        + b["credit_utilization"].to_numpy() * 1.8
        + b["delinquencies_24m"].to_numpy() * 0.35
        + b["has_bankruptcy"].to_numpy() * 0.9
        + t["nsf_count_12m"].to_numpy() * 0.20
        + t["late_repayments"].to_numpy() * 0.15
        - np.minimum(t["on_time_repayments"].to_numpy() / 24, 1.0) * 0.6
    )
    p_default = 1 / (1 + np.exp(-risk_logit))
    defaulted = (rng.random(rows) < p_default).astype(int)

    # Fraud probability
    f = fraud.set_index("application_id").loc[apps["application_id"]]
    fraud_logit = (
        -4.0
        + f["watchlist_hit"].to_numpy() * 3.5
        + (1 - f["device_fingerprint_match"].astype(int).to_numpy()) * 1.2
        + (1 - f["ip_country_match"].astype(int).to_numpy()) * 1.4
        + (1 - f["phone_verified"].astype(int).to_numpy()) * 1.1
        + (f["email_age_days"].to_numpy() < 30).astype(int) * 1.6
        + f["synthetic_id_score"].to_numpy() * 3.0
        + f["velocity_24h"].to_numpy() * 0.5
    )
    p_fraud = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(rows) < p_fraud).astype(int)

    # Outcome status only known for some apps; the rest are "in_progress"
    decisioned = rng.random(rows) < 0.80
    status = np.where(decisioned,
                      np.where(is_fraud == 1, "fraud_chargeback",
                               np.where(defaulted == 1, "defaulted", "paid_or_current")),
                      "in_progress")
    observed_at = pd.to_datetime(apps["application_date"]) + pd.to_timedelta(
        rng.integers(60, 540, size=rows), unit="D"
    )

    df = pd.DataFrame({
        "outcome_id": [f"O{4000000 + i}" for i in range(rows)],
        "application_id": apps["application_id"].to_numpy(),
        "customer_id": apps["customer_id"].to_numpy(),
        "observed_at": observed_at,
        "defaulted": defaulted,
        "is_fraud": is_fraud,
        "outcome_status": status,
        "p_default_true": np.round(p_default, 4),  # kept for audit / debug
        "p_fraud_true": np.round(p_fraud, 4),
    })
    return df


def _random_dates(rng: np.random.Generator, n: int, days_back: int) -> pd.Series:
    base = datetime.now() - timedelta(days=days_back)
    offsets = rng.integers(0, days_back, size=n)
    return pd.to_datetime([base + timedelta(days=int(d)) for d in offsets])


def generate_all(rows: int = 1000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = _rng(seed)
    kyc = gen_kyc(rows, rng)
    apps = gen_applications(kyc, rng)
    bureau = gen_bureau(apps, kyc, rng)
    txn = gen_transactions(kyc, apps, rng)
    fraud = gen_fraud_events(apps, kyc, rng)
    outcomes = gen_outcomes(apps, bureau, fraud, txn, kyc, rng)
    return {
        "kyc_identity": kyc,
        "applications": apps,
        "credit_bureau": bureau,
        "internal_transactions": txn,
        "fraud_events": fraud,
        "historical_outcomes": outcomes,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./data/synthetic", help="Output directory for CSVs")
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = generate_all(rows=args.rows, seed=args.seed)
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {len(df):>5} rows -> {path}")


if __name__ == "__main__":
    main()
