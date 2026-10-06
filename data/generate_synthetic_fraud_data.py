"""Generate deterministic, non-trivial synthetic fraud data for the case study."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 10_000
N_TRANSACTIONS = 100_000
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31 23:59:59")

COUNTRIES = np.array(["US", "GB", "FR", "DE", "ES", "MA", "CA", "BR"])
COUNTRY_WEIGHTS = np.array([0.34, 0.13, 0.12, 0.11, 0.08, 0.08, 0.08, 0.06])
CATEGORIES = np.array(
    ["grocery", "dining", "fuel", "retail", "travel", "electronics", "digital_goods", "cash_withdrawal"]
)
CATEGORY_WEIGHTS = np.array([0.24, 0.18, 0.10, 0.17, 0.07, 0.08, 0.09, 0.07])
CATEGORY_BASE = {
    "grocery": 42, "dining": 34, "fuel": 49, "retail": 78,
    "travel": 310, "electronics": 220, "digital_goods": 28, "cash_withdrawal": 135,
}


def _other_country(rng: np.random.Generator, home: str) -> str:
    choices = COUNTRIES[COUNTRIES != home]
    return str(rng.choice(choices))


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    customer_ids = [f"CUST_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
    signup_dates = START_DATE - pd.to_timedelta(rng.integers(30, 1_825, N_CUSTOMERS), unit="D")
    segments = rng.choice(["standard", "premium", "student"], N_CUSTOMERS, p=[0.70, 0.18, 0.12])
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates.strftime("%Y-%m-%d"),
            "home_country": rng.choice(COUNTRIES, N_CUSTOMERS, p=COUNTRY_WEIGHTS),
            "customer_age": np.clip(rng.normal(42, 14, N_CUSTOMERS).round(), 18, 85).astype(int),
            "account_tenure_days": (START_DATE - signup_dates).days,
            "customer_segment": segments,
        }
    )
    return customers


def generate_transactions(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # A mild activity skew gives realistic differences in customer history depth.
    activity = rng.lognormal(mean=0.0, sigma=0.65, size=N_CUSTOMERS)
    activity /= activity.sum()
    customer_idx = rng.choice(N_CUSTOMERS, size=N_TRANSACTIONS, p=activity)
    customer_ids = customers["customer_id"].to_numpy()[customer_idx]
    home_countries = customers["home_country"].to_numpy()[customer_idx]
    segments = customers["customer_segment"].to_numpy()[customer_idx]

    seconds = rng.integers(0, int((END_DATE - START_DATE).total_seconds()), N_TRANSACTIONS)
    timestamps = START_DATE + pd.to_timedelta(seconds, unit="s")
    categories = rng.choice(CATEGORIES, N_TRANSACTIONS, p=CATEGORY_WEIGHTS)
    base_amount = np.array([CATEGORY_BASE[c] for c in categories], dtype=float)
    segment_multiplier = np.select(
        [segments == "premium", segments == "student"], [1.55, 0.70], default=1.0
    )
    customer_spend_factor = rng.lognormal(0.0, 0.30, N_CUSTOMERS)[customer_idx]
    amounts = rng.lognormal(np.log(base_amount * segment_multiplier * customer_spend_factor), 0.62)

    # Risk scenarios affect several—but not always all—observable signals.
    risky_scenario = rng.random(N_TRANSACTIONS) < 0.045
    high_amount = risky_scenario & (rng.random(N_TRANSACTIONS) < 0.38)
    amounts[high_amount] *= rng.uniform(3.0, 8.0, high_amount.sum())

    international = rng.random(N_TRANSACTIONS) < (0.065 + 0.22 * risky_scenario)
    merchant_countries = home_countries.copy()
    for i in np.flatnonzero(international):
        merchant_countries[i] = _other_country(rng, home_countries[i])

    primary_devices = np.array([f"DEV_{i:05d}_A" for i in range(1, N_CUSTOMERS + 1)])[customer_idx]
    new_device_signal = rng.random(N_TRANSACTIONS) < (0.06 + 0.25 * risky_scenario)
    devices = primary_devices.copy()
    random_device_ids = rng.integers(1, 50_000_000, new_device_signal.sum())
    devices[new_device_signal] = np.array([f"DEV_X{x:08d}" for x in random_device_ids])

    channels = rng.choice(["pos", "online", "mobile", "atm"], N_TRANSACTIONS, p=[0.46, 0.28, 0.19, 0.07])
    force_online = risky_scenario & (rng.random(N_TRANSACTIONS) < 0.32)
    channels[force_online] = "online"

    # Move a subset of risky events to overnight hours; this is a weak signal, not a label proxy.
    unusual_time = risky_scenario & (rng.random(N_TRANSACTIONS) < 0.25)
    timestamps = pd.Series(timestamps)
    timestamps.loc[unusual_time] = (
        timestamps.loc[unusual_time].dt.normalize()
        + pd.to_timedelta(rng.choice([0, 1, 2, 3, 4], unusual_time.sum()), unit="h")
        + pd.to_timedelta(rng.integers(0, 3_600, unusual_time.sum()), unit="s")
    )

    transactions = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "transaction_timestamp": timestamps,
            "amount": np.round(np.maximum(amounts, 1.0), 2),
            "merchant_category": categories,
            "merchant_country": merchant_countries,
            "transaction_channel": channels,
            "device_id": devices,
        }
    ).sort_values(["customer_id", "transaction_timestamp"]).reset_index(drop=True)

    # Create modest velocity bursts by moving selected events close to the preceding event.
    can_burst = transactions.groupby("customer_id").cumcount() > 0
    burst_mask = can_burst & (rng.random(N_TRANSACTIONS) < 0.022)
    previous_time = transactions.groupby("customer_id")["transaction_timestamp"].shift(1)
    transactions.loc[burst_mask, "transaction_timestamp"] = (
        previous_time.loc[burst_mask]
        + pd.to_timedelta(rng.integers(30, 2_400, burst_mask.sum()), unit="s")
    )
    transactions = transactions.sort_values(["customer_id", "transaction_timestamp"]).reset_index(drop=True)

    prior_count = transactions.groupby("customer_id").cumcount()
    prior_mean = (
        transactions.groupby("customer_id")["amount"].cumsum() - transactions["amount"]
    ) / prior_count.replace(0, np.nan)
    amount_ratio = transactions["amount"] / prior_mean.clip(lower=10).fillna(transactions["amount"].median())
    prior_device_count = transactions.groupby(["customer_id", "device_id"]).cumcount()
    previous_time = transactions.groupby("customer_id")["transaction_timestamp"].shift(1)
    minutes_since_previous = (
        transactions["transaction_timestamp"] - previous_time
    ).dt.total_seconds().div(60)
    unusual_hour = transactions["transaction_timestamp"].dt.hour.isin([0, 1, 2, 3, 4])
    is_international = transactions["merchant_country"].ne(
        transactions["customer_id"].map(customers.set_index("customer_id")["home_country"])
    )
    is_new_device = prior_device_count.eq(0) & prior_count.gt(0)
    rapid_repeat = minutes_since_previous.lt(30)
    online = transactions["transaction_channel"].isin(["online", "mobile"])

    # Labels are sampled from a noisy probability: patterns matter, but certainty is impossible.
    hidden_customer_risk = rng.normal(0, 0.55, N_CUSTOMERS)[
        transactions["customer_id"].str[-5:].astype(int).to_numpy() - 1
    ]
    international_new_device = is_international & is_new_device
    high_amount_velocity = amount_ratio.ge(3) & rapid_repeat.fillna(False)
    overnight_new_device = unusual_hour & is_new_device
    risk_score = (
        -7.35
        + 1.15 * np.log1p(amount_ratio.clip(0, 20))
        + 0.95 * is_international.astype(float)
        + 1.20 * is_new_device.astype(float)
        + 0.95 * unusual_hour.astype(float)
        + 1.30 * rapid_repeat.fillna(False).astype(float)
        + 0.35 * online.astype(float)
        + 1.00 * international_new_device.astype(float)
        + 1.10 * high_amount_velocity.astype(float)
        + 0.80 * overnight_new_device.astype(float)
        + 0.45 * hidden_customer_risk
        + rng.normal(0, 0.35, N_TRANSACTIONS)
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-risk_score))
    transactions["is_fraud"] = rng.binomial(1, fraud_probability)

    transactions["transaction_id"] = [f"TX_{i:06d}" for i in range(1, N_TRANSACTIONS + 1)]
    transactions["is_card_present"] = transactions["transaction_channel"].eq("pos").astype(int)
    transactions["is_international"] = is_international.astype(int)
    transactions["transaction_status"] = rng.choice(
        ["approved", "declined", "reversed"], N_TRANSACTIONS, p=[0.965, 0.025, 0.010]
    )
    columns = [
        "transaction_id", "customer_id", "transaction_timestamp", "amount",
        "merchant_category", "merchant_country", "transaction_channel", "device_id",
        "is_card_present", "is_international", "transaction_status", "is_fraud",
    ]
    result = transactions[columns].sort_values("transaction_timestamp").reset_index(drop=True)
    result["transaction_timestamp"] = result["transaction_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return result


def validate(customers: pd.DataFrame, transactions: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(transactions["transaction_timestamp"])
    assert customers["customer_id"].is_unique
    assert transactions["transaction_id"].is_unique
    assert set(transactions["customer_id"]).issubset(set(customers["customer_id"]))
    assert transactions["amount"].ge(0).all()
    assert timestamps.between(START_DATE, END_DATE).all()
    assert timestamps.max() < pd.Timestamp.now()
    assert set(transactions["is_fraud"].unique()).issubset({0, 1})
    forbidden = {"is_suspicious", "fraud_probability", "risk_score"}
    assert not forbidden.intersection(transactions.columns)


def main() -> None:
    rng = np.random.default_rng(SEED)
    customers = generate_customers(rng)
    transactions = generate_transactions(customers, rng)
    validate(customers, transactions)
    output_dir = Path(__file__).resolve().parent / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(output_dir / "customers.csv", index=False)
    transactions.to_csv(output_dir / "transactions.csv", index=False)
    print(f"Customers: {len(customers):,}")
    print(f"Transactions: {len(transactions):,}")
    print(f"Date range: {transactions.transaction_timestamp.min()} to {transactions.transaction_timestamp.max()}")
    print(f"Fraud rate: {transactions.is_fraud.mean():.3%} ({transactions.is_fraud.sum():,} cases)")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
