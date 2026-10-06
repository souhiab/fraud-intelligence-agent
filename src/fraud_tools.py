"""Small, readable tools used by the deterministic investigation workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FraudTools:
    """Data and policy lookup tools. Data loads lazily on first use."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.data_dir = self.root / "data" / "sample"
        self._transactions: pd.DataFrame | None = None
        self._customers: pd.DataFrame | None = None
        self._scores: pd.DataFrame | None = None
        self._policy_sections: list[dict[str, str]] | None = None
        self._policy_matrix = None
        self._vectorizer: TfidfVectorizer | None = None

    def _load_data(self) -> None:
        if self._transactions is None:
            self._transactions = pd.read_csv(
                self.data_dir / "transactions.csv", parse_dates=["transaction_timestamp"]
            )
            self._customers = pd.read_csv(self.data_dir / "customers.csv")
            self._scores = pd.read_csv(
                self.data_dir / "scored_test_transactions.csv",
                parse_dates=["transaction_timestamp"],
            )

    def _load_policy(self) -> None:
        if self._policy_sections is not None:
            return
        text = (self.root / "policies" / "fraud_policy.md").read_text(encoding="utf-8")
        parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
        self._policy_sections = []
        for part in parts:
            title, body = part.split("\n", 1)
            self._policy_sections.append({"section": title.strip(), "text": body.strip()})
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [f"{s['section']} {s['text']}" for s in self._policy_sections]
        self._policy_matrix = self._vectorizer.fit_transform(corpus)

    def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """Return an alert's transaction details, model score, and model features."""
        self._load_data()
        assert self._scores is not None
        match = self._scores.loc[self._scores["transaction_id"].eq(transaction_id)]
        if match.empty:
            raise KeyError(f"No scored test transaction found for {transaction_id}")
        row = match.iloc[0]
        fields = [
            "transaction_id", "customer_id", "transaction_timestamp", "amount",
            "merchant_category", "merchant_country", "transaction_channel", "device_id",
            "is_card_present", "is_international", "transaction_hour",
            "previous_transaction_count", "avg_previous_amount",
            "amount_vs_customer_average", "minutes_since_previous_transaction",
            "transactions_previous_1h", "transactions_previous_24h",
            "historical_international_rate", "new_device_indicator",
            "new_category_indicator", "unusual_hour_indicator", "fraud_probability",
            "selected_threshold", "risk_level",
        ]
        result = {field: row[field] for field in fields}
        result["transaction_timestamp"] = pd.Timestamp(result["transaction_timestamp"]).isoformat()
        for key, value in list(result.items()):
            if pd.isna(value):
                result[key] = None
            elif hasattr(value, "item"):
                result[key] = value.item()
        return result

    def get_customer_history(
        self, customer_id: str, before_timestamp: str | pd.Timestamp, limit: int = 8
    ) -> dict[str, Any]:
        """Return history strictly before the alert timestamp."""
        self._load_data()
        assert self._transactions is not None
        before = pd.Timestamp(before_timestamp)
        history = self._transactions.loc[
            self._transactions["customer_id"].eq(customer_id)
            & self._transactions["transaction_timestamp"].lt(before)
        ].sort_values("transaction_timestamp")
        recent = history.tail(limit).copy()
        previous_hour = history.loc[
            history["transaction_timestamp"].ge(before - pd.Timedelta(hours=1))
        ]
        previous_day = history.loc[
            history["transaction_timestamp"].ge(before - pd.Timedelta(hours=24))
        ]
        rows = recent[
            ["transaction_id", "transaction_timestamp", "amount", "merchant_country",
             "merchant_category", "transaction_channel", "device_id"]
        ].copy()
        rows["transaction_timestamp"] = rows["transaction_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "customer_id": customer_id,
            "history_cutoff": before.isoformat(),
            "transaction_count": int(len(history)),
            "historical_average_amount": round(float(history["amount"].mean()), 2) if len(history) else None,
            "historical_max_amount": round(float(history["amount"].max()), 2) if len(history) else None,
            "historical_international_rate": round(float(history["is_international"].mean()), 4) if len(history) else 0.0,
            "known_countries": sorted(history["merchant_country"].dropna().unique().tolist()),
            "known_devices": int(history["device_id"].nunique()),
            "merchant_categories": sorted(history["merchant_category"].dropna().unique().tolist()),
            "transactions_previous_1h": int(len(previous_hour)),
            "transactions_previous_24h": int(len(previous_day)),
            "recent_transactions": rows.to_dict(orient="records"),
        }

    def search_fraud_policy(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve the most relevant synthetic policy sections with TF-IDF."""
        self._load_policy()
        assert self._vectorizer is not None and self._policy_sections is not None
        scores = cosine_similarity(self._vectorizer.transform([query]), self._policy_matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [
            {
                "section": self._policy_sections[i]["section"],
                "text": self._policy_sections[i]["text"],
                "similarity": round(float(scores[i]), 3),
                "citation": f"[Policy: {self._policy_sections[i]['section']}]",
            }
            for i in order
        ]

    def get_risk_score(self, transaction_id: str) -> dict[str, Any]:
        """Return the model probability, selected threshold, and display risk level."""
        transaction = self.get_transaction(transaction_id)
        return {
            "transaction_id": transaction_id,
            "fraud_probability": transaction["fraud_probability"],
            "selected_threshold": transaction["selected_threshold"],
            "risk_level": transaction["risk_level"],
        }

    def get_model_metadata(self) -> dict[str, Any]:
        with (self.data_dir / "model_metadata.json").open(encoding="utf-8") as handle:
            return json.load(handle)
