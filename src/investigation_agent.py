"""Deterministic, evidence-grounded fraud investigation assistant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.fraud_tools import FraudTools


class InvestigationAgent:
    """A transparent tool-driven workflow; it never performs financial actions."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.tools = FraudTools(project_root)

    @staticmethod
    def _identify_signals(transaction: dict[str, Any], history: dict[str, Any]) -> list[str]:
        signals: list[str] = []
        ratio = float(transaction["amount_vs_customer_average"] or 1.0)
        if ratio >= 3:
            signals.append(
                f"Amount ${transaction['amount']:,.2f} is {ratio:.1f}x the customer's prior average."
            )
        if transaction["is_international"]:
            prior_rate = float(history["historical_international_rate"])
            signals.append(
                f"International activity is unusual relative to the {prior_rate:.1%} prior international rate."
            )
        if transaction["merchant_country"] not in history["known_countries"] and history["transaction_count"] > 0:
            signals.append(
                f"Merchant country {transaction['merchant_country']} is not present in prior customer history."
            )
        if transaction["new_device_indicator"]:
            signals.append("The device was not seen in the customer's prior transactions.")
        if int(transaction["transactions_previous_1h"] or 0) >= 2:
            signals.append(
                f"There were {int(transaction['transactions_previous_1h'])} earlier transactions in the previous hour."
            )
        if transaction["unusual_hour_indicator"]:
            signals.append(
                f"The transaction occurred at {int(transaction['transaction_hour']):02d}:00, an overnight hour."
            )
        if transaction["new_category_indicator"]:
            signals.append(
                f"Merchant category '{transaction['merchant_category']}' was not present in prior history."
            )
        return signals

    def investigate(self, transaction_id: str) -> dict[str, Any]:
        """Run the fixed investigation sequence and return a structured case summary."""
        trace: list[str] = []
        transaction = self.tools.get_transaction(transaction_id)
        trace.append("get_transaction")
        history = self.tools.get_customer_history(
            transaction["customer_id"], transaction["transaction_timestamp"]
        )
        trace.append("get_customer_history")
        signals = self._identify_signals(transaction, history)
        query_terms: list[str] = []
        if float(transaction["amount_vs_customer_average"] or 1.0) >= 3:
            query_terms.append("unusual transaction amount historical spending")
        if transaction["is_international"]:
            query_terms.append("international activity geographic anomaly")
        if transaction["merchant_country"] not in history["known_countries"] and history["transaction_count"] > 0:
            query_terms.append("new unexpected country geography")
        if transaction["new_device_indicator"]:
            query_terms.append("new device account takeover")
        if int(transaction["transactions_previous_1h"] or 0) >= 2:
            query_terms.append("high transaction velocity rapid activity")
        if transaction["unusual_hour_indicator"]:
            query_terms.append("unusual transaction hours overnight")
        if transaction["new_category_indicator"]:
            query_terms.append("merchant category change")
        query = " ".join(query_terms) if query_terms else "model alert analyst review evidence quality"
        policies = self.tools.search_fraud_policy(query, top_k=3)
        trace.append("search_fraud_policy")
        risk = self.tools.get_risk_score(transaction_id)
        trace.append("get_risk_score")

        probability = float(risk["fraud_probability"])
        threshold = float(risk["selected_threshold"])
        if probability >= threshold and len(signals) >= 3:
            recommendation = "ESCALATE"
            reason = "The score exceeds the review threshold and multiple independent signals justify prompt analyst review."
        elif probability >= threshold * 0.60 or len(signals) >= 2:
            recommendation = "REVIEW"
            reason = "The available evidence warrants analyst review, but is not sufficient for a high-confidence escalation."
        else:
            recommendation = "CLOSE"
            reason = "The current score and evidence do not meet the demonstration review criteria."

        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(probability, 4),
            "selected_threshold": round(threshold, 4),
            "risk_level": risk["risk_level"],
            "signals": signals,
            "retrieved_policies": [
                {"section": p["section"], "citation": p["citation"], "similarity": p["similarity"]}
                for p in policies
            ],
            "recommendation": recommendation,
            "reason": reason,
            "requires_human_review": recommendation in {"REVIEW", "ESCALATE"},
            "tool_trace": trace,
            "safety_note": "Recommendation only; a human analyst controls all high-impact actions.",
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Investigate one scored transaction")
    parser.add_argument("transaction_id")
    args = parser.parse_args()
    print(json.dumps(InvestigationAgent().investigate(args.transaction_id), indent=2))


if __name__ == "__main__":
    main()
