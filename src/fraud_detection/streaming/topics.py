"""Versioned Kafka topic registry."""

TRANSACTIONS_TOPIC = "transactions.v1"
PREDICTIONS_TOPIC = "fraud_predictions.v1"
ALERTS_TOPIC = "fraud_alerts.v1"
LABELS_TOPIC = "fraud_labels.v1"
DEAD_LETTER_TOPIC = "dead_letter_transactions.v1"

ALL_TOPICS = [
    TRANSACTIONS_TOPIC,
    PREDICTIONS_TOPIC,
    ALERTS_TOPIC,
    LABELS_TOPIC,
    DEAD_LETTER_TOPIC,
]
