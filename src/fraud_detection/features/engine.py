"""Event-time feature engine used as the online reference implementation."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class HistoricalEvent:
    timestamp: datetime
    amount: float
    merchant_id: str
    latitude: float
    longitude: float


@dataclass
class CustomerState:
    events: deque[HistoricalEvent] = field(default_factory=deque)
    amount_30d: float = 0.0
    confirmed_labels: int = 0
    confirmed_fraud: int = 0


@dataclass
class MerchantState:
    events: deque[tuple[datetime, float]] = field(default_factory=deque)
    amount_30d: float = 0.0
    confirmed_labels: int = 0
    confirmed_fraud: int = 0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(max(1 - value, 0)))


class StreamingFeatureEngine:
    """Computes pre-event features and then updates state, preventing self-leakage."""

    def __init__(self, label_delay_days: int = 7) -> None:
        self.label_delay = timedelta(days=label_delay_days)
        self.customers: defaultdict[str, CustomerState] = defaultdict(CustomerState)
        self.merchants: defaultdict[str, MerchantState] = defaultdict(MerchantState)
        self.pending_labels: list[tuple[datetime, int, str, str, int]] = []
        self._sequence = 0

    def _apply_available_labels(self, timestamp: datetime) -> None:
        while self.pending_labels and self.pending_labels[0][0] <= timestamp:
            _, _, customer_id, merchant_id, label = heapq.heappop(self.pending_labels)
            customer = self.customers[customer_id]
            merchant = self.merchants[merchant_id]
            customer.confirmed_labels += 1
            merchant.confirmed_labels += 1
            customer.confirmed_fraud += label
            merchant.confirmed_fraud += label

    @staticmethod
    def _prune_customer(state: CustomerState, timestamp: datetime) -> None:
        cutoff = timestamp - timedelta(days=30)
        while state.events and state.events[0].timestamp < cutoff:
            state.amount_30d -= state.events.popleft().amount

    @staticmethod
    def _prune_merchant(state: MerchantState, timestamp: datetime) -> None:
        cutoff = timestamp - timedelta(days=30)
        while state.events and state.events[0][0] < cutoff:
            _, amount = state.events.popleft()
            state.amount_30d -= amount

    def process(self, event: dict[str, Any], label: int | None = None) -> dict[str, float]:
        timestamp = event["event_timestamp"]
        if not isinstance(timestamp, datetime):
            raise TypeError("event_timestamp must be a datetime")
        self._apply_available_labels(timestamp)
        customer = self.customers[event["customer_id"]]
        merchant = self.merchants[event["merchant_id"]]
        self._prune_customer(customer, timestamp)
        self._prune_merchant(merchant, timestamp)

        one_hour = timestamp - timedelta(hours=1)
        one_day = timestamp - timedelta(days=1)
        seven_days = timestamp - timedelta(days=7)
        five_minutes = timestamp - timedelta(minutes=5)
        thirty_minutes = timestamp - timedelta(minutes=30)
        recent_hour = [item for item in customer.events if item.timestamp >= one_hour]
        recent_day = [item for item in customer.events if item.timestamp >= one_day]
        recent_week = [item for item in customer.events if item.timestamp >= seven_days]
        previous = customer.events[-1] if customer.events else None
        amount = float(event["amount"])
        average_30d = customer.amount_30d / len(customer.events) if customer.events else amount
        average_7d = (
            sum(item.amount for item in recent_week) / len(recent_week) if recent_week else amount
        )
        time_since = (timestamp - previous.timestamp).total_seconds() if previous else 0.0
        distance_previous = (
            haversine_km(
                previous.latitude,
                previous.longitude,
                float(event["merchant_latitude"]),
                float(event["merchant_longitude"]),
            )
            if previous
            else 0.0
        )
        speed = distance_previous / (time_since / 3600) if time_since > 0 else 0.0
        home_latitude = event.get("customer_home_latitude")
        home_longitude = event.get("customer_home_longitude")
        home_distance = (
            haversine_km(
                float(home_latitude),
                float(home_longitude),
                float(event["merchant_latitude"]),
                float(event["merchant_longitude"]),
            )
            if home_latitude is not None and home_longitude is not None
            else 0.0
        )
        merchant_24h = [item for item in merchant.events if item[0] >= one_day]
        merchant_average = merchant.amount_30d / len(merchant.events) if merchant.events else amount
        hour = timestamp.hour + timestamp.minute / 60
        features = {
            "amount": amount,
            "log_amount": math.log1p(amount),
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "weekday_sin": math.sin(2 * math.pi * timestamp.weekday() / 7),
            "weekday_cos": math.cos(2 * math.pi * timestamp.weekday() / 7),
            "is_weekend": float(timestamp.weekday() >= 5),
            "home_merchant_distance_km": home_distance,
            "log_city_population": math.log1p(float(event.get("city_population") or 0)),
            "customer_avg_amount_7d": average_7d,
            "customer_avg_amount_30d": average_30d,
            "amount_vs_customer_avg_30d": amount / max(average_30d, 0.01),
            "customer_transactions_5m": float(
                sum(item.timestamp >= five_minutes for item in customer.events)
            ),
            "customer_transactions_30m": float(
                sum(item.timestamp >= thirty_minutes for item in customer.events)
            ),
            "customer_transactions_1h": float(len(recent_hour)),
            "customer_transactions_24h": float(len(recent_day)),
            "customer_amount_1h": sum(item.amount for item in recent_hour),
            "customer_unique_merchants_1h": float(len({item.merchant_id for item in recent_hour})),
            "time_since_previous_seconds": time_since,
            "distance_from_previous_km": distance_previous,
            "travel_speed_kmh": speed,
            "impossible_travel_flag": float(distance_previous >= 500 and speed >= 900),
            "customer_confirmed_fraud_rate": (customer.confirmed_fraud + 1)
            / (customer.confirmed_labels + 200),
            "merchant_transactions_24h": float(len(merchant_24h)),
            "merchant_avg_amount_30d": merchant_average,
            "merchant_confirmed_fraud_rate": (merchant.confirmed_fraud + 1)
            / (merchant.confirmed_labels + 200),
            "customer_cold_start": float(not customer.events),
            "merchant_cold_start": float(not merchant.events),
            "history_unavailable": 0.0,
        }

        historical = HistoricalEvent(
            timestamp=timestamp,
            amount=amount,
            merchant_id=event["merchant_id"],
            latitude=float(event["merchant_latitude"]),
            longitude=float(event["merchant_longitude"]),
        )
        customer.events.append(historical)
        customer.amount_30d += amount
        merchant.events.append((timestamp, amount))
        merchant.amount_30d += amount
        if label is not None:
            self._sequence += 1
            heapq.heappush(
                self.pending_labels,
                (
                    timestamp + self.label_delay,
                    self._sequence,
                    event["customer_id"],
                    event["merchant_id"],
                    int(label),
                ),
            )
        return features

    def apply_observed_label(self, customer_id: str, merchant_id: str, label: int) -> None:
        """Apply a label that has actually arrived; callers enforce availability time."""
        customer = self.customers[customer_id]
        merchant = self.merchants[merchant_id]
        customer.confirmed_labels += 1
        merchant.confirmed_labels += 1
        customer.confirmed_fraud += int(label)
        merchant.confirmed_fraud += int(label)


FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "home_merchant_distance_km",
    "log_city_population",
    "customer_avg_amount_7d",
    "customer_avg_amount_30d",
    "amount_vs_customer_avg_30d",
    "customer_transactions_5m",
    "customer_transactions_30m",
    "customer_transactions_1h",
    "customer_transactions_24h",
    "customer_amount_1h",
    "customer_unique_merchants_1h",
    "time_since_previous_seconds",
    "distance_from_previous_km",
    "travel_speed_kmh",
    "impossible_travel_flag",
    "customer_confirmed_fraud_rate",
    "merchant_transactions_24h",
    "merchant_avg_amount_30d",
    "merchant_confirmed_fraud_rate",
    "customer_cold_start",
    "merchant_cold_start",
    "history_unavailable",
]
