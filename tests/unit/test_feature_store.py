from fraud_detection.inference.feature_store import InMemoryFeatureStore


def test_online_store_is_idempotent_and_degraded(event) -> None:
    store = InMemoryFeatureStore(degraded=True)
    first = store.get_and_update(event)
    second = store.get_and_update(event)
    assert first.features == second.features
    assert first.degraded
    assert first.features["history_unavailable"] == 1
