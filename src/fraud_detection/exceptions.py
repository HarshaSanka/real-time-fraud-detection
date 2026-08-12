"""Domain exceptions."""


class FraudDetectionError(Exception):
    """Base error for the fraud platform."""


class DataValidationError(FraudDetectionError):
    """Raised when source or event validation fails."""


class ModelNotReadyError(FraudDetectionError):
    """Raised when no valid model bundle is available."""


class FeatureStoreUnavailableError(FraudDetectionError):
    """Raised when online history cannot be retrieved."""
