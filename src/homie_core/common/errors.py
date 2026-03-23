"""Homie-specific exception hierarchy.

All Homie exceptions inherit from HomieError, making it easy to catch
any framework-level error with a single ``except HomieError`` clause while
still allowing callers to handle specific failure modes individually.
"""


class HomieError(Exception):
    """Base exception for all Homie errors."""

    def __init__(self, message: str = "", *, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)


class ModelNotLoadedError(HomieError):
    """Model engine is not loaded."""
    pass


class InferenceError(HomieError):
    """Inference failed."""
    pass


class StorageError(HomieError):
    """Storage operation failed."""
    pass


class ConfigurationError(HomieError):
    """Invalid configuration."""
    pass


class AgentError(HomieError):
    """Agent execution failed."""
    pass


class VoiceError(HomieError):
    """Voice pipeline error."""
    pass


class NetworkError(HomieError):
    """Network/sync operation failed."""
    pass
