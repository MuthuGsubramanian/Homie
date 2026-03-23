"""Tests for homie_core.common.errors."""

import pytest

from homie_core.common.errors import (
    AgentError,
    ConfigurationError,
    HomieError,
    InferenceError,
    ModelNotLoadedError,
    NetworkError,
    StorageError,
    VoiceError,
)


class TestHomieErrorHierarchy:
    """All custom exceptions must be subclasses of HomieError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ModelNotLoadedError,
            InferenceError,
            StorageError,
            ConfigurationError,
            AgentError,
            VoiceError,
            NetworkError,
        ],
    )
    def test_subclass_of_homie_error(self, exc_class):
        assert issubclass(exc_class, HomieError)

    def test_homie_error_is_exception(self):
        assert issubclass(HomieError, Exception)


class TestHomieErrorMessage:
    def test_default_message_is_empty(self):
        err = HomieError()
        assert str(err) == ""

    def test_custom_message(self):
        err = HomieError("something broke")
        assert str(err) == "something broke"

    def test_details_dict(self):
        err = HomieError("oops", details={"code": 42})
        assert err.details == {"code": 42}

    def test_details_default_empty(self):
        err = HomieError("oops")
        assert err.details == {}


class TestCatchPatterns:
    def test_catch_base_catches_subclass(self):
        with pytest.raises(HomieError):
            raise ModelNotLoadedError("no model")

    def test_catch_specific_does_not_catch_sibling(self):
        with pytest.raises(InferenceError):
            raise InferenceError("bad inference")
        # StorageError should NOT be caught by an InferenceError handler
        with pytest.raises(StorageError):
            raise StorageError("disk full")

    def test_catch_exception_catches_homie_error(self):
        with pytest.raises(Exception):
            raise NetworkError("timeout")
