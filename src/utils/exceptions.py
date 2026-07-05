"""Custom exception hierarchy for the price predictor project.

Every module raises one of these instead of letting raw library exceptions
(KeyError, requests.RequestException, torch errors, ...) bubble up unlabeled.
This makes failures traceable to a pipeline stage from the exception type alone.
"""


class PricePredictorError(Exception):
    """Base class for all project-specific errors."""


class ConfigError(PricePredictorError):
    """Raised when a config file is missing, malformed, or missing required keys."""


class DataError(PricePredictorError):
    """Raised for schema violations, missing files, or corrupt data during loading."""


class ImageDownloadError(DataError):
    """Raised when an image cannot be downloaded or decoded after retries."""


class EncoderError(PricePredictorError):
    """Raised when a text/image encoder fails to load or fails during encode()."""


class FusionError(PricePredictorError):
    """Raised for shape mismatches or invalid configuration in a fusion module."""


class ModelBuildError(PricePredictorError):
    """Raised when the full price model cannot be assembled from a config."""


class TrainingError(PricePredictorError):
    """Raised for unrecoverable failures during the training loop (e.g. NaN loss)."""


class CheckpointError(PricePredictorError):
    """Raised when a checkpoint cannot be saved or loaded."""


class InferenceError(PricePredictorError):
    """Raised when end-to-end prediction on a new raw input fails."""
