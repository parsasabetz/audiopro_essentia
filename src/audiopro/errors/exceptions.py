"""Custom exceptions for audio processing."""

# typing imports
from typing import Optional, Any, Dict

# system imports
import inspect
import os


class AudioProcessingError(Exception):
    """
    Exception raised for errors in the audio processing module.

    Attributes:
        message (str): Explanation of the error.
        details (Optional[Dict[str, Any]]): Additional details about the error.

    Methods:
        __str__(): Returns a string representation of the error, including details if available.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}

        # Add call location information automatically
        frame = inspect.currentframe()
        if frame:
            caller = frame.f_back  # Get the caller's frame
            if caller:
                self.details.update(
                    {
                        "file": os.path.basename(caller.f_code.co_filename),
                        "function": caller.f_code.co_name,
                        "line": caller.f_lineno,
                    }
                )
        super().__init__(self.message)

    def __str__(self) -> str:
        location = f"[{self.details.get('file')}:{self.details.get('function')}:{self.details.get('line')}]"
        if self.details:
            return f"{location} {self.message} - Details: {self.details}"
        return f"{location} {self.message}"


class FeatureExtractionError(AudioProcessingError):
    """
    Exception raised for errors occurring during feature extraction in audio processing.

    Attributes:
        message (str): Human-readable description of the error.
        frame_index (Optional[int]): Index of the frame where the error occurred.
        feature_name (Optional[str]): Name of the feature being extracted when the error occurred.
        error_type (Optional[str]): Type of the error.
        **kwargs: Additional keyword arguments to provide more details about the error.

    Args:
        message (str): Human-readable description of the error.
        frame_index (Optional[int], optional): Index of the frame where the error occurred. Defaults to None.
        feature_name (Optional[str], optional): Name of the feature being extracted when the error occurred. Defaults to None.
        error_type (Optional[str], optional): Type of the error. Defaults to None.
        **kwargs: Additional keyword arguments to provide more details about the error.
    """

    def __init__(
        self,
        message: str,
        frame_index: Optional[int] = None,
        feature_name: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs,
    ):
        details = {
            "frame_index": frame_index,
            "feature_name": feature_name,
            "error_type": error_type,
            **kwargs,
        }
        super().__init__(message, details)


class ExtractionPipelineError(AudioProcessingError):
    """
    Exception raised for pipeline-level errors.

    Attributes:
        message (str): The error message.
        error_count (Optional[int]): The number of errors encountered during the pipeline execution.
        total_frames (Optional[int]): The total number of frames processed.
        failed_frames (Optional[list]): A list of frames that failed during processing.
        **kwargs: Additional keyword arguments to include in the error details.

    Args:
        message (str): The error message.
        error_count (Optional[int], optional): The number of errors encountered during the pipeline execution. Defaults to None.
        total_frames (Optional[int], optional): The total number of frames processed. Defaults to None.
        failed_frames (Optional[list], optional): A list of frames that failed during processing. Defaults to None.
        **kwargs: Additional keyword arguments to include in the error details.
    """

    def __init__(
        self,
        message: str,
        error_count: Optional[int] = None,
        total_frames: Optional[int] = None,
        failed_frames: Optional[list] = None,
        **kwargs,
    ):
        details = {
            "error_count": error_count,
            "total_frames": total_frames,
            "failed_frames": failed_frames,
            **kwargs,
        }
        super().__init__(message, details)


class AudioValidationError(AudioProcessingError):
    """
    Exception raised for errors in the audio validation process.

    Attributes:
        message (str): Explanation of the error.
        parameter (Optional[str]): The parameter that caused the error, if applicable.
        expected (Optional[Any]): The expected value of the parameter, if applicable.
        actual (Optional[Any]): The actual value of the parameter, if applicable.
        **kwargs: Additional keyword arguments to include in the error details.
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        **kwargs,
    ):
        details = {
            "parameter": parameter,
            "expected": expected,
            "actual": actual,
            **kwargs,
        }
        super().__init__(message, details)


class FrameProcessingError(FeatureExtractionError):
    """
    Exception raised for errors that occur during frame processing in feature extraction.

    Attributes:
        message (str): Description of the error.
        frame_index (int): Index of the frame where the error occurred.
        frame_time (Optional[float]): Time of the frame where the error occurred.
        frame_shape (Optional[tuple]): Shape of the frame where the error occurred.
        **kwargs: Additional keyword arguments passed to the base exception class.

    Args:
        message (str): Description of the error.
        frame_index (int): Index of the frame where the error occurred.
        frame_time (Optional[float], optional): Time of the frame where the error occurred. Defaults to None.
        frame_shape (Optional[tuple], optional): Shape of the frame where the error occurred. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base exception class.
    """

    def __init__(
        self,
        message: str,
        frame_index: int,
        frame_time: Optional[float] = None,
        frame_shape: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            frame_index=frame_index,
            frame_time=frame_time,
            frame_shape=frame_shape,
            **kwargs,
        )


class SpectralFeatureError(FeatureExtractionError):
    """
    Exception raised for errors in the spectral feature extraction process.

    Attributes:
        message (str): Explanation of the error.
        feature_name (str): Name of the spectral feature that caused the error.
        spectrum_shape (Optional[tuple]): Shape of the spectrum data, if applicable.
        frequency_range (Optional[tuple]): Frequency range of the spectrum data, if applicable.

    Args:
        message (str): Explanation of the error.
        feature_name (str): Name of the spectral feature that caused the error.
        spectrum_shape (Optional[tuple], optional): Shape of the spectrum data, if applicable. Defaults to None.
        frequency_range (Optional[tuple], optional): Frequency range of the spectrum data, if applicable. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        message: str,
        feature_name: str,
        spectrum_shape: Optional[tuple] = None,
        frequency_range: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            feature_name=feature_name,
            spectrum_shape=spectrum_shape,
            frequency_range=frequency_range,
            **kwargs,
        )


class AudioIOError(AudioProcessingError):
    """
    Exception raised for audio file I/O operations.

    Examples:
        - File not found
        - Invalid file format
        - File read/write errors
        - Codec errors
        - File permission issues

    Attributes:
        message (str): Description of the error
        filepath (Optional[str]): Path to the problematic file
        operation (Optional[str]): The operation that failed (read/write/decode)
        **kwargs: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        details = {
            "filepath": filepath,
            "operation": operation,
            **kwargs,
        }
        super().__init__(message, details)
