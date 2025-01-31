"""
Constants for audio processing.

Attributes:
    FRAME_LENGTH (int): The length of each audio frame in samples.
    HOP_LENGTH (int): The number of samples between the start of consecutive frames.
    BATCH_SIZE (int): The number of frames to process in a single batch.
    FREQUENCY_BANDS (dict): A dictionary defining the frequency ranges for different audio bands.
        Keys are the names of the bands (str), and values are tuples containing the lower and upper
        frequency limits in Hz.
"""

FRAME_LENGTH = 2048

HOP_LENGTH = 512

BATCH_SIZE = 1000 # Process frames in batches

FREQUENCY_BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "upper_mid": (2000, 5000),
    "treble": (5000, 20000),
}
