# AudioPro

A high-performance audio processing library with built-in performance monitoring.

## Installation

Ensure you have Python 3.7 or higher installed. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

- This project uses `essentia` for audio processing and employs multiprocessing for performance optimization.

**Requirements:**
- Python 3.12 or higher
- For Apple Silicon users: `xcode-select --install` may be required

```bash
pip install git+ssh://git@github.com/parsasabetz/audiopro.git
```

## Usage

### Command Line Usage
```bash
# Analyze all features (creates output.msgpack)
python -m audiopro input.wav output

# Analyze specific features only (computes only the selected features):
python -m audiopro input.wav output --features rms spectral_centroid mfcc

# To compute all features, either omit --features or pass an empty list:
python -m audiopro input.wav output --features

# Output as JSON (creates output.json)
python -m audiopro input.wav output --format json

# Skip performance monitoring
python -m audiopro input.wav output --skip-monitoring

# INCORRECT - don't include extension in output name
❌ python -m audiopro input.wav output.json  # This will raise an error

# CORRECT - specify format with --format flag
✓ python -m audiopro input.wav output --format json
```

### Programmatic Usage
```python
import asyncio
from audiopro import analyze_audio, FeatureConfig

# Basic usage - analyze all features
await analyze_audio(
    file_path="input.wav",
    output_path="output",  # No extension needed
    output_format="msgpack"  # This determines the extension (.msgpack)
)

# Analyze specific features only. 
# Note: Passing an empty dictionary (or one with all False values) computes all features.
feature_config: FeatureConfig = {
    "rms": True,
    "spectral_centroid": True,
    "mfcc": True,
    # Features not included or set to False will be excluded. 
    # To compute all features, simply pass None or an empty dict.
}

await analyze_audio(
    file_path="input.wav",
    output_path="output",  # No extension needed
    output_format="json",  # This determines the extension (.json)
    feature_config=feature_config
)

# Batch processing multiple files
from audiopro.batch import batch_process_audio

results = await batch_process_audio(
    input_files=["file1.wav", "file2.mp3"],
    output_dir="output",  # Directory name only
    feature_config=feature_config,
    max_concurrent=2  # Process 2 files at a time
)
```

### Output Format
The library supports two output formats:
- `msgpack` (default): Binary MessagePack format for efficient storage (creates `.msgpack` files)
- `json`: Human-readable JSON format (creates `.json` files)

**Important**: Never include the file extension in the output path. Instead:
1. Provide the output path without extension
2. Use `output_format` or `--format` to specify the format
3. The library will automatically add the correct extension

For example:
```python
# INCORRECT ❌
await analyze_audio(
    file_path="input.wav",
    output_path="output.json",  # Don't include extension
    output_format="json"
)

# CORRECT ✓
await analyze_audio(
    file_path="input.wav",
    output_path="output",  # No extension
    output_format="json"  # This determines the extension
)
```

### Feature Selection
You can selectively enable/disable any of these features:
- `rms`: Root Mean Square energy value
- `spectral_centroid`: Weighted mean of frequencies
- `spectral_bandwidth`: Variance of frequencies around the centroid
- `spectral_flatness`: Measure of how noise-like the signal is
- `spectral_rolloff`: Frequency below which most spectral energy exists
- `zero_crossing_rate`: Rate of signal polarity changes
- `mfcc`: Mel-frequency cepstral coefficients (13 values)
- `frequency_bands`: Energy in different frequency bands
- `chroma`: Distribution of spectral energy across pitch classes

Feature selection can be done in three ways:
1. Pass `None` as `feature_config` to compute all features (default behavior)
2. Include only the features you want with `True` values
3. Explicitly disable features with `False` values (optional)

**Important:**  
- The `feature_config` argument controls which features to compute.  
  - If you pass `None`, the analysis computes all features.  
  - If you pass a dictionary with selected features enabled, only those will be computed.  
  - If an empty dictionary (or a dictionary where no feature is enabled) is provided, it will default to computing all features.
- The output includes an `included_features` field that lists which features were computed.  
  - An empty list indicates that all available features were computed.

Example configurations:
```python
from audiopro import FeatureConfig

# Compute all features
feature_config = None // or
feature_config = {}

# Compute only RMS and MFCC
feature_config: FeatureConfig = {
    "rms": True,
    "mfcc": True
}

# Compute everything _except_ spectral features
feature_config: FeatureConfig = {
    "spectral_centroid": False,
    "spectral_bandwidth": False,
    "spectral_flatness": False,
    "spectral_rolloff": False
}
```

### Output Structure
```python
{
    "metadata": {
        "file_info": {
            "filename": str,
            "format": str,
            "size_mb": float,
            "created_date": str,  # ISO format
            "mime_type": str,
            "sha256_hash": str
        },
        "audio_info": {
            "duration_seconds": float,
            "sample_rate": int,
            "channels": int,
            "peak_amplitude": float,
            "rms_amplitude": float,
            "dynamic_range_db": float,
            "quality_metrics": {
                "dc_offset": float,
                "silence_ratio": float,
                "potentially_clipped_samples": int
            }
        }
    },
    "tempo": float,  # Beats per minute
    "beats": [float],  # List of beat timestamps in seconds
    "included_features": [], // List of enabled features; empty means all were computed.
    "features": [
        {
            "time": float,  # Time in milliseconds (always included)
            # Only requested features will be included:
            "rms": float,   # Optional
            "spectral_centroid": float,  # Optional
            "spectral_bandwidth": float,  # Optional
            "spectral_flatness": float,  # Optional
            "spectral_rolloff": float,  # Optional
            "zero_crossing_rate": float,  # Optional
            "mfcc": [float],  # Optional, 13 coefficients
            "frequency_bands": {  # Optional
                "sub_bass": float,    # 20-60 Hz
                "bass": float,        # 60-250 Hz
                "low_mid": float,     # 250-500 Hz
                "mid": float,         # 500-2000 Hz
                "upper_mid": float,   # 2000-5000 Hz
                "treble": float       # 5000-20000 Hz
            },
            "chroma": [float]  # Optional, 12 values representing pitch classes
        }
    ]
}
```

## Features

- **Selective Feature Computation**: Choose which audio features to compute to optimize processing time and output size
- **Audio Analysis**: Extract tempo, beats, and spectral features using `essentia`
- **Performance Monitoring**: Built-in CPU and memory usage tracking leveraging multiprocessing
- **Multiprocessing**: Parallel processing for faster analysis of large audio files
- **Flexible Output**: Choose between JSON and MessagePack formats
- **Resource Efficient**: Optimized for large audio files with efficient multiprocessing
- **Metadata Extraction**: Extract detailed audio metadata (e.g., duration, sample rate)
- **Batch Processing**: Process multiple files concurrently with error handling

## Project Structure
```
audiopro/
└── src/
    └── audiopro/
        ├── __init__.py         # Public API
        ├── analysis/           # Analysis controller
        ├── audio/             # Audio processing
        │   ├── extractor.py   # Feature extraction
        │   ├── processors.py  # Frame processing
        │   └── models.py      # Data models
        ├── output/            # Output handling
        │   ├── types.py       # Type definitions
        │   └── handler.py     # Output writing
        ├── monitor/           # Performance monitoring
        └── utils/             # Utility functions
```

## Development

### Setting up development environment
```bash
git clone git@github.com:parsasabetz/audiopro.git
cd audiopro
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Commit Convention
Follow the Commitizen format:
```
<type>(<scope>): <subject>
```

Make commits using:
```bash
git cz
```

Version bumping:
```bash
cz bump
```

## License

This project is closed source. All rights reserved.