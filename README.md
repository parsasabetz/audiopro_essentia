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

### Basic Usage
```python
from audiopro import analyze_audio

# Default MessagePack output
analyze_audio(
    file_path="input.mp3",
    output_file="analysis_output",  # Creates analysis_output.msgpack
)

# JSON output (explicitly specified)
analyze_audio(
    file_path="input.mp3",
    output_file="analysis_output",
    output_format="json",  # Creates analysis_output.json
)

# skip monitoring (default: False)
analyze_audio(
    file_path="input.mp3",
    output_file="analysis_output",
    skip_monitoring=True,
)
```

### Output Format
The library supports two output formats:
- `msgpack` (default): Binary MessagePack format for efficient storage
- `json`: Human-readable JSON format

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
    "features": [
        {
            "time": float,  # Time in seconds
            "rms": float,   # Root mean square energy
            "spectral_centroid": float,
            "spectral_bandwidth": float,
            "spectral_flatness": float,
            "spectral_rolloff": float,
            "zero_crossing_rate": float,
            "mfcc": [float],  # 13 coefficients
            "frequency_bands": {
                "sub_bass": float,    # 20-60 Hz
                "bass": float,        # 60-250 Hz
                "low_mid": float,     # 250-500 Hz
                "mid": float,         # 500-2000 Hz
                "upper_mid": float,   # 2000-5000 Hz
                "treble": float       # 5000-20000 Hz
            },
            "chroma": [float]  # 12 values representing pitch classes
        }
    ]
}
```

## Features

- **Audio Analysis**: Extract tempo, beats, and spectral features using `essentia`.
- **Performance Monitoring**: Built-in CPU and memory usage tracking leveraging multiprocessing.
- **Multiprocessing**: Parallel processing for faster analysis of large audio files.
- **Flexible Output**: Choose between JSON and MessagePack formats.
- **Resource Efficient**: Optimized for large audio files with efficient multiprocessing.
- **Metadata Extraction**: Extract detailed audio metadata (e.g., duration, sample rate)

## Project Structure
```
audiopro/
└── src/
    └── audiopro/
        ├── __init__.py      # Public API
        ├── process.py       # Core processing
        ├── extractor.py     # Feature extraction
        └── metadata.py      # Metadata extraction
        └── monitor.py       # Performance monitoring
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