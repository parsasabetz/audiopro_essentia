# AudioPro

A high-performance audio processing library with built-in performance monitoring.

## Installation

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

# Default JSON output
analyze_audio(
    file_path="input.mp3",
    output_file="analysis_output",  # Will create analysis_output.json
)

# MessagePack output
analyze_audio(
    file_path="input.mp3",
    output_file="analysis_output",
    output_format="msgpack"  # Will create analysis_output.msgpack
)
```

### Output Format
The library supports two output formats:
- `json` (default): Human-readable JSON format
- `msgpack`: Binary MessagePack format for efficient storage

### Output Structure
```python
{
    "tempo": float,  # Beats per minute
    "beats": [float],  # List of beat timestamps in seconds
    "features": [
        {
            "time": float,  # Time in seconds
            "rms": float,   # Root mean square energy
            "spectral_centroid": float,
            "spectral_bandwidth": float,
            "frequency_bands": {
                "low": [float],    # < 250 Hz
                "mid": [float],    # 250-2000 Hz
                "high": [float]    # > 2000 Hz
            }
        }
    ]
}
```

## Features

- **Audio Analysis**: Extract tempo, beats, and spectral features
- **Performance Monitoring**: Built-in CPU and memory usage tracking
- **Multi-threaded**: Parallel processing for faster analysis
- **Flexible Output**: Choose between JSON and MessagePack formats
- **Resource Efficient**: Optimized for large audio files
- **GPU Support**: Optional GPU monitoring when available

## Project Structure
```
audiopro/
└── src/
    └── audiopro/
        ├── __init__.py      # Public API
        ├── process.py       # Core processing
        ├── extractor.py     # Feature extraction
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