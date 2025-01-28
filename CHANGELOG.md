## 0.3.0 (2025-01-29)

### BREAKING CHANGE

- The audio processing module has been refactored to use Essentia library instead of Librosa.

### Feat

- **metadata**: Add get_file_metadata for public export
- **process**: Integrate metadata module with main process
- **metadata**: Add dedicated metadata module for audio file analysis

### Fix

- **metadata**: Correct comment for size calculation in get_file_metadata function

### Refactor

- **audiopro**: Replace librosa with Essentia for audio processing, enhance spectral analysis
- **monitor**: Enhance CPU monitoring logic and improve performance statistics reporting, reduce CPU overhead in the monitor file for better performance.

### Perf

- **metadata**: Increase block size for hash calculation and improve RMS calculation efficiency
- **metadata**: Ensure consistent type conversion for metadata
- **process**: Simplify input validation and improve error handling
- **extractor**: Improve feature extraction robustness and JSON serialization, add new features for processing

## 0.2.0 (2025-01-27)

### BREAKING CHANGE

- The changes in this commit fully transition the scripts into a complete python library.

### Fix

- **monitor.py**: make GPU monitoring optional for better compatibility
- **deps**: update dependencies and add setuptools for Python 3.12 support
- **process.py**: correct output format handling and add parameter validation

### Refactor

- **src/audiopro**: relocate core modules to src/audiopro directory

## 0.1.0 (2025-01-27)

### BREAKING CHANGE

- To run a process, you should now provide the name and the extension of the audio file but for the output, only the name of the output file. You can also use the `--format` flag to output a msgpack file instead of JSON, as per your needs.

### Feat

- **process.py**: This commit implements the feature of having MessagePack as an option for output file

### Refactor

- **.cz.json**: Minor formatting for better readability
