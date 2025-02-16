## 1.2.0 (2025-02-16)

### Feat

- **Chroma**: implement cached pitch class mapping for optimized Chroma computation in spectrogram processing
- implement feature flag management using bit operations for efficient feature tracking

### Fix

- enhance audio loading and processing logic, improve metadata extraction, and update logging, fix the extraction duration bug where the entire audio would not be processed because of multi-channel processing, switching to mono processing now

### Refactor

- optimize feature checking by implementing feature flags for efficient processing
- update output module documentation and enhance type definitions for feature management for a more efficient solution
- update `get_transforms` to conditionally compute spectrum and MFCC transformations based on feature configuration

## 1.1.0 (2025-02-15)

### Feat

- add frequency and spectrogram processing modules with optimized constants and error handling
- refactor rhythm extraction to use TorchAudio for beat tracking and processing
- replace `essentia` with `torchaudio` for audio loading and processing
- add `SpectrogramConfig` class for spectrogram generation and validation

### Fix

- improve logging for feature extraction and add error message for pipeline failures

### Refactor

- update requirements to replace `audioflux` with `torchaudio`, add `ffmpeg-python` and add additional dependencies
- update metadata structure to include audio `codec` and improve documentation
- replace `torchaudio` with `ffmpeg` for audio metadata extraction and improve logging
- improve docstring for `get_transforms` function to enhance clarity and detail
- enhance docstrings for `spectrogram` processing functions to improve clarity and usability
- optimize audio processing pipeline for improved performance and maintainability
- update audio module exports to include spectrogram processing and remove unused frequency functions
- remove redundant frame generator function and import `create_frame_generator` for improved memory efficiency
- implement memory-efficient audio frame generator using constants for frame and hop lengths
- enhance error handling in feature extraction and remove unused `total_frames` parameter
- simplify frame generator by removing `total_frames` parameter and optimizing memory usage
- streamline feature extraction by using `FRAME_LENGTH` for FFT size and optimizing worker calculation, refactor to use constants instead of redeclaring them
- use constant for hop length in rhythm extraction
- update `SpectrogramConfig` to use constants for FFT size and hop length
- Refactor to use TorchAudio, improve logging and simplify feature extraction process
- update `TimeRange` and `LoaderMetadata` types for optional fields and simplified `mime_type`

## 1.0.0 (2025-02-15)

## 0.14.0 (2025-02-14)

### Feat

- **package**: change package name to lowercase in `pyproject.toml`
- **package**: change package name to lowercase and specify package directory in `setup.py`

### Fix

- **package**: update package discovery to support namespace packages
- **output**: expose `AVAILABLE_FEATURES` and `create_feature_config` in output module

## 0.13.0 (2025-02-10)

### Feat

- **output**: add `--gzip` option for enabling gzip compression in `msgpack` output
- **output**: add gzip compression option for msgpack output in `write_output` function
- **package**: update package name to `AudioPro` in `pyproject.toml` and `setup.py`
- **errors**: enhance `AudioProcessingError` message formatting for better debugging
- **docs**: update README to include file validation and partial processing features
- **docs**: add audio validation utilities and update README structure
- **types, docs**: enhance MIME type handling with autocomplete support in `TypedDicts`
- **audio_loader**: enhance audio loading with validation and improved metadata handling
- add audio file validation utilities and enhance module exports

## 0.12.0 (2025-02-05)

### Feat

- **extractor**: improve memory management by releasing batch frames and forcing garbage collection
- **processors**: enhance docstring for `process_frame` function with detailed parameter and return information
- **processors**: enhance error tracking and stats for processing frames for feature extraction
- **extractor**: enhance error tracking and logging in audio feature extraction
- **errors**: add error tracking utilities and enhance error statistics management
- **processors**: enhance frame processing with improved error handling and validation
- **extractor**: improve error handling with detailed messages for audio extraction
- **audio_loader**: enhance error handling with custom exceptions for audio loading
- **errors**: add custom exceptions for audio processing and create error module
- **processors, extractor**: enhance frame processing with improved validation and error handling
- **exceptions**: add custom exceptions for audio processing errors

### Fix

- **output**: update time attribute description to clarify it as a Unix timestamp in seconds
- **output**: remove `strict_types` from msgpack packing to improve compatibility

### Refactor

- **audio_loader**: update in-place even-length adjustment logic to drop last sample if odd to reduce memory allocation
- **exceptions**: clarify comment on automatic call location information
- **audio**: remove obsolete exception imports from audio module
- **exceptions**: remove obsolete custom exceptions for audio processing

## 0.11.0 (2025-02-04)

### Feat

- **audio**: optimize spectral bandwidth computation with LRU caching and einsum
- **process**: add LRU cache to `calculate_max_workers` for improved performance

### Refactor

- **processors**: implement cached algorithm creators for spectrum, MFCC, and HPCP to enhance performance
- **extractor**: add early return for disabled features and optimize chunk size calculation
- **metadata**: streamline audio metadata extraction by simplifying `file_info` and optimizing metric calculations
- **output**: consolidate feature definitions and improve configuration structure
- **audio**: enhance memory efficiency in rhythm extraction and spectral bandwidth computation
- **constants**: encapsulate audio constants in a frozen dataclass for better organization and memory efficiency
- **path**: replace set with frozenset for supported formats and use tuple-based mapping for format extensions
- **logger**: replace dictionary with `WeakValueDictionary` for better memory management of loggers

## 0.10.0 (2025-02-04)

### Feat

- **audio**: add duration return value to audio loading function for enhanced audio metadata
- **audio**: add time range support for audio analysis and processing efficiently without writing any files
- **audio/processors**: add `volume` feature calculation and update feature configuration and docs
- **audio/extractor**: improve frame calculation and logging for audio feature extraction and fix frame dropping bugs
- **analysis/extractor**: add channel handling for audio feature extraction and duration calculation to calculate the duration of the audio correctly
- **audio/processors**: add multi-channel to mono conversion in frame processing for proper array shape handling
- **utils/audio**: add logging for rhythm extraction and handle multi-channel audio conversion
- **analysis/controller**: enhance audio loading with additional metadata and refactor `get_file_metadata` parameters

### Fix

- **audio/utils**: recalculate tempo based on median beat interval for improved accuracy in rhythm extraction
- **controller**: recalculate tempo based on median beat interval for improved rhythm analysis, fix the wrong tempo value bug
- **audio_loader**: enable MD5 computation during audio file loading
- **audio_loader**: optimize even-length adjustment for audio data using in-place padding
- **types**: update time attribute to reflect milliseconds instead of seconds

### Refactor

- **audio**: replace `start_time` parameter with `start_sample` for improved audio frame processing
- **utils**: enhance documentation for `optimized_convert_to_native_types` function, detailing input types and conversion process
- **audio/processors**: streamline frame processing by simplifying mono conversion and removing explicit deletion
- **utils**: optimize conversion of numpy types to native Python types using vectorized operations

## 0.9.0 (2025-02-02)

### Feat

- **analysis/controller**: optimize thread pool usage and improve `included_features` determination
- **audio/analysis**: add `included_features` to analysis results and update type definitions to have an insight of the output structure in the top of the file
- **output/types**: define spectral features requiring spectrum computation
- **output**: add asynchronous JSON and MessagePack writing functions to their own files, enhance and optimize the logic for output writing

### Fix

- **analysis/controller**: handle empty `feature_config` by allowing computation of all features
- **audiopro/main**: remove unnecessary blank line in main execution block

### Refactor

- **output/types**: centralize feature names and improve type definitions for better clarity and performance
- **audio/processors**: optimize feature computation by filtering enabled features and reducing spectrum calculations
- **audio/models**: streamline time validation and enhance create method type hints

## 0.8.0 (2025-02-01)

### Feat

- **audio/models**: enhance `FrameFeatures` with type validation and improved type hints for computed features
- **audio/processors**: enhance feature computation by adding configurable feature selection, allowing selective computation based on provided configuration
- **analysis/controller**: convert `FrameFeatures` to dictionary format before appending to features list, fixing the feature extraction bugs
- **audio/models, processors**: refactor `FrameFeatures` to use a dictionary for computed features and add validation for time, fixes the `null` values being shown in the output file
- **output/types**: change `AVAILABLE_FEATURES` to a frozenset for immutability and performance improvements, other minor changes to improve the code
- **arg_parser**: update feature selection to use `AVAILABLE_FEATURES` and streamline feature config creation
- **output_types**: add feature configuration and validation for audio analysis features
- **arg_parser, output_handler, path**: update argument parsing and output handling to use `OutputFormat` type for better type safety
- **output_handler**: refactor `write_output` to validate output path and improve documentation
- **main**: update main execution flow to pass raw output path and include feature configuration
- **arg_parser**: enhance argument parsing to include feature selection and improve output format handling
- **extractor**: add feature configuration options to `extract_features` and improve logging
- **extractor**: enhance audio analysis by adding feature configuration options and improving error handling
- **utils**: add path handling utilities for output path validation and format extension management
- **extractor**: make audio feature attributes optional and add `to_dict` method for `FrameFeatures`

### Refactor

- **audio/extractor**: optimize feature extraction by creating process function once per batch
- **output/types**: format `create_feature_config` function for improved readability
- **output/types**: improve feature configuration handling and default behavior

## 0.7.0 (2025-02-01)

### BREAKING CHANGE

- Frame indexing logic has been modified to ensure complete audio coverage

### Feat

- **extractor**: update audio module documentation and reorganize imports for clarity, remove unused feature utilities, optimize memory usage and error handling
- **extractor**: add `FrameFeatures` data class and implement frame processing for spectral feature extraction with error handling, separate audio processing code into modular structure
- **extractor**: implement memory-efficient frame generator and enhance feature extraction with error handling and progress tracking
- **extractor**: refactor frame processing to use type-safe container for features and optimize frequency bin calculations
- **extractor**: add `on_feature` callback to `extract_features` for immediate feature dispatch, this is to optimize memory usage, improve performance, and ensure system robustness during audio analysis
- **extractor**: optimize frame processing with improved memory usage and progress tracking
- **extractor**: enhance frame extraction with precise calculations and progress tracking, this fixes issues with frame processing
- **audio**: update time calculation in `process_frame` to return correct value milliseconds
- **performance**: enhance execution time logging with milliseconds
- **audiopro/analysis**: add analysis module with audio analysis export
- **audiopro/audio**: add audio module with feature extraction utilities and exports
- **utils/process**: add `calculate_max_workers` function and update exports in init file
- **audiopro/analysis/controller**: implement audio analysis function with monitoring and output handling
- **utils**: implement graceful shutdown context manager for process termination
- **monitor**: add dynamic loading of monitor functions for performance tracking

### Fix

- **__init__**: add missing newline at end of file

### Refactor

- **audiopro/audio/extractor**: streamline feature extraction process and improve batch processing efficiency
- **audiopro/main**: simplify main script by consolidating audio analysis logic and improving documentation
- **main**: remove unused imports and streamline monitoring function loading
- **audiopro/main**: dynamically load monitoring functions to optimize performance monitoring

## 0.6.0 (2025-01-31)

### Feat

- **logger**: implement singleton logger with global exception handling and memory optimization
- **output**: enhance output handling with `AudioAnalysis` type and improve function signatures
- **output**: add type definitions for audio analysis output

### Fix

- **logger**: enhance LoggerSingleton class with detailed docstrings and improve handler management, include file name in the logs properly
- **utils**: restore `optimized_convert_to_native_types` to module exports
- **monitor**: improve error handling in CPU usage monitoring and enhance code clarity
- **metadata**: enhance error handling and logging in file hash calculation
- **logging**: update logger message formatting in `load_and_preprocess_audio` function
- **types**: add missing newline for improved readability in AudioFeature class
- **analyze_audio**: rename `output_file` parameter to `output_path` for clarity

### Refactor

- **logging**: streamline audio loading logs by removing redundant messages
- **extractor**: move audio processing constants to utils for better organization
- **utils**: add constants for audio processing and update exports
- **main**: improve code formatting and enhance readability in audio analysis function
- **main**: implement graceful shutdown context manager and improve CPU monitoring logic
- **extractor**: implement memory-efficient frame generator and adjust batch processing logic
- **monitor**: adjust CPU usage monitoring timeout and improve error handling
- **output**: enhance output writing with improved error handling and serialization for JSON and MessagePack formats
- **logging**: streamline logging in audio analysis and output handling, reducing redundancy and improving clarity
- **logging**: replace standard logging with custom logger in audio, monitor, and output modules for consistency
- **metadata**: enhance docstrings for `calculate_file_hash` and `get_file_metadata` functions for better clarity and detail
- **logging**: replace standard logging setup with custom logger for consistency across modules
- **audio**: implement singleton pattern for rhythm extraction for better efficiency and enhance error handling
- **utils**: enhance `optimized_convert_to_native_types` for improved type handling and efficiency
- **pylintrc**: reduce max-statements limit for improved code maintainability
- **extractor**: improve readability and error handling in `process_frame` and `extract_features` functions
- **extractor**: reorganize imports for clarity and consistency
- **audio**: remove unused import of warnings for cleaner code
- **output_handler**: enhance logging format by using placeholders for consistency
- **analyze_audio**: improve logging format and consistency by using placeholders
- **analyze_audio**: streamline analysis result compilation and type conversion
- **main**: remove unused warnings import for cleaner code

### Perf

- **monitor**: reduce CPU usage monitoring interval for improved responsiveness

## 0.5.0 (2025-01-30)

### BREAKING CHANGE

- metadata and rhythm extraction are now sync functions

### Feat

- **audiopro**: Enhance output writing with error handling for asynchronous file operations
- **audiopro**: Integrate rhythm extraction functionality into audio analysis process by utility imports
- **audiopro**: Move spectral bandwidth computation to audio module and add rhythm extraction function
- **audiopro**: Add asynchronous audio analysis with performance monitoring and flexible output format support, now in the `main.py` file
- **audiopro**: Add asynchronous output handler for analysis results with JSON and MessagePack support, to the output directory
- **audiopro**: Add utility functions for spectral bandwidth computation and optimized type conversion, in their own directory
- **monitor**: Add CPU and performance monitoring module with enhanced docstrings and output handling, in their own directory
- **audio**: Add audio loading and metadata extraction modules with preprocessing and async hash calculation, in their own directory
- **audiopro**: Add audio loading and preprocessing functionality with async output handling, extract these features from the process file to their own files

### Fix

- **.gitignore**: Update ignore rules to correctly exclude all __pycache__ directories
- **audiopro/process**: Optimize audio analysis output by directly serializing to MessagePack bytes

### Refactor

- **audiopro**: Remove `modified_date` from file metadata retrieval for cleaner output
- **audiopro**: Update package exports and remove unused modules for cleaner structure, now using a better folder structure
- **process**: Remove unused argparse import from process.py
- **audiopro/process**: Extract argument parsing logic to a separate module and update default output format to MessagePack
- **audiopro**: Move spectral bandwidth computation and type conversion to utils, optimize type conversion method

### Perf

- **core**: optimize audio processing pipeline
- **audiopro**: Combine empty/silent check with signal energy validation in audio loading process

## 0.4.0 (2025-01-29)

### BREAKING CHANGE

- The default output format is now MessagePack instead of the previously outputted JSON.
- This commit replaces the ThreadPoolExecutor with the multiprocessing module for improved performance. The aiofiles library is used for asynchronous file handling and metadata extraction. This means that processing is now done asynchronously, which can improve performance and scalability.

### Feat

- **audiopro**: Implement asynchronous file handling and metadata extraction using aiofiles for processing and metadata
- **audiopro/extractor**: Replace ThreadPoolExecutor with multiprocessing for improved performance
- **audiopro/extractor**: Dynamically calculate max workers based on audio data length
- **audiopro/monitor**: Enhance performance stats reporting with execution summary option
- **audiopro**: Add option to skip performance monitoring for reduced overhead, modify monitoring to be more efficient

### Fix

- **audiopro/extractor**: Reduce max workers to prevent resource exhaustion
- **.gitignore**: Update __pycache__ pattern to include nested directories

### Refactor

- **src/audiopro**: Change the default export format from JSON to MessagePack for better performance, update docs to reflect on the change

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
