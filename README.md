# Audio Process

Audio Processing and Performance Monitoring Tool.

## Installation

Ensure you have Python installed. Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the `process.py` script with the input audio file and desired output JSON file as arguments.

The `process.py` script is designed to dynamically handle audio file analysis. It accepts command-line arguments for input and output file paths, allowing flexibility in processing different audio files without modifying the script. The tool leverages parallel processing to efficiently extract features from audio frames, ensuring optimal performance even with large files. By utilizing multi-threading, the feature extraction process is accelerated, making the analysis both faster and more resource-efficient.

```bash
python process.py <input_file> <output_file>
```

### Example

```bash
python process.py love.mp3 analysis_results.json
```

## Features

- Analyzes audio files to extract tempo, beats, and various audio features.
- Monitors CPU and memory usage during processing.
- Saves analysis results in a structured JSON format.

## Commit Message Convention

This project uses the following commit message convention (following the Commitizen format):

```
<type>(<scope>): <subject>
```

Where `<type>` is one of the following:

### Code Changes
- `feat`: A new feature.
- `fix`: A bug fix.
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `perf`: A code change that improves performance.
- `style`: Non-functional changes (e.g., white-space, formatting).

### Testing
- `test`: Adding or updating tests.
- `hotfix`: A quick fix for a critical issue.

### Tooling and Configuration
- `chore`: Changes to the build process or auxiliary tools and libraries.
- `build`: Changes to the build system or external dependencies.
- `ci`: Changes to the CI configuration files and scripts.
- `config`: Configuration changes.
- `update`: Updating dependencies.

### Documentation
- `docs`: Documentation changes.

### Project and Releases
- `init`: Initial commit.
- `release`: Release a new version.
- `breaking`: A breaking change.

### File Operations
- `remove`: Remove code or files.
- `move`: Move or rename code or files.
- `rename`: Rename code or files.
- `clean`: Clean up code or files.

### Deployment
- `deploy`: Deployment changes.

### Miscellaneous
- `security`: Fixing security issues.
- `merge`: Merge changes.
- `wip`: Work in progress.
- `revert`: Revert a previous commit.

Make sure to use the command `git cz` to commit changes using the Commitizen format.
Make sure to use the command `cd bump` to bump the version number.


## License

This project is currently closed source.