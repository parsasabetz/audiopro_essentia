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
