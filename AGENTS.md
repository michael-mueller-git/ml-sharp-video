# Development Guidelines for ML-SHARP

## Build and Test Commands

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-webui.txt
pip install -e .
```

### Linting and Type Checking
```bash
# Pre-commit hooks (recommended before committing)
pre-commit run --all-files

# Individual tools
ruff check src/                # Linting
ruff format src/               # Formatting
mypy src/                      # Type checking with Pyright
```

### Running Tests
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_filename.py

# Run a specific test function
pytest tests/test_filename.py::test_function_name

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src
```

### Running the Application
```bash
# CLI prediction
sharp predict -i /path/to/input/images -o /path/to/output/gaussians

# CLI rendering (CUDA GPU required)
sharp render -i /path/to/gaussians -o /path/to/renderings

# Start WebUI
python webui.py
# or use the launcher scripts:
./run_webui.sh     # Linux/Mac
run_webui.bat      # Windows
```

## Code Style Guidelines

### Imports
- Use `from __future__ import annotations` at the top of all Python files.
- Standard library imports first, then third-party packages, then local modules.
- Use absolute imports from `sharp` package (e.g., `from sharp.models import Predictor`).
- Group imports with blank lines between groups.

### Formatting
- Line length: 100 characters (configured in `pyproject.toml`).
- Use Ruff formatter for consistent code style.
- Use Google-style docstrings for all public functions and classes.

### Type Hints
- Use type hints for all function parameters and return values.
- Use `typing.Literal`, `typing.NamedTuple`, and `dataclasses.dataclass` for type safety.
- Use `torch.Tensor`, `np.ndarray`, and other explicit types.
- Use `Optional[X]` or `X | None` for optional types.
- Use `# type: ignore` sparingly and with justification.

### Naming Conventions
- Classes: `PascalCase` (e.g., `RGBGaussianPredictor`, `GaussianComposer`).
- Functions and variables: `snake_case` (e.g., `predict_image`, `gaussian_base_values`).
- Constants: `UPPER_CASE` (e.g., `DEFAULT_MODEL_URL`).
- Private members: Single underscore prefix (e.g., `_helper_function`).
- Modules and packages: `snake_case` (e.g., `sharp/models/params.py`).

### Error Handling
- Use Python exceptions for error cases.
- Use specific exception types (e.g., `ValueError`, `KeyError`, `RuntimeError`).
- Include descriptive error messages that explain the context.
- Log warnings using the `logging` module for recoverable issues.
- Use `torch.no_grad()` decorator for inference code.

### Logging
- Use the `logging` module with `logging.getLogger(__name__)`.
- Use appropriate log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- Log user-facing information at `INFO` level.
- Log detailed debugging information at `DEBUG` level.
- Configure logging in CLI entry points with `-v` flag for verbose output.

### Documentation
- All public functions and classes require docstrings.
- Use Google-style docstrings with Args, Returns, and Notes sections.
- Include type information in docstrings for complex types.
- Add `Note:` sections for important implementation details.
- Document any non-obvious behavior or constraints.

### PyTorch Best Practices
- Use `nn.Module` for all neural network components.
- Call `model.eval()` and use `torch.no_grad()` for inference.
- Move tensors to the correct device (`cuda`, `cpu`, `mps`).
- Use `torch.load(..., weights_only=True)` for loading checkpoints.
- Prefer `torch.Tensor` operations over NumPy for GPU compatibility.
- Use proper dtype handling (`torch.float32`, etc.).

### Data Structures
- Use `@dataclasses.dataclass` for configuration parameters.
- Use `typing.NamedTuple` for immutable data containers.
- Use `Literal` types for constrained string values.
- Define type aliases at module level for complex types.

### External Code
- Code in `src/sharp/external/` may have different style requirements.
- Exclude external code from linting in `.pre-commit-config.yaml`.