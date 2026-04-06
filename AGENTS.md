# Project Guidelines

## Code Style
- Follow the repository hooks and lint stack in `.pre-commit-config.yaml`.
- For Python, rely on `ruff`, `ruff-format`, `isort`, and NumPy-style docstrings configured in `pyproject.toml`.
- Keep imports backend-light at module top level. The repo bans module-level heavy backend imports (for example `torch`, `tensorflow`, `jax`, `paddle`) in many paths; see `[tool.ruff.lint.flake8-tidy-imports]` in `pyproject.toml`.
- Respect existing license header conventions; pre-commit inserts headers automatically.

## Architecture
- `deepmd/`: primary Python package, including CLI entrypoints and backend-specific implementations (`tf`, `pt`, `jax`, `pd`).
- `backend/`: build backend helpers used by `scikit-build-core` to detect TensorFlow/PyTorch and generate CMake flags.
- `source/`: C/C++ core libraries, operators, and interfaces (`api_c`, `api_cc`, `lmp`, `ipi`, `gmx`).
- `source/tests/`: main test tree for Python/C++ integration and backend behavior.
- `doc/`: Sphinx documentation and developer guides.

## Build And Test
- Preferred local setup (matches CI setup logic):
  - `uv venv venv && source venv/bin/activate`
  - `uv pip install --group pin_tensorflow_cpu --group pin_pytorch_cpu --torch-backend cpu`
  - `uv pip install -e .[cpu,test]`
- Fast validation:
  - `dp --version`
  - `python -m deepmd -h`
  - `pytest source/tests -k <pattern>`
- Full Python test sweep used in CI:
  - `pytest --cov=deepmd source/tests`
- Build C++ interfaces when needed:
  - `source/install/build_cc.sh`
  - Enable related features through environment variables before build (for example `DP_ENABLE_PYTORCH`, `DP_ENABLE_IPI`, `DP_LAMMPS_VERSION`, `DP_VARIANT`).
- Build docs from `doc/`:
  - `make html`

## Conventions
- This project is multi-backend. Avoid introducing unconditional backend assumptions in shared code paths.
- Check environment-driven behavior before changing build/runtime logic:
  - Runtime/threading/precision: `deepmd/env.py`
  - Build flags from env vars: `backend/read_env.py`
- For backend enablement and build options, prefer existing `DP_*` variables instead of new ad-hoc flags.
- Keep changes scoped to the touched backend unless a cross-backend refactor is explicitly required.

## Pitfalls
- `pip install -e .` triggers CMake via `scikit-build-core`; missing backend dependencies can look like generic build failures.
- `DP_VARIANT` controls CPU/CUDA/ROCm build mode in the backend logic; mismatched environment leads to confusing CMake errors.
- Tests are under `source/tests/` (not a top-level `tests/` directory), so default test discovery assumptions may miss coverage.

## Reference Docs (Link, Don’t Embed)
- Project overview and top-level structure: `README.md`
- Contribution workflow: `CONTRIBUTING.md`
- Build configuration and lint/test settings: `pyproject.toml`
- Backend and environment variables: `doc/backend.md`, `doc/env.md`
- Source build and CMake details: `doc/install/install-from-source.md`, `doc/development/cmake.md`
- CI/testing behavior: `.github/workflows/test_python.yml`, `.github/workflows/build_cc.yml`, `doc/development/cicd.md`

## High-Value Paths
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `deepmd/env.py`
- `backend/read_env.py`
- `backend/dp_backend.py`
- `source/CMakeLists.txt`
- `source/tests/`
- `deepmd/pt/model/model/`
