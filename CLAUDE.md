# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run all checks (formatting, linting, type checking, tests)
./scripts/run_linters_and_checks.sh -c

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a single test by name
uv run pytest -k "test_name"

# Formatting
uv run ruff format

# Linting
uv run ruff check
uv run ruff check --fix

# Type checking
uv run mypy

# Spell checking
uv run codespell --check-filenames

```

Always use `uv run` to execute commands, `uv add` to add dependencies, and `uv sync` to set up the environment. Never use bare `pip` or `python`.

## Architecture

This is a Python package using a `src/` layout. Source code lives in `src/aind_mri_utils/`, tests in `tests/`.

- Build system: hatchling
- Formatting/linting: ruff (line length 120, numpy docstring convention)
- Testing: pytest with coverage reporting
- Type checking: mypy (strict mode)
- Versioning: commitizen (semantic versioning via conventional commits)
