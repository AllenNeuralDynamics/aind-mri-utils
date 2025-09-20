# aind-mri-utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-ruff-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-98.2%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-63%25-red?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)
![MyPy](https://img.shields.io/badge/mypy-typed-blue)

MRI utilities toolkit for neuroscience experiment planning developed by Allen Institute for Neural Dynamics.

## Installation

```bash
pip install aind-mri-utils
```

For development:
```bash
git clone https://github.com/AllenNeuralDynamics/aind-mri-utils
cd aind-mri-utils
uv sync --group dev
```

## Usage

```python
# Arc angle conversions for probe positioning systems
from aind_mri_utils.arc_angles import vector_to_arc_angles, arc_angles_to_vector
probe_vector = [0.0, 0.5, -0.866]  # 30° from vertical
arc_angles = vector_to_arc_angles(probe_vector)  # → (30.0, 0.0)

# Reticle calibration from measurement data
from aind_mri_utils.reticle_calibrations import fit_rotation_params_from_parallax
calibration_file = "path/to/parallax_measurements.xlsx"
rotation_params = fit_rotation_params_from_parallax(calibration_file)

# Chemical shift correction for MRI images
from aind_mri_utils.chemical_shift import compute_chemical_shift
import SimpleITK as sitk
mri_image = sitk.ReadImage("brain_scan.nii")
shift_vector = compute_chemical_shift(mri_image, ppm=3.5, mag_freq=599.0)

# 3D geometric measurements
from aind_mri_utils.measurement import find_circle, dist_point_to_line
circle_center, radius = find_circle(measurement_points)
distance = dist_point_to_line(point, line_start, line_end)

# Medical image I/O
from aind_mri_utils.file_io import read_dicom, write_nii
dicom_volume = read_dicom("dicom_folder/")
write_nii(processed_volume, "output.nii")
```

## Contributing

### Development workflow

```bash
# Setup development environment
uv sync --group dev

# Run linting and formatting
uv run ruff check
uv run ruff format

# Run type checking
uv run mypy src/

# Run tests with coverage
uv run pytest

# Run all checks
./scripts/run_linters_and_checks.sh --checks
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To build documentation:
```bash
uv sync --group docs
uv run mkdocs serve
```
