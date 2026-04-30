# AIND MRI Utils

![CI](https://github.com/AllenNeuralDynamics/aind-mri-utils/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-mri-utils)](https://pypi.org/project/aind-mri-utils/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

MRI utilities library for aind teams.

## Installation

```bash
pip install aind-mri-utils
```

For development, clone the repository and run:
```bash
git clone https://github.com/AllenNeuralDynamics/aind-mri-utils
cd aind-mri-utils
uv sync
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

## Development

Please test your changes using the full linting and testing suite:

```bash
./scripts/run_linters_and_checks.sh -c
```

Or run individual commands:
```bash
uv run --frozen ruff format          # Code formatting
uv run --frozen ruff check           # Linting
uv run --frozen mypy                 # Type checking
uv run --frozen interrogate -v src   # Documentation coverage
uv run --frozen codespell --check-filenames  # Spell checking
uv run --frozen pytest --cov aind_mri_utils # Tests with coverage
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style commit messages, roughly:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci-call.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
