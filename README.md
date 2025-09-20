# Calibration Library

[![CI](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)
[![Static Analysis](https://github.com/VitalyVorobyev/calibration/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/static-analysis.yml)
[![Test Coverage](https://codecov.io/gh/VitalyVorobyev/calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/VitalyVorobyev/calibration)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

A C++ library for camera calibration and vision-related geometric transformations.

## Features

- Intrinsic and extrinsic calibration from planar targets
- Hand-eye calibration with bundle adjustment
- Homography and planar pose refinement utilities
- Support for multiple camera models, including Scheimpflug projection
- Unified optimisation interface with covariance estimation
- JSON configuration import/export
- Automatic JSON serialization for aggregate types

## Dependencies

- Eigen3: Linear algebra library
- Ceres: Non-linear optimization
- nlohmann-json: JSON parsing and serialization
- CLI11: Command-line argument parsing
- GoogleTest & GoogleMock: Unit testing frameworks
- Boost.PFR: Header-only reflection for aggregates

## Build

### Linux and macOS

Install the build dependencies using your system package manager.

#### Ubuntu

```bash
sudo apt update
sudo apt install -y cmake ninja-build libeigen3-dev libceres-dev nlohmann-json3-dev libgtest-dev libgmock-dev libboost-dev libcli11-dev
```

#### macOS

```bash
brew update
brew install cmake ninja eigen ceres-solver nlohmann-json googletest boost cli11
```

Configure and build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

Run a smoke test:

```bash
./build/examples/homography <<EOF
4
0   0    10 20
100 0    110 18
100 50   120 70
0   50   8  72
EOF
```

### Windows

Install dependencies with [vcpkg](https://github.com/microsoft/vcpkg):

```powershell
git clone https://github.com/microsoft/vcpkg $env:USERPROFILE\vcpkg
& $env:USERPROFILE\vcpkg\bootstrap-vcpkg.bat
& $env:USERPROFILE\vcpkg\vcpkg.exe install ceres eigen3 nlohmann-json gtest boost-pfr cli11 --triplet x64-windows
```

Configure and build:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release -j2
```

## Usage

The library exposes lightweight C++ APIs with a common optimisation interface.
Typical entry points include:

- `optimize_intrinsics` – refine camera intrinsics and poses from planar views
- `optimize_extrinsics` – calibrate camera poses relative to a target
- `optimize_handeye` – refine gripper→camera transform from motion pairs
- `optimize_planar_pose` – estimate single planar pose with distortion

Example for hand‑eye calibration:

```cpp
#include <calib/handeye.h>
using namespace calib;

std::vector<Eigen::Isometry3d> base_se3_gripper = ...;
std::vector<Eigen::Isometry3d> camera_se3_target = ...;
Eigen::Isometry3d guess = ...; // initial gripper->camera transform

HandeyeOptions opts;
auto result = optimize_handeye(base_se3_gripper, camera_se3_target, guess, opts);
if (result.success) {
    std::cout << result.report << std::endl;
    std::cout << "Estimated transform:\n" << result.g_se3_c.matrix() << std::endl;
}
```

### Command-line tool

A small `calib_app` utility is available for running calibrations from JSON
files. It accepts the path to a config file and optional overrides:

```bash
calib_app --config input.json [--task intrinsics] [--output result.json]
```

The config file specifies the calibration input and task type, while the
command-line options provided by CLI11 allow overriding the task mode or the
output file.

### ChArUco intrinsics example

An end-to-end intrinsic calibration example is available under `examples/`. It
takes a high-level config file together with ChArUco detections that follow the
schema documented in `examples/charuco_features_schema.json`:

```bash
cmake --build build -j2
./build/examples/charuco_intrinsics \
  --config examples/charuco_intrinsics_config.json \
  --features charuco_detection_cam1.json \
  --output charuco_cam1_intrinsics.json
```

During the run the executable reports which views were accepted, the initial
linear estimate and the refined reprojection statistics. The JSON report it
writes groups results by calibration type so it can be extended with additional
cameras or future extrinsic/hand-eye calibrations without changing the schema.

The sample config highlights the most relevant knobs:

- `point_scale` rescales the board coordinates if the detector reports values in
  centimetres or arbitrary units.
- `auto_center_points` recentres the board origin before scaling; supply
  `point_center` to override the automatically detected midpoint.
- `homography_ransac` mirrors the options exposed by the library for robust
  homography estimation.
- `fixed_distortion_indices`/`fixed_distortion_values` let you freeze individual
  Brown-Conrady coefficients (indexed as `[k1, k2, k3, p1, p2]`) when you want
  to assume zero distortion or keep prior values.

The detections file consumed by `--features` is validated against
`examples/charuco_features_schema.json` and captures:

- the image directory, detector metadata (`feature_type`, `algo_version`, `params_hash`)
- per-image entries (`file`, detected `count`) and the list of ChArUco corners
- per-corner measurements with image pixels (`x`, `y`), marker `id`, and board-space
  coordinates (`local_x`, `local_y`, `local_z`)

The resulting calibration report stores all calibration runs under a single
`calibrations` array so that multiple cameras or additional task types can be
aggregated in future extensions without breaking downstream tooling.

## Code Quality

This project maintains high code quality through:

- **Static Analysis**: clang-tidy and cppcheck integration in CI
- **Test Coverage**: Comprehensive unit tests with coverage reporting
- **Code Formatting**: Consistent style with clang-format
- **Pre-commit Hooks**: Automated quality checks before commits

### Running Static Analysis Locally

```bash
# Install tools
sudo apt install clang-tidy cppcheck clang-format

# Run static analysis
make lint

# Format code
make format

# Generate coverage report
make coverage
```

## Documentation

### Generating Documentation

This project uses [Doxygen](https://www.doxygen.nl/) to generate comprehensive API documentation. The documentation includes:

- Complete API reference with class diagrams
- Usage examples and tutorials
- Module organization and dependency graphs
- Detailed descriptions of algorithms and mathematical concepts

#### Prerequisites

Install Doxygen and Graphviz:

```bash
# Ubuntu/Debian
sudo apt-get install doxygen graphviz

# macOS
brew install doxygen graphviz

# Windows
# Download from https://www.doxygen.nl/download.html
```

#### Generate Documentation

```bash
# Generate documentation
./generate_docs.sh

# Generate and automatically open in browser
./generate_docs.sh --open

# Alternative: use CMake
cmake --build build --target doc
```

#### View Documentation

```bash
# Serve documentation locally (recommended)
./serve_docs.py

# Alternative: serve on specific port
./serve_docs.py --port 8080

# Or manually open the HTML file
open doc/doxygen/html/index.html
```

The generated documentation includes:

- **Main Page**: Overview and getting started guide
- **Modules**: Organized by functionality (calibration, distortion, optimization, etc.)
- **Classes**: Complete API reference with inheritance diagrams
- **Files**: Source code browser with syntax highlighting
- **Examples**: Code examples and usage patterns

### Documentation Structure

- **Camera Calibration**: Intrinsic and extrinsic parameter estimation
- **Distortion Correction**: Lens distortion models and correction algorithms  
- **Hand-Eye Calibration**: Robot-camera calibration methods
- **Geometric Transforms**: Homography and planar pose estimation
- **Optimization Framework**: Ceres-based bundle adjustment and covariance estimation

For detailed documentation, see the generated HTML documentation or browse the [doc](doc/) directory.

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
