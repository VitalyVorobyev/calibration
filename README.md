# Calibration Library

[![Build Status](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)
[![Static Analysis](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg?job=static-analysis)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/VitalyVorobyev/calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/VitalyVorobyev/calibration)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)alibration Library

[![Build Status](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)

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
sudo apt install -y cmake ninja-build libeigen3-dev libceres-dev nlohmann-json3-dev libgtest-dev libgmock-dev libboost-dev cli11
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

std::vector<Eigen::Affine3d> base_T_gripper = ...;
std::vector<Eigen::Affine3d> camera_T_target = ...;
Eigen::Affine3d guess = ...; // initial gripper->camera transform

HandeyeOptions opts;
auto result = optimize_handeye(base_T_gripper, camera_T_target, guess, opts);
if (result.success) {
    std::cout << result.report << std::endl;
    std::cout << "Estimated transform:\n" << result.g_T_c.matrix() << std::endl;
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

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
