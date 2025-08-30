# Calibration Library

[![Build Status](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)

A C++ library for camera calibration and vision-related geometric transformations.

## Features

- Camera intrinsic and extrinsic calibration from planar targets
- Distortion correction utilities
- Hand-eye calibration with bundle adjustment (single or multi-camera)
- Support for various camera models
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

Run a smoke test:

```powershell
build\examples\homography.exe <<EOF
4
0   0    10 20
100 0    110 18
100 50   120 70
0   50   8  72
EOF
```

## Usage

The library exposes lightweight C++ APIs.  The most common entry points are:

- `calibrate_camera_planar` – full camera calibration from several planar views.
- `optimize_planar_pose` – single planar pose refinement.
- `calibrate_hand_eye` – bundle adjustment for hand‑eye calibration.

Example for hand‑eye calibration:

```cpp
#include <calibration/handeye.h>
using namespace calib;

std::vector<HandEyeObservation> observations = ...; // fill with data
std::vector<CameraMatrix> intrinsics = ...;         // initial intrinsics
Eigen::Affine3d hand_eye_guess = ...;               // gripper->reference camera
std::vector<Eigen::Affine3d> extrinsics = ...;      // reference->camera (for cams>0)

HandEyeOptions opts; // customise which parameters to optimise
HandEyeResult result = calibrate_hand_eye(observations, intrinsics,
                                          hand_eye_guess, extrinsics,
                                          Eigen::Affine3d::Identity(), opts);
```

The first camera in `intrinsics` is treated as the reference camera.  The
`hand_eye_guess` specifies the gripper pose relative to this camera.  For any
additional cameras, provide their poses relative to the reference camera via
`extrinsics`.

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
