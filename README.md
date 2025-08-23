# Calibration Library

[![Build Status](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)

A C++ library for camera calibration and vision-related geometric transformations.

## Features

- Camera intrinsic and extrinsic calibration
- Distortion correction
- Stereo camera calibration
- Support for various camera models
- JSON configuration import/export
- Hand-eye calibration with reference camera support

## Dependencies

- Eigen3: Linear algebra library
- Ceres: Non-linear optimization
- nlohmann-json: JSON parsing and serialization
- GoogleTest & GoogleMock: Unit testing frameworks

## Build

### Linux and macOS

Install the build dependencies using your system package manager.

#### Ubuntu

```bash
sudo apt update
sudo apt install -y cmake ninja-build libeigen3-dev libceres-dev nlohmann-json3-dev libgtest-dev libgmock-dev
```

#### macOS

```bash
brew update
brew install cmake ninja eigen ceres-solver nlohmann-json googletest
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
& $env:USERPROFILE\vcpkg\vcpkg.exe install ceres eigen3 nlohmann-json gtest --triplet x64-windows
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

### Hand-Eye Calibration

The library models hand-eye calibration around a *reference camera*.  The
reference camera uses the optimized hand–eye pose directly, while additional
cameras are expressed via extrinsic transforms relative to this reference.  A
simple call looks like:

```cpp
#include "calibration/handeye.h"

using namespace vitavision;

HandEyeOptions opts;
opts.optimize_extrinsics = true;            // also refine per-camera extrinsics
std::vector<Eigen::Affine3d> initial_ext;    // size = num_cams - 1
HandEyeResult res = calibrate_hand_eye(initial_ext, opts);
// res.extrinsics[0] is identity for the reference camera
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
