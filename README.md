# Calibration Library

[![Build Status](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)

A C++ library for camera calibration and vision-related geometric transformations.

## Features

- Camera intrinsic and extrinsic calibration
- Distortion correction
- Stereo camera calibration
- Support for various camera models
- JSON configuration import/export

## Dependencies

- Eigen3: Linear algebra library
- Ceres: Non-linear optimization
- nlohmann-json: JSON parsing and serialization
- GoogleTest & GoogleMock: Unit testing frameworks

## Installation

### Ubuntu

```bash
sudo apt install libeigen3-dev libceres-dev nlohmann-json3-dev libgtest-dev libgmock-dev
```
These packages install both GoogleTest and GoogleMock, which are needed to build and run the unit tests.

### Other platforms

For other platforms, please install the equivalent dependencies using your system's package manager.

## Build

### Linux and macOS

Install the build dependencies using your system package manager.

#### Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build libeigen3-dev libceres-dev nlohmann-json3-dev
```

#### macOS

```bash
brew update
brew install cmake ninja eigen ceres-solver nlohmann-json
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
& $env:USERPROFILE\vcpkg\vcpkg.exe install ceres eigen3 nlohmann-json --triplet x64-windows
```

Configure and build:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release -j2
```

Run a smoke test:

```powershell
build\examples\homography.exe <<'EOF'
4
0   0    10 20
100 0    110 18
100 50   120 70
0   50   8  72
EOF
```

## Usage

```cpp
#include <calib/camera.h>

// Example code will go here
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
