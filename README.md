# Calibration Library

[![Build Status](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/ci/build.yml)

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

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install" ..
cmake --build . -j4
```

To install:

```bash
cmake --build . --target install
```

## Usage

```cpp
#include <calib/camera.h>

// Example code will go here
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
