# Calibration Library

[![CI](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/ci.yml)
[![Static Analysis](https://github.com/VitalyVorobyev/calibration/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration/actions/workflows/static-analysis.yml)
[![Test Coverage](https://codecov.io/gh/VitalyVorobyev/calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/VitalyVorobyev/calibration)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

Modern C++ utilities for camera calibration, geometric vision, and robot–sensor alignment.

## Highlights

- Modular targets exported as `calib::core`, `calib::models`, `calib::estimation`, `calib::pipeline`, `calib::utils`, and `calib::io`
- Intrinsic & extrinsic planar calibration with covariance estimation
- Homography and planar pose estimation (DLT, RANSAC, non-linear refinement)
- Hand-eye calibration starting from DLT seeds and refined via bundle adjustment
- Line-scan / laser plane calibration pipeline components
- JSON configuration I/O helpers for reproducible calibration sessions

## Repository Layout

```
include/calib/        Public headers organised by module
src/                  Library implementation (estimation/, pipeline/, utils/, ...)
apps/cli/             Command line front-ends (e.g. calib_cli)
apps/examples/        Tutorial executables and sample JSON schemas
tests/unit/           GoogleTest unit + integration suites
cmake/                Shared CMake modules and package config templates
```

Each module re-exports headers under `include/calib/` and can be consumed individually. A compatibility alias `calibration` points at the umbrella `calib::calib` interface target so downstream projects using the previous static library continue to link.

## Dependencies

- [Eigen3](https://eigen.tuxfamily.org/) – linear algebra
- [Ceres Solver](http://ceres-solver.org/) – non-linear optimisation
- [nlohmann_json](https://github.com/nlohmann/json) – JSON parsing/serialisation
- [CLI11](https://github.com/CLIUtils/CLI11) – CLI argument parsing (used by apps/examples)
- [GoogleTest](https://github.com/google/googletest) – unit testing (enabled when `CALIB_BUILD_TESTS=ON`)

## Building

CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `CALIB_BUILD_APPS` | `ON` | Build CLI tools under `apps/cli` |
| `CALIB_BUILD_EXAMPLES` | `ON` | Build tutorial executables in `apps/examples` |
| `CALIB_BUILD_TESTS` | `ON` | Enable GoogleTest targets in `tests/unit` |
| `CALIB_ENABLE_COVERAGE` | `OFF` | Enable GCov/LLVM coverage instrumentation |
| `CALIB_ENABLE_WERROR` | `OFF` | Treat warnings as errors on supported toolchains |

### Linux / macOS

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Quick homography sanity check:

```bash
./build/apps/examples/calib_example_homography <<'EOF'
4
0   0    10 20
100 0    110 18
100 50   120 70
0   50    8 72
EOF
```

### Windows (vcpkg toolchain)

```powershell
git clone https://github.com/microsoft/vcpkg $env:USERPROFILE/vcpkg
& $env:USERPROFILE/vcpkg/bootstrap-vcpkg.bat
& $env:USERPROFILE/vcpkg/vcpkg.exe install ceres eigen3 nlohmann-json gtest cli11 --triplet x64-windows

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release -j2
```

### Installing

```bash
cmake --install build --prefix /desired/prefix
```

The install exports `calibTargets.cmake` and a `calibConfig.cmake` package so downstream projects can simply call `find_package(calib CONFIG REQUIRED)` and link against the `calib::` targets they need.

## Using the Library

Most workflows live in `calib::estimation` and `calib::pipeline`:

```cpp
#include <calib/estimation/handeye.h>

HandeyeOptions opts;
auto result = calib::optimize_handeye(base_gripper_poses,
                                      camera_target_poses,
                                      initial_guess,
                                      opts);
if (result.success) {
    std::cout << result.report << '
'
              << result.g_se3_c.matrix() << std::endl;
}
```

### CLI utility

`calib_cli` executes declarative calibration jobs from JSON descriptions:

```bash
./build/apps/cli/calib_cli --config configs/planar_intrinsics.json --task intrinsics --output result.json
```

### Examples

Tutorial executables reside in `apps/examples/`. The planar intrinsics walkthrough expects a configuration JSON and detector output following the schema shipped alongside the example:

```bash
./build/apps/examples/calib_example_planar_intrinsics \
  --config apps/examples/planar_intrinsics_config.json \
  --features path/to/planar_detections.json \
  --output planar_intrinsics_result.json
```

The resulting report lists accepted views, linear initialisation, refined statistics, and persists the outputs under a `calibrations` array to support multi-camera expansion.

## Quality Gates

- **Static analysis** – clang-tidy / cppcheck via CI (`make lint` locally)
- **Unit tests** – GoogleTest suites (`ctest` with `CALIB_BUILD_TESTS=ON`)
- **Formatting** – clang-format helpers (`make format`)
- **Coverage** – `make coverage` when `CALIB_ENABLE_COVERAGE=ON`

## Documentation

[Doxygen](https://www.doxygen.nl/) powers the API reference. Generate and serve docs with:

```bash
./generate_docs.sh            # plain generation
./generate_docs.sh --open     # open in a browser
./serve_docs.py --port 8080   # lightweight HTTP server
cmake --build build --target doc
```

The generated manual groups content by functional modules (calibration, distortion, optimisation, pipelines) and cross-links tutorial executables with the relevant headers.

## License

Distributed under the MIT License. See [LICENCE](LICENCE) for details.

## Contributing

Bug reports, feature proposals, and pull requests are welcome. Open an issue for larger changes so the design can be discussed before implementation starts.
