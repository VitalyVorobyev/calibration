# Calibration Library - AI Assistant Instructions

## Project Overview

This is a C++ camera calibration library focusing on geometric computer vision algorithms. The library provides modular components for:
- Camera intrinsic/extrinsic calibration from planar targets
- Distortion correction and modeling
- Hand-eye calibration with bundle adjustment (single/multi-camera)
- Homography estimation and pose computation

## Architecture & Key Components

### Core Module Structure
```
include/calib/          # Public headers (lightweight APIs)
├── calib.h            # Main calibration entry point
├── handeye.h          # Hand-eye calibration algorithms
├── bundle.h           # Bundle adjustment for multi-camera systems
├── planarpose.h       # Planar target pose estimation
├── intrinsics.h       # Camera intrinsic parameter estimation
├── distortion.h       # Lens distortion models with concepts
├── cameramatrix.h     # Camera intrinsic parameter representation
└── serialization.h    # JSON I/O for all data structures

src/                   # Implementation files
├── *residual.h        # Ceres cost function implementations
└── observationutils.h # Internal utilities for optimization
```

### Data Flow Pattern
1. **Observations** → `PlanarView` (vector of `PlanarObservation`)
2. **Initial Estimates** → DLT/linear methods (e.g., `estimate_planar_pose`)
3. **Refinement** → Ceres-based bundle adjustment
4. **Results** → Structured result types with covariance and error metrics

## Development Patterns

### Template-Based Design
- Heavy use of templated functions for scalar type flexibility (`float`/`double`)
- C++20 concepts for distortion models: `distortion_model` concept
- Template specialization in headers for compile-time optimization

### JSON Serialization Convention
- Automatic serialization using Boost.PFR reflection for aggregate types
- Manual `to_json`/`from_json` for complex types in `serialization.h`
- All result structures are JSON-serializable for CLI tool integration

### Ceres Optimization Pattern
- Custom residual structs in `*residual.h` files (e.g., `handeyeresidual.h`)
- Variable projection for non-linear problems
- Consistent use of `Eigen::Isometry3d` for 3D transformations
- Robust loss functions (Huber) with configurable deltas

### Error Handling
- `std::optional` return types for algorithms that may fail
- Result structures include error metrics (`reprojection_error`, `view_errors`)
- Covariance matrices provided for uncertainty quantification

## Build & Test Workflow

### Standard Build Commands
```bash
# Configure (from repo root)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j2

# Test
cd build && ctest

# Run example
./build/examples/homography < input_data.txt
```

### Dependencies (managed via system packages or vcpkg)
- **Eigen3**: Core linear algebra (all algorithms heavily dependent)
- **Ceres**: Non-linear optimization engine
- **nlohmann::json**: JSON I/O throughout
- **CLI11**: Command-line parsing for `calib_app`
- **GoogleTest/GMock**: Testing framework
- **Boost.PFR**: Reflection for automatic JSON serialization

### Testing Patterns
- Each component has dedicated test file: `*_test.cpp`
- Tests use synthetic data generation with known ground truth
- Error tolerance testing with `EXPECT_LT(error, threshold)`
- Noise robustness testing in optimization algorithms

## Key Integration Points

### Command-Line Application (`app/`)
- JSON-driven configuration via `AppConfig` struct
- Task-based execution: `intrinsics`, `extrinsics`, `handeye`, etc.
- CLI11 for argument parsing with config file overrides

### Camera Model Abstraction
- `CameraMatrix` for intrinsic parameters (fx, fy, cx, cy, skew)
- Templated `normalize()`/`denormalize()` methods for coordinate transforms
- Support for various distortion models through concept-based design

### Bundle Adjustment Architecture
- `BundleObservation` links robot poses (`b_se3_g`) with camera views
- Multi-camera support via `camera_index` and relative `extrinsics`
- Configurable optimization via `BundleOptions` (what to optimize)

## Common Pitfalls & Project-Specific Notes

- **Coordinate Conventions**: World→camera transformations, Z=0 for planar targets
- **Template Instantiation**: Many algorithms require explicit template parameter specification
- **JSON Schema**: Use existing serialization in `serialization.h` rather than manual JSON handling
- **Ceres Integration**: Always use provided residual classes rather than writing raw cost functions
- **Error Handling**: Check `std::optional` returns before using results
- **Memory Layout**: Eigen matrices are column-major by default (affects Ceres parameter blocks)

## Documentation
Detailed algorithm documentation available in `doc/` directory with separate markdown files per component.
