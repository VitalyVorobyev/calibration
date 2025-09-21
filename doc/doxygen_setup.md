# Doxygen Documentation Setup Summary

This document summarizes the Doxygen documentation setup for the Calibration Library.

## Files Created and Modified

### 1. Documentation Configuration

- **`Doxyfile`**: Main Doxygen configuration file
  - Configured for C++20 features and modern Doxygen
  - Includes comprehensive project settings
  - Excludes test files and build artifacts
  - Enables HTML output with navigation tree
  - Configured for Graphviz integration

- **`doc/mainpage.dox`**: Main documentation page
  - Comprehensive project overview
  - Feature descriptions and architecture
  - Getting started examples
  - Module organization with @defgroup tags

### 2. Build Integration

- **`CMakeLists.txt`**: Added documentation target
  - Optional Doxygen dependency detection
  - Custom target `doc` for building documentation
  - Integration with existing build system

### 3. Scripts and Utilities

- **`generate_docs.sh`**: Documentation generation script
  - Checks for Doxygen installation
  - Creates output directories
  - Provides helpful feedback and options
  - Optional browser opening

- **`serve_docs.py`**: Local documentation server
  - Python HTTP server for viewing docs
  - Automatic browser opening
  - Configurable port settings
  - Error handling and user feedback

### 4. Enhanced Header Documentation

Enhanced the following key header files with comprehensive Doxygen comments:

- **`include/calib/models/distortion.h`**:
  - File-level documentation with @file and @ingroup
  - Detailed concept documentation for `distortion_model`
  - Function and struct documentation with @brief and @param

- **`include/calib/models/pinhole.h`**:
  - Complete camera model documentation
  - Template parameter descriptions
  - Method documentation with examples

- **`include/calib/estimation/handeye.h`**:
  - Hand-eye calibration algorithm documentation
  - Motion pair structure descriptions
  - Parameter filtering explanations

- **`include/calib/estimation/homography.h`**:
  - Homography estimation documentation
  - Algorithm descriptions and usage notes

- **`include/calib/estimation/intrinsics.h`**:
  - Intrinsic calibration method documentation
  - Linear and non-linear approach descriptions

### 5. Documentation Structure

The documentation is organized into the following modules:

1. **Camera Calibration** (`@defgroup camera_calibration`)
   - Intrinsic and extrinsic parameter estimation
   - Multi-view calibration
   - Covariance estimation

2. **Distortion Correction** (`@defgroup distortion_correction`)
   - Lens distortion models
   - Forward and inverse mapping
   - C++20 concept-based interface

3. **Hand-Eye Calibration** (`@defgroup hand_eye_calibration`)
   - AX=XB problem solving
   - Bundle adjustment methods
   - Multi-camera systems

4. **Geometric Transformations** (`@defgroup geometric_transforms`)
   - Homography estimation
   - Planar pose computation
   - DLT algorithms

5. **Optimization Framework** (`@defgroup optimization`)
   - Ceres-based optimization
   - Custom residual functions
   - Robust loss functions

### 6. Configuration Updates

- **`.gitignore`**: Added documentation output directories
- **`README.md`**: Added comprehensive documentation section
  - Generation instructions
  - Viewing options
  - Structure overview

## Usage

### Generate Documentation

```bash
# Simple generation
./generate_docs.sh

# Generate and open in browser
./generate_docs.sh --open

# Using CMake
cmake --build build --target doc
```

### View Documentation

```bash
# Serve locally (recommended)
./serve_docs.py

# Custom port
./serve_docs.py --port 8080

# Don't open browser automatically
./serve_docs.py --no-browser
```

### Access Documentation

After generation, documentation is available at:
- **Local file**: `doc/doxygen/html/index.html`
- **Local server**: `http://localhost:8080` (when using serve script)

## Features

### Documentation Quality
- ✅ Comprehensive API coverage
- ✅ Module organization with groups
- ✅ Cross-references and inheritance diagrams
- ✅ Source code browser
- ✅ Search functionality
- ✅ Mobile-responsive design

### Build Integration
- ✅ CMake integration
- ✅ Optional dependency handling
- ✅ Build target creation
- ✅ CI/CD ready

### User Experience
- ✅ Easy generation scripts
- ✅ Local development server
- ✅ Automatic browser opening
- ✅ Clear error messages
- ✅ Progress feedback

### Content Quality
- ✅ Mathematical concepts explained
- ✅ Usage examples provided
- ✅ Parameter documentation
- ✅ Return value descriptions
- ✅ Exception specifications

## Maintenance

### Adding Documentation to New Files

1. Add file-level documentation:
```cpp
/**
 * @file filename.h
 * @brief Brief description
 * @ingroup appropriate_group
 */
```

2. Document classes and functions:
```cpp
/**
 * @brief Brief description
 * @param param_name Parameter description
 * @return Return value description
 */
```

3. Use appropriate groups with `@ingroup` tags

### Updating Configuration

- Modify `Doxyfile` for configuration changes
- Update `doc/mainpage.dox` for content changes
- Regenerate with `./generate_docs.sh`

## Notes

- Documentation warnings are displayed during generation
- Graphviz is required for class diagrams
- Generated documentation is excluded from version control
- Server script requires Python 3.x
- HTML output is optimized for modern browsers
