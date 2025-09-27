/**
 * @file linescan.h
 * @brief Simple facade for line-scan laser plane calibration
 */

#pragma once

// std
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/estimation/linear/linescan.h"
#include "calib/models/pinhole.h"

namespace calib::pipeline {

/**
 * @brief Result of a line-scan calibration run.
 */
struct LinescanCalibrationRunResult final {
    bool success = false;              ///< True if calibration completed
    std::size_t used_views = 0;        ///< Number of views used
    LineScanCalibrationResult result;  ///< Fitted plane and stats
};

/**
 * @brief Facade wrapping linear laser plane calibration for a single camera.
 *
 * Converts a Brown-Conrady camera to a dual-distortion camera, invokes the
 * linear calibrate_laser_plane routine and packages the result for pipelines
 * or examples.
 */
class LinescanCalibrationFacade final {
  public:
    [[nodiscard]] auto calibrate(const PinholeCamera<BrownConradyd>& camera,
                                 const std::vector<LineScanView>& views) const
        -> LinescanCalibrationRunResult;
};

}  // namespace calib::pipeline

