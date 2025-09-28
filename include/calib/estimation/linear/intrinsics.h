/** @brief Linear intrinsic estimation interfaces */

#pragma once

#include <Eigen/Geometry>
#include <optional>
#include <vector>

#include "calib/estimation/common/intrinsics_utils.h"
#include "calib/estimation/common/ransac.h"      // RansacOptions
#include "calib/estimation/linear/homography.h"  // HomographyResult
#include "calib/estimation/linear/planarpose.h"  // PlanarObservation/View
#include "calib/models/cameramodel.h"
#include "calib/models/distortion.h"
#include "calib/models/pinhole.h"

namespace calib {

/**
 * @brief Options for linear intrinsic estimation from planar views
 * @ingroup camera_calibration
 *
 * Controls bounds and whether skew is estimated when computing the
 * initial camera matrix from a collection of @ref PlanarView observations.
 */
struct IntrinsicsEstimateOptions final {
    std::optional<CalibrationBounds> bounds = std::nullopt;         ///< Optional parameter bounds
    std::optional<RansacOptions> homography_ransac = std::nullopt;  ///< Optional RANSAC opts
    bool use_skew = false;                                          ///< Estimate skew parameter
};

struct ViewEstimateData final {
    size_t view_index = 0;
    Eigen::Isometry3d c_se3_t = Eigen::Isometry3d::Identity();
    // Diagnostics
    HomographyResult homography;
    double forward_rms_px = 0.0;
};

/**
 * @brief Result of linear intrinsic estimation
 * @ingroup camera_calibration
 *
 * Contains the estimated camera matrix and the per-view poses recovered
 * from homography decomposition.
 */
struct IntrinsicsEstimateResult final {
    bool success{false};

    CameraMatrix kmtx;                        ///< Estimated intrinsic matrix
    std::vector<double> dist = {0, 0, 0, 0};  ///< Distortion coefficients (k1, k2, p1, p2)
    std::vector<ViewEstimateData> views;      ///< Per-view estimation data
    std::string log;
};

/** Estimate camera intrinsics from planar views using a linear method */
auto estimate_intrinsics(const std::vector<PlanarView>& views,
                         const IntrinsicsEstimateOptions& opts = {}) -> IntrinsicsEstimateResult;

/** Linear estimate with normalized observations */
auto estimate_intrinsics_linear(const std::vector<Observation<double>>& observations,
                                std::optional<CalibrationBounds> bounds = std::nullopt,
                                bool use_skew = false) -> std::optional<CameraMatrix>;

/** Improved linear initialization with distortion estimation */
constexpr int k_default_max_iterations = 5;
auto estimate_intrinsics_linear_iterative(const std::vector<Observation<double>>& observations,
                                          int num_radial,
                                          int max_iterations = k_default_max_iterations,
                                          bool use_skew = false)
    -> std::optional<PinholeCamera<BrownConradyd>>;

}  // namespace calib
