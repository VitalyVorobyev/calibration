/** @brief Initialize camera intrinsics from a set of planar views */

#pragma once

// std
#include <optional>
#include <string>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/homography.h"
#include "calib/planarpose.h"

namespace calib {

struct ImageSize final {
    int width{0};
    int height{0};
};

struct IntrinsicsInit final {
    bool success{false};

    // Intrinsics
    double fx{0}, fy{0}, cx{0}, cy{0}, skew{0};
    Eigen::Matrix3d kmtx = Eigen::Matrix3d::Identity();

    // Distortion (Brown–Conrady init): [k1, k2, p1, p2]
    std::vector<double> dist = {0, 0, 0, 0};

    // Per-view SE3 (camera_T_target)
    std::vector<Eigen::Isometry3d> c_se3_t;

    // Diagnostics
    std::vector<HomographyResult> homographies;
    std::vector<double> per_view_forward_rms_px;
    std::string log;
};

struct InitOptions final {
    std::optional<RansacOptions> homography_ransac;  // if set, use RANSAC to filter each view

    bool assume_zero_skew = false;    // if true, zero out skew after Zhang init
    bool estimate_tangential = true;  // p1, p2
    bool estimate_k3 = false;         // not used in the LS below (kept off by default)

    bool drop_bad_views = true;
    double max_view_transfer_rms_px = 3.0;
};

/// End-to-end initializer (pre-Ceres).
/// Returns K, per-view extrinsics, and initial distortion from the given planar correspondences.
/// Any views rejected by RANSAC / quality checks are ignored in the final solve.
IntrinsicsInit initialize_from_planar(const std::vector<PlanarView>& views,
                                      const ImageSize& image_size, const InitOptions& opts = {});

}  // namespace calib
