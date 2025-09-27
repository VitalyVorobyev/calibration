#include "calib/estimation/posefromhomography.h"

// std
#include <cmath>
#include <iostream>
#include <limits>

#include <calib/estimation/common/se3_utils.h>

namespace calib {

auto pose_from_homography(const CameraMatrix& kmtx, const Eigen::Matrix3d& hmtx)
    -> PoseFromHResult {
    PoseFromHResult out;

    // Basic checks
    if (!std::isfinite(kmtx.fx) || !std::isfinite(kmtx.fy) || kmtx.cx <= 0 || kmtx.cy <= 0) {
        std::cerr << "Invalid camera matrix K:\n" << kmtx.matrix() << "\n";
        out.message = "Invalid camera matrix K";
        return out;
    }
    if (!std::isfinite(hmtx(2, 2))) {
        out.message = "Invalid homography H.";
        return out;
    }

    // 1) Remove intrinsics
    const Eigen::Matrix3d hnorm = kmtx.matrix().inverse() * hmtx;

    // 2) Compute scale from first two columns
    const double n1 = hnorm.col(0).norm();
    const double n2 = hnorm.col(1).norm();

    constexpr double eps = 1e-15;
    if (!(n1 > eps) || !(n2 > eps)) {
        out.message = "Degenerate H: zero column norm.";
        return out;
    }

    const double scale = 1.0 / ((n1 + n2) * 0.5);  // mean-based scale is numerically stable
    out.scale = scale;
    out.cond_check = (n1 > n2) ? (n1 / n2) : (n2 / n1);

    // 3) Raw r1,r2,t (before orthonormalization)
    Eigen::Matrix3d rotation;
    rotation.col(0) = scale * hnorm.col(0);
    rotation.col(1) = scale * hnorm.col(1);
    rotation.col(2) = rotation.col(0).cross(rotation.col(1));
    project_to_so3(rotation);

    Eigen::Vector3d translation = scale * hnorm.col(2);

    // 4) Build preliminary rotation with r3 = r1 x r2

    // 6) Prefer a solution with the target plane in front of the camera (t_z > 0)
    //    For points near the plane origin (X≈Y≈0), Zc ≈ t_z.
    if (translation.z() <= 0) {
        rotation = -rotation;
        translation = -translation;
    }

    out.success = true;
    out.c_se3_t.linear() = rotation;
    out.c_se3_t.translation() = translation;
    out.message = "OK";
    return out;
}

auto homography_consistency_fro(const CameraMatrix& kmtx, const Eigen::Isometry3d& c_se3_t,
                                const Eigen::Matrix3d& hmtx) -> double {
    Eigen::Matrix3d hrt = Eigen::Matrix3d::Zero();
    hrt.col(0) = c_se3_t.linear().col(0);
    hrt.col(1) = c_se3_t.linear().col(1);
    hrt.col(2) = c_se3_t.translation();
    const Eigen::Matrix3d hmtx_hat = kmtx.matrix() * hrt;

    const double num = (hmtx_hat - hmtx).norm();
    const double den = hmtx.norm();
    return (den > 0) ? (num / den) : std::numeric_limits<double>::infinity();
}

}  // namespace calib
