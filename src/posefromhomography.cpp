#include "calibration/posefromhomography.h"

// std
#include <algorithm>
#include <numeric>

#include "calibration/homography.h"

namespace vitavision {

// Decompose homography in normalized camera coords: H = [r1 r2 t]
Eigen::Affine3d pose_from_homography_normalized(const Eigen::Matrix3d& H) {
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);

    double s = std::sqrt(h1.norm() * h2.norm());
    if (s < 1e-12) s = 1.0;
    Eigen::Vector3d r1 = h1 / s;
    Eigen::Vector3d r2 = h2 / s;
    Eigen::Vector3d r3 = r1.cross(r2);

    // Orthonormalize to the nearest rotation
    Eigen::Matrix3d Rinit;
    Rinit.col(0) = r1;
    Rinit.col(1) = r2;
    Rinit.col(2) = r3;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rinit, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = svd.matrixU() * V.transpose();
    }
    Eigen::Vector3d t = h3 / s;
    if (R(2, 2) < 0) { // enforce cheirality (Z forward)
        R = -R; t = -t;
    }

    auto pose = Eigen::Affine3d::Identity();
    pose.linear() = R;
    pose.translation() = t;
    return pose;
}

// Convenience: one-shot planar pose from pixels & K
// Returns true on success; outputs R (world->cam) and t
Eigen::Affine3d estimate_planar_pose_dlt(const std::vector<Eigen::Vector2d>& obj_xy,
                                         const std::vector<Eigen::Vector2d>& img_uv,
                                         const Intrinsic& intrinsics) {
    if (obj_xy.size() < 4 || obj_xy.size() != img_uv.size()) {
        return Eigen::Affine3d::Identity();
    }
    
    std::vector<Eigen::Vector2d> img_norm(img_uv.size());
    std::transform(img_uv.begin(), img_uv.end(), img_norm.begin(),
        [&intrinsics](const Eigen::Vector2d& pix) {
            return intrinsics.pixel_to_norm(pix);
        });

    Eigen::Matrix3d H = estimate_homography_dlt(obj_xy, img_norm);
    return pose_from_homography_normalized(H);
}

}  // namespace vitavision
