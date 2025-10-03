#include "calib/estimation/linear/planarpose.h"

// std
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "detail/homographyestimator.h"

namespace calib {

Eigen::Isometry3d pose_from_homography_normalized(const Eigen::Matrix3d& hmtx) {
    Eigen::Vector3d hcol1 = hmtx.col(0);
    Eigen::Vector3d hcol2 = hmtx.col(1);
    Eigen::Vector3d hcol3 = hmtx.col(2);

    double s = std::sqrt(hcol1.norm() * hcol2.norm());
    if (s < 1e-12) {
        s = 1.0;
    }
    Eigen::Vector3d rcol1 = hcol1 / s;
    Eigen::Vector3d rcol2 = hcol2 / s;
    Eigen::Vector3d rcol3 = rcol1.cross(rcol2);

    Eigen::Matrix3d r_init;
    r_init.col(0) = rcol1;
    r_init.col(1) = rcol2;
    r_init.col(2) = rcol3;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(r_init, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
    if (rotation.determinant() < 0) {
        Eigen::Matrix3d vmtx = svd.matrixV();
        vmtx.col(2) *= -1.0;
        rotation = svd.matrixU() * vmtx.transpose();
    }
    Eigen::Vector3d translation = hcol3 / s;
    if (rotation(2, 2) < 0) {
        rotation = -rotation;
        translation = -translation;
    }

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = rotation;
    pose.translation() = translation;
    return pose;
}

auto estimate_planar_pose(PlanarView view, const CameraMatrix& intrinsics) -> Eigen::Isometry3d {
    if (view.size() < 4) {
        return Eigen::Isometry3d::Identity();
    }

    std::for_each(view.begin(), view.end(), [&intrinsics](PlanarObservation& obs) {
        obs.image_uv = normalize(intrinsics, obs.image_uv);
    });

    std::vector<int> sample(view.size());
    std::iota(sample.begin(), sample.end(), 0);
    auto h_opt = HomographyEstimator::fit(view, sample);
    if (!h_opt.has_value()) {
        std::cerr << "Failed to estimate homography using DLT" << std::endl;
        return Eigen::Isometry3d::Identity();
    }

    Eigen::Matrix3d hmtx = h_opt.value();
    if (std::abs(hmtx(2, 2)) > 1e-15) {
        hmtx /= hmtx(2, 2);
    }
    return pose_from_homography_normalized(hmtx);
}

}  // namespace calib
