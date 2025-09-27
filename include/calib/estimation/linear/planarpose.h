/** @brief Linear planar pose estimation interfaces (header-only helpers)
 */

#pragma once

// std
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/models/camera_matrix.h"
#include "calib/models/cameramodel.h"

namespace calib {

struct PlanarObservation {
    Eigen::Vector2d object_xy;  // Planar target coordinates (Z=0)
    Eigen::Vector2d image_uv;   // Corresponding pixel measurements
};
using PlanarView = std::vector<PlanarObservation>;

// Decompose homography in normalized camera coords: H = [r1 r2 t]
auto pose_from_homography_normalized(const Eigen::Matrix3d& hmtx) -> Eigen::Isometry3d;

// Convenience: one-shot planar pose from pixels & kmtx
auto estimate_planar_pose(PlanarView view, const CameraMatrix& intrinsics) -> Eigen::Isometry3d;

// Generic overload: estimate planar pose for an arbitrary camera_model by
// normalizing pixel coordinates via the camera's intrinsics.
template <camera_model CameraT>
auto estimate_planar_pose(PlanarView view, const CameraT& camera) -> Eigen::Isometry3d {
    if (view.size() < 4) {
        return Eigen::Isometry3d::Identity();
    }

    // Normalize image coordinates using the camera's intrinsics only
    std::for_each(view.begin(), view.end(), [&camera](PlanarObservation& obs) {
        obs.image_uv = camera.template apply_intrinsics<double>(obs.image_uv);
    });

    // Estimate a homography using a simple normalized DLT and recover pose
    auto normalize_points_2d = [](const std::vector<Eigen::Vector2d>& pts,
                                  std::vector<Eigen::Vector2d>& out) -> Eigen::Matrix3d {
        out.resize(pts.size());
        const Eigen::Vector2d centroid =
            std::accumulate(pts.begin(), pts.end(), Eigen::Vector2d{0, 0}) /
            std::max<size_t>(1, pts.size());
        const double mean_dist = std::accumulate(pts.begin(), pts.end(), 0.0,
                                                 [&centroid](double sum, const Eigen::Vector2d& p) {
                                                     return sum + (p - centroid).norm();
                                                 }) /
                                 static_cast<double>(std::max<size_t>(1, pts.size()));
        const double sigma = (mean_dist > 0) ? std::sqrt(2.0) / mean_dist : 1.0;

        Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
        transform(0, 0) = sigma;
        transform(1, 1) = sigma;
        transform(0, 2) = -sigma * centroid.x();
        transform(1, 2) = -sigma * centroid.y();

        std::transform(pts.begin(), pts.end(), out.begin(),
                       [&transform](const Eigen::Vector2d& pt) -> Eigen::Vector2d {
                           Eigen::Vector3d hp(pt.x(), pt.y(), 1.0);
                           Eigen::Vector3d hn = transform * hp;
                           return hn.hnormalized();
                       });
        return transform;
    };

    std::vector<Eigen::Vector2d> src_xy;
    std::vector<Eigen::Vector2d> dst_uv;
    src_xy.reserve(view.size());
    dst_uv.reserve(view.size());
    for (const auto& obs : view) {
        src_xy.push_back(obs.object_xy);
        dst_uv.push_back(obs.image_uv);
    }

    std::vector<Eigen::Vector2d> src_n, dst_n;
    const Eigen::Matrix3d Tsrc = normalize_points_2d(src_xy, src_n);
    const Eigen::Matrix3d Tdst = normalize_points_2d(dst_uv, dst_n);

    const auto npts = static_cast<Eigen::Index>(src_n.size());
    Eigen::MatrixXd A(2 * npts, 9);
    for (Eigen::Index i = 0; i < npts; ++i) {
        const double x = src_n[i].x();
        const double y = src_n[i].y();
        const double u = dst_n[i].x();
        const double v = dst_n[i].y();
        A.row(2 * i) << -x, -y, -1, 0, 0, 0, u * x, u * y, u;
        A.row(2 * i + 1) << 0, 0, 0, -x, -y, -1, v * x, v * y, v;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);
    Eigen::Matrix3d Hnorm;
    Hnorm << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    Eigen::Matrix3d hmtx = Tdst.inverse() * Hnorm * Tsrc;
    if (std::abs(hmtx(2, 2)) > 1e-15) {
        hmtx /= hmtx(2, 2);
    }
    return pose_from_homography_normalized(hmtx);
}

}  // namespace calib
