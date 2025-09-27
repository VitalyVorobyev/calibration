#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>

#include "calib/estimation/linear/homography.h"
#include "calib/estimation/linear/planarpose.h"  // PlanarView
#include "calib/estimation/linear/planefit.h"
#include "calib/models/cameramodel.h"

namespace calib {

struct LineScanView final {
    PlanarView target_view;
    std::vector<Eigen::Vector2d> laser_uv;
};

struct LineScanCalibrationResult final {
    Eigen::Vector4d plane;
    Eigen::Matrix4d covariance;
    Eigen::Matrix3d homography;
    double rms_error = 0.0;
    std::string summary;
};

inline void validate_observations(const std::vector<LineScanView>& views) {
    if (views.size() < 2) {
        throw std::invalid_argument("At least 2 views are required");
    }
    if (std::any_of(views.begin(), views.end(),
                    [](const auto& v) { return v.target_view.size() < 4; })) {
        throw std::invalid_argument("Each view requires >=4 target correspondences");
    }
}

inline Eigen::Matrix3d build_plane_homography(const Eigen::Vector4d& plane) {
    Eigen::Vector3d nvec = plane.head<3>();
    Eigen::Vector3d p0 = -plane[3] * nvec;
    Eigen::Vector3d tmp =
        (std::abs(nvec.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
    Eigen::Vector3d e1 = nvec.cross(tmp).normalized();
    Eigen::Vector3d e2 = nvec.cross(e1).normalized();
    Eigen::Matrix3d plane_to_norm;
    plane_to_norm.col(0) = e1;
    plane_to_norm.col(1) = e2;
    plane_to_norm.col(2) = p0;
    return plane_to_norm.inverse();
}

template <camera_model CameraT>
std::vector<Eigen::Vector3d> points_from_view(LineScanView view, const CameraT& camera) {
    std::for_each(view.target_view.begin(), view.target_view.end(),
                  [&camera](PlanarObservation& item) {
                      item.image_uv = camera.template unproject<double>(item.image_uv);
                  });

    auto hres = estimate_homography(view.target_view);
    if (!hres.success) {
        return {};
    }

    Eigen::Isometry3d pose = pose_from_homography_normalized(hres.hmtx);
    Eigen::Matrix3d h_norm_to_obj = hres.hmtx.inverse();
    if (std::abs(h_norm_to_obj(2, 2)) > 1e-15) {
        h_norm_to_obj /= h_norm_to_obj(2, 2);
    }

    std::vector<Eigen::Vector3d> points;
    points.reserve(view.laser_uv.size());
    for (const auto& lpix : view.laser_uv) {
        Eigen::Vector2d norm = camera.template unproject<double>(lpix);
        Eigen::Vector3d hp = h_norm_to_obj * Eigen::Vector3d(norm.x(), norm.y(), 1.0);
        Eigen::Vector2d plane_xy = hp.hnormalized();
        Eigen::Vector3d obj_pt(plane_xy.x(), plane_xy.y(), 0.0);
        points.push_back(pose * obj_pt);
    }
    return points;
}

inline double plane_rms(const std::vector<Eigen::Vector3d>& pts, const Eigen::Vector4d& plane) {
    double ss = 0.0;
    for (const auto& p : pts) {
        double r = plane.head<3>().dot(p) + plane[3];
        ss += r * r;
    }
    return std::sqrt(ss / static_cast<double>(pts.size()));
}

template <camera_model CameraT>
LineScanCalibrationResult calibrate_laser_plane(const std::vector<LineScanView>& views,
                                                const CameraT& camera) {
    validate_observations(views);

    LineScanCalibrationResult result;
    std::vector<Eigen::Vector3d> all_points;
    for (const auto& view : views) {
        auto pts = points_from_view(view, camera);
        all_points.insert(all_points.end(), pts.begin(), pts.end());
    }
    if (all_points.size() < 3) {
        throw std::invalid_argument("Not enough laser points to fit a plane");
    }

    result.plane = fit_plane_svd(all_points);
    result.rms_error = plane_rms(all_points, result.plane);
    result.homography = build_plane_homography(result.plane);
    result.covariance.setZero();
    result.summary = "linear_svd";
    return result;
}

}  // namespace calib
