/** @brief Linear multi-camera extrinsics initialisation (DLT) */

#pragma once

// std
#include <string>
#include <vector>

// eigen
#include <Eigen/Geometry>

#include "calib/estimation/linear/planarpose.h"
#include "calib/models/cameramodel.h"
#include "calib/models/pinhole.h"

namespace calib {

// [camera]
using MulticamPlanarView = std::vector<PlanarView>;

struct ExtrinsicPoses final {
    std::vector<Eigen::Isometry3d> c_se3_r;  // reference->camera
    std::vector<Eigen::Isometry3d> r_se3_t;  // target->reference
};

template <camera_model CameraT>
auto estimate_extrinsic_dlt(const std::vector<MulticamPlanarView>& views,
                            const std::vector<CameraT>& cameras) -> ExtrinsicPoses {
    if (views.empty() || cameras.empty()) {
        throw std::runtime_error("Empty views or cameras provided");
    }

    const size_t num_cameras = cameras.size();
    const size_t num_views = views.size();

    // Helper: average a set of SE3 poses (translation + quaternion averaging)
    auto average_affines_local = [](const std::vector<Eigen::Isometry3d>& poses) {
        if (poses.empty()) return Eigen::Isometry3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        Eigen::Quaterniond qsum(0, 0, 0, 0);
        for (const auto& p : poses) {
            t += p.translation();
            Eigen::Quaterniond q(p.linear());
            if (qsum.coeffs().dot(q.coeffs()) < 0.0) q.coeffs() *= -1.0;
            qsum.coeffs() += q.coeffs();
        }
        t /= static_cast<double>(poses.size());
        qsum.normalize();
        Eigen::Isometry3d avg = Eigen::Isometry3d::Identity();
        avg.linear() = qsum.toRotationMatrix();
        avg.translation() = t;
        return avg;
    };

    // Step 1: Per-view camera poses relative to a transient reference (the target)
    std::vector<std::vector<Eigen::Isometry3d>> cam_se3_ref(
        num_views, std::vector<Eigen::Isometry3d>(num_cameras, Eigen::Isometry3d::Identity()));

    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
        const auto& view = views[view_idx];
        if (view.size() != num_cameras) {
            throw std::runtime_error(
                "View " + std::to_string(view_idx) + " has wrong number of cameras: expected " +
                std::to_string(num_cameras) + ", got " + std::to_string(view.size()));
        }
        for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
            cam_se3_ref[view_idx][cam_idx] = estimate_planar_pose(view[cam_idx], cameras[cam_idx]);
        }
    }

    // Step 2: Compute relative camera poses c_se3_r: relative to camera 0
    std::vector<Eigen::Isometry3d> c_se3_r(num_cameras, Eigen::Isometry3d::Identity());
    for (size_t cam_idx = 1; cam_idx < num_cameras; ++cam_idx) {
        std::vector<Eigen::Isometry3d> rels;
        for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
            const auto& obs_ref = views[view_idx][0];
            const auto& obs_cam = views[view_idx][cam_idx];
            if (obs_ref.size() < 4 || obs_cam.size() < 4) continue;
            rels.push_back(cam_se3_ref[view_idx][cam_idx] * cam_se3_ref[view_idx][0].inverse());
        }
        if (!rels.empty()) c_se3_r[cam_idx] = average_affines_local(rels);
    }

    // Step 3: Compute target poses r_se3_t per view
    std::vector<Eigen::Isometry3d> r_se3_t(num_views, Eigen::Isometry3d::Identity());
    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
        std::vector<Eigen::Isometry3d> tposes;
        for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
            if (views[view_idx][cam_idx].size() < 4) continue;
            tposes.push_back(c_se3_r[cam_idx].inverse() * cam_se3_ref[view_idx][cam_idx]);
        }
        if (!tposes.empty()) r_se3_t[view_idx] = average_affines_local(tposes);
    }

    return ExtrinsicPoses{c_se3_r, r_se3_t};
}

}  // namespace calib
