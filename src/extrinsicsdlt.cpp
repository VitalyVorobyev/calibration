#include "calib/extrinsics.h"

// std
#include <algorithm>
#include <iostream>

#include "observationutils.h"

namespace calib {

static auto estimate_camera_poses_for_views(const std::vector<MulticamPlanarView>& views,
                                            const std::vector<Camera<DualDistortion>>& cameras)
    -> std::vector<std::vector<Eigen::Isometry3d>> {
    const size_t num_cameras = cameras.size();
    const size_t num_views = views.size();

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
            cam_se3_ref[view_idx][cam_idx] =
                estimate_planar_pose(view[cam_idx], cameras[cam_idx].kmtx);
        }
    }

    return cam_se3_ref;
}

static auto compute_relative_camera_poses(
    const std::vector<std::vector<Eigen::Isometry3d>>& cam_se3_ref,
    const std::vector<MulticamPlanarView>& views) -> std::vector<Eigen::Isometry3d> {
    const size_t num_cameras = cam_se3_ref.empty() ? 0 : cam_se3_ref[0].size();
    const size_t num_views = cam_se3_ref.size();

    std::vector<Eigen::Isometry3d> c_se3_r(num_cameras, Eigen::Isometry3d::Identity());

    // Reference camera (camera 0) stays at identity
    for (size_t cam_idx = 1; cam_idx < num_cameras; ++cam_idx) {
        std::vector<Eigen::Isometry3d> relative_poses;

        for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
            const auto& obs_ref = views[view_idx][0];
            const auto& obs_cam = views[view_idx][cam_idx];

            // Skip views with insufficient observations
            if (obs_ref.size() < 4 || obs_cam.size() < 4) {
                std::cerr << "Warning: Insufficient observations for view " << view_idx
                          << " and camera " << cam_idx << " (need ≥4 points)\n";
                continue;
            }

            // Compute relative pose: cam_idx relative to camera 0
            const auto relative_pose =
                cam_se3_ref[view_idx][cam_idx] * cam_se3_ref[view_idx][0].inverse();
            relative_poses.push_back(relative_pose);
        }

        if (!relative_poses.empty()) {
            c_se3_r[cam_idx] = average_affines(relative_poses);
        } else {
            std::cerr << "Warning: No valid relative poses found for camera " << cam_idx << "\n";
        }
    }

    return c_se3_r;
}

static auto compute_target_poses(const std::vector<std::vector<Eigen::Isometry3d>>& cam_se3_ref,
                                 const std::vector<MulticamPlanarView>& views,
                                 const std::vector<Eigen::Isometry3d>& c_se3_r)
    -> std::vector<Eigen::Isometry3d> {
    const size_t num_views = cam_se3_ref.size();
    const size_t num_cameras = c_se3_r.size();

    std::vector<Eigen::Isometry3d> r_se3_t(num_views, Eigen::Isometry3d::Identity());

    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
        std::vector<Eigen::Isometry3d> target_poses;

        for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
            const auto& obs_cam = views[view_idx][cam_idx];

            // Skip cameras with insufficient observations
            if (obs_cam.size() < 4) {
                continue;
            }

            // Transform camera pose to reference frame
            const auto target_pose = c_se3_r[cam_idx].inverse() * cam_se3_ref[view_idx][cam_idx];
            target_poses.push_back(target_pose);
        }

        if (!target_poses.empty()) {
            r_se3_t[view_idx] = average_affines(target_poses);
        } else {
            std::cerr << "Warning: No valid target poses found for view " << view_idx << "\n";
        }
    }

    return r_se3_t;
}

auto estimate_extrinsic_dlt(const std::vector<MulticamPlanarView>& views,
                            const std::vector<Camera<DualDistortion>>& cameras) -> ExtrinsicPoses {
    if (views.empty() || cameras.empty()) {
        throw std::runtime_error("Empty views or cameras provided");
    }

    // Step 1: Estimate individual camera poses for each view using DLT
    const auto cam_se3_ref = estimate_camera_poses_for_views(views, cameras);

    // Step 2: Compute relative camera poses (extrinsics)
    const auto c_se3_r = compute_relative_camera_poses(cam_se3_ref, views);

    // Step 3: Compute target poses in reference frame
    const auto r_se3_t = compute_target_poses(cam_se3_ref, views, c_se3_r);

    return ExtrinsicPoses{c_se3_r, r_se3_t};
}

}  // namespace calib
