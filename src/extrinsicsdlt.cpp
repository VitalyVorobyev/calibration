#include "calib/extrinsics.h"

// std
#include <iostream>

#include "observationutils.h"

namespace calib {

auto estimate_extrinsic_dlt(const std::vector<MulticamPlanarView>& views,
                            const std::vector<Camera<DualDistortion>>& cameras) -> ExtrinsicPoses {
    const size_t num_cameras = cameras.size();
    const size_t num_views = views.size();
    std::vector<std::vector<Eigen::Affine3d>> cam_T_ref(
        num_views, std::vector<Eigen::Affine3d>(num_cameras, Eigen::Affine3d::Identity()));

    // Estimate per-camera reference poses via DLT
    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
        for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
            const auto& obs = views[view_idx];
            if (cam_idx >= obs.size()) { continue; }
            const auto& obs_cam = obs[cam_idx];
            if (obs_cam.size() < 4) { continue; }
            std::vector<Eigen::Vector2d> obj_xy, img_uv;
            obj_xy.reserve(obs_cam.size());
            img_uv.reserve(obs_cam.size());
            for (const auto& obs_elem : obs_cam) {
                obj_xy.push_back(obs_elem.object_xy);
                img_uv.push_back(obs_elem.image_uv);
            }
            cam_T_ref[view_idx][cam_idx] = estimate_planar_pose_dlt(obj_xy, img_uv, cameras[cam_idx].K);
        }
    }

    ExtrinsicPoses guess;
    guess.c_T_r.assign(num_cameras, Eigen::Affine3d::Identity());
    guess.r_T_t.assign(num_views, Eigen::Affine3d::Identity());

    // Compute camera poses relative to first camera (reference)
    for (size_t cam_idx = 1; cam_idx < num_cameras; ++cam_idx) {
        std::vector<Eigen::Affine3d> rels;
        for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
            if (cam_idx >= views[view_idx].size()) {
                throw std::runtime_error("Camera index out of bounds");
            }
            const auto& obs0 = views[view_idx][0];
            const auto& obs_cam = views[view_idx][cam_idx];
            if (obs0.size() < 4 || obs_cam.size() < 4) {
                std::cerr << "Insufficient observations for view " << view_idx << " and camera " << cam_idx << "\n";
                continue;
            }
            rels.push_back(cam_T_ref[view_idx][cam_idx] * cam_T_ref[view_idx][0].inverse());
        }
        if (!rels.empty()) {
            guess.c_T_r[cam_idx] = average_affines(rels);
        }
    }

    // Compute target poses in reference frame
    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
        std::vector<Eigen::Affine3d> tposes;
        for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
            if (cam_idx >= views[view_idx].size()) { continue; }
            const auto& obs_cam = views[view_idx][cam_idx];
            if (obs_cam.size() < 4) { continue; }
            tposes.push_back(guess.c_T_r[cam_idx].inverse() * cam_T_ref[view_idx][cam_idx]);
        }
        if (!tposes.empty()) {
            guess.r_T_t[view_idx] = average_affines(tposes);
        }
    }

    return guess;
}

}  // namespace calib