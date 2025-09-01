#include "calib/extrinsics.h"

// std
#include <iostream>

#include "observationutils.h"

namespace calib {

ExtrinsicPoses estimate_extrinsic_dlt(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<Camera<DualDistortion>>& cameras
) {
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    std::vector<std::vector<Eigen::Affine3d>> c_T_t(
        num_views, std::vector<Eigen::Affine3d>(num_cams, Eigen::Affine3d::Identity()));

    // Estimate per-camera target poses via DLT
    for (size_t v = 0; v < num_views; ++v) {
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = views[v];
            if (c >= obs.size()) continue;
            const auto& ob_c = obs[c];
            if (ob_c.size() < 4) continue;
            std::vector<Eigen::Vector2d> obj_xy, img_uv;
            obj_xy.reserve(ob_c.size());
            img_uv.reserve(ob_c.size());
            for (const auto& o : ob_c) {
                obj_xy.push_back(o.object_xy);
                img_uv.push_back(o.image_uv);
            }
            c_T_t[v][c] = estimate_planar_pose_dlt(obj_xy, img_uv, cameras[c].K);
        }
    }

    ExtrinsicPoses guess;
    guess.c_T_r.assign(num_cams, Eigen::Affine3d::Identity());
    guess.r_T_t.assign(num_views, Eigen::Affine3d::Identity());

    // Compute camera poses relative to first camera (reference)
    for (size_t c = 1; c < num_cams; ++c) {
        std::vector<Eigen::Affine3d> rels;
        for (size_t v = 0; v < num_views; ++v) {
            if (c >= views[v].size()) {
                throw std::runtime_error("Camera index out of bounds");
            }
            const auto& obs0 = views[v][0];
            const auto& obsC = views[v][c];
            if (obs0.size() < 4 || obsC.size() < 4) {
                std::cerr << "Insufficient observations for view " << v
                    << " and camera " << c << std::endl;
                continue;
            }
            rels.push_back(c_T_t[v][c] * c_T_t[v][0].inverse());
        }
        if (!rels.empty()) {
            guess.c_T_r[c] = average_affines(rels);
        }
    }

    // Compute target poses in reference frame
    for (size_t v = 0; v < num_views; ++v) {
        std::vector<Eigen::Affine3d> tposes;
        for (size_t c = 0; c < num_cams; ++c) {
            if (c >= views[v].size()) continue;
            const auto& ob_c = views[v][c];
            if (ob_c.size() < 4) continue;
            tposes.push_back(guess.c_T_r[c].inverse() * c_T_t[v][c]);
        }
        if (!tposes.empty()) {
            guess.r_T_t[v] = average_affines(tposes);
        }
    }

    return guess;
}

}  // namespace calib