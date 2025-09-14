#include "calib/handeye.h"

// std
#include <spdlog/spdlog.h>
#include <numbers>

#include "observationutils.h"

namespace calib {

static auto make_motion_pair(const Eigen::Isometry3d& base_se3_gripper_a,
                             const Eigen::Isometry3d& cam_se3_target_a,
                             const Eigen::Isometry3d& base_se3_gripper_b,
                             const Eigen::Isometry3d& cam_se3_target_b) -> MotionPair {
    Eigen::Isometry3d affine_a = base_se3_gripper_a.inverse() * base_se3_gripper_b;
    Eigen::Isometry3d affine_b = cam_se3_target_a * cam_se3_target_b.inverse();
    MotionPair motion_pair;
    motion_pair.rot_a = project_to_so3(affine_a.linear());
    motion_pair.rot_b = project_to_so3(affine_b.linear());
    motion_pair.tra_a = affine_a.translation();
    motion_pair.tra_b = affine_b.translation();
    return motion_pair;
}

static auto is_good_pair(const MotionPair& motion_pair, double min_angle, bool reject_axis_parallel,
                         double axis_parallel_eps) -> bool {
    Eigen::Vector3d alpha = log_so3(motion_pair.rot_a);
    Eigen::Vector3d beta = log_so3(motion_pair.rot_b);
    const double norm_a = alpha.norm();
    const double norm_b = beta.norm();
    const double min_rot = std::min(norm_a, norm_b);
    if (min_rot < min_angle) {
        spdlog::warn("Motion pair with too small motion: {} deg", min_rot * 180.0 / std::numbers::pi);
        return false;
    }
    if (reject_axis_parallel) {
        const double aa = (norm_a < 1e-9) ? 0.0 : 1.0;
        const double bb = (norm_b < 1e-9) ? 0.0 : 1.0;
        if (aa * bb > 0.0) {
            double sin_axis = (alpha.normalized().cross(beta.normalized())).norm();
            if (sin_axis < axis_parallel_eps) {
                spdlog::warn("Motion pair with near-parallel axes");
                return false;
            }
        }
    }
    return true;
}

auto build_all_pairs(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                     const std::vector<Eigen::Isometry3d>& cam_se3_target,
                     double min_angle_deg,       // discard too-small motions
                     bool reject_axis_parallel,  // guard against ill-conditioning
                     double axis_parallel_eps) -> std::vector<MotionPair> {
    if (base_se3_gripper.size() < 2 || base_se3_gripper.size() != cam_se3_target.size()) {
        throw std::runtime_error("Inconsistent hand-eye input sizes");
    }
    const size_t num_poses = base_se3_gripper.size();
    const double min_angle = min_angle_deg * std::numbers::pi / 180.0;
    std::vector<MotionPair> pairs;
    pairs.reserve(num_poses * (num_poses - 1) / 2);
    for (size_t idx_i = 0; idx_i + 1 < num_poses; ++idx_i) {
        for (size_t idx_j = idx_i + 1; idx_j < num_poses; ++idx_j) {
            MotionPair motion_pair =
                make_motion_pair(base_se3_gripper[idx_i], cam_se3_target[idx_i],
                                 base_se3_gripper[idx_j], cam_se3_target[idx_j]);

            if (is_good_pair(motion_pair, min_angle, reject_axis_parallel, axis_parallel_eps)) {
                pairs.push_back(std::move(motion_pair));
            } else {
                spdlog::warn("Skipping pair ({},{})", idx_i, idx_j);
            }
        }
    }
    if (pairs.empty()) {
        throw std::runtime_error(
            "No valid motion pairs after filtering. Increase motion or relax thresholds.");
    }
    return pairs;
}

// ---------- weighted Tsai–Lenz rotation over all pairs ----------
static auto estimate_rotation_allpairs_weighted(const std::vector<MotionPair>& pairs)
    -> Eigen::Matrix3d {
    const int num_pairs = static_cast<int>(pairs.size());
    Eigen::MatrixXd mat_m(3 * num_pairs, 3);
    Eigen::VectorXd vec_d(3 * num_pairs);
    for (int idx = 0; idx < num_pairs; ++idx) {
        Eigen::Vector3d alpha = log_so3(pairs[idx].rot_a);
        Eigen::Vector3d beta = log_so3(pairs[idx].rot_b);
        constexpr double k_weight = 1.0;
        mat_m.block<3, 3>(static_cast<Eigen::Index>(3) * idx, 0) = k_weight * skew(alpha + beta);
        vec_d.segment<3>(static_cast<Eigen::Index>(3) * idx) = k_weight * (beta - alpha);
    }
    constexpr double k_ridge = 1e-12;
    Eigen::Vector3d rot_vec = ridge_llsq(mat_m, vec_d, k_ridge);
    return exp_so3(rot_vec);
}

// ---------- weighted Tsai–Lenz translation over all pairs ----------
static auto estimate_translation_allpairs_weighted(
    const std::vector<MotionPair>& pairs, const Eigen::Matrix3d& rot_x) -> Eigen::Vector3d {
    const int num_pairs = static_cast<int>(pairs.size());
    Eigen::MatrixXd mat_c(3 * num_pairs, 3);
    Eigen::VectorXd vec_w(3 * num_pairs);
    for (int idx = 0; idx < num_pairs; ++idx) {
        const Eigen::Matrix3d& rot_a = pairs[idx].rot_a;
        const Eigen::Vector3d& tran_a = pairs[idx].tra_a;
        const Eigen::Vector3d& tran_b = pairs[idx].tra_b;
        constexpr double k_weight = 1.0;
        mat_c.block<3, 3>(static_cast<Eigen::Index>(3) * idx, 0) =
            k_weight * (rot_a - Eigen::Matrix3d::Identity());
        vec_w.segment<3>(static_cast<Eigen::Index>(3) * idx) = k_weight * (rot_x * tran_b - tran_a);
    }
    constexpr double k_ridge = 1e-12;
    return ridge_llsq(mat_c, vec_w, k_ridge);
}

// ---------- public API: linear init + non-linear refine ----------
auto estimate_handeye_dlt(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                          const std::vector<Eigen::Isometry3d>& camera_se3_target,
                          double min_angle_deg) -> Eigen::Isometry3d {
    auto pairs = build_all_pairs(base_se3_gripper, camera_se3_target, min_angle_deg);
    const Eigen::Matrix3d rot_x = estimate_rotation_allpairs_weighted(pairs);
    const Eigen::Vector3d g_tra_c = estimate_translation_allpairs_weighted(pairs, rot_x);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = rot_x;
    pose.translation() = g_tra_c;
    return pose;
}

}  // namespace calib
