#include "calib/handeye.h"

// std
#include <iostream>
#include <numbers>

#include "observationutils.h"

namespace calib {

auto build_all_pairs(
    const std::vector<Eigen::Affine3d>& ref_T_gripper, const std::vector<Eigen::Affine3d>& cam_T_target,
    double min_angle_deg,       // discard too-small motions
    bool reject_axis_parallel,  // guard against ill-conditioning
    double axis_parallel_eps) -> std::vector<MotionPair> {
    if (ref_T_gripper.size() < 2 || ref_T_gripper.size() != cam_T_target.size()) {
        throw std::runtime_error("Inconsistent hand-eye input sizes");
    }
    const size_t num_poses = ref_T_gripper.size();
    const double min_angle = min_angle_deg * std::numbers::pi / 180.0;
    std::vector<MotionPair> pairs;
    pairs.reserve(num_poses * (num_poses - 1) / 2);
    for (size_t idx_i = 0; idx_i + 1 < num_poses; ++idx_i) {
        for (size_t idx_j = idx_i + 1; idx_j < num_poses; ++idx_j) {
            Eigen::Affine3d affine_a = ref_T_gripper[idx_i].inverse() * ref_T_gripper[idx_j];
            Eigen::Affine3d affine_b = cam_T_target[idx_i] * cam_T_target[idx_j].inverse();
            MotionPair motion_pair;
            motion_pair.RA = projectToSO3(affine_a.linear());
            motion_pair.RB = projectToSO3(affine_b.linear());
            motion_pair.tA = affine_a.translation();
            motion_pair.tB = affine_b.translation();
            Eigen::Vector3d alpha = logSO3(motion_pair.RA);
            Eigen::Vector3d beta = logSO3(motion_pair.RB);
            const double norm_a = alpha.norm();
            const double norm_b = beta.norm();
            const double min_rot = std::min(norm_a, norm_b);
            if (min_rot < min_angle) {
                std::cerr << "Skipping pair (" << idx_i << "," << idx_j
                          << ") with too small motion: " << (min_rot * 180.0 / std::numbers::pi)
                          << " deg\n";
                continue;
            }
            if (reject_axis_parallel) {
                const double aa = (norm_a < 1e-9) ? 0.0 : 1.0;
                const double bb = (norm_b < 1e-9) ? 0.0 : 1.0;
                if (aa * bb > 0.0) {
                    double sin_axis = (alpha.normalized().cross(beta.normalized())).norm();
                    if (sin_axis < axis_parallel_eps) {
                        std::cerr << "Skipping pair (" << idx_i << "," << idx_j
                                  << ") with near-parallel axes\n";
                        continue;
                    }
                }
            }
            pairs.push_back(std::move(motion_pair));
        }
    }
    if (pairs.empty()) {
        throw std::runtime_error(
            "No valid motion pairs after filtering. Increase motion or relax thresholds.");
    }
    return pairs;
}

// ---------- weighted Tsai–Lenz rotation over all pairs ----------
static auto estimate_rotation_allpairs_weighted(const std::vector<MotionPair>& pairs) -> Eigen::Matrix3d {
    const int num_pairs = static_cast<int>(pairs.size());
    Eigen::MatrixXd mat_M(3 * num_pairs, 3);
    Eigen::VectorXd vec_d(3 * num_pairs);
    for (int idx = 0; idx < num_pairs; ++idx) {
        Eigen::Vector3d alpha = logSO3(pairs[idx].RA);
        Eigen::Vector3d beta = logSO3(pairs[idx].RB);
        constexpr double kWeight = 1.0;
        mat_M.block<3, 3>(static_cast<Eigen::Index>(3 * idx), 0) = kWeight * skew(alpha + beta);
        vec_d.segment<3>(static_cast<Eigen::Index>(3 * idx)) = kWeight * (beta - alpha);
    }
    constexpr double kRidge = 1e-12;
    Eigen::Vector3d rot_vec = ridge_llsq(mat_M, vec_d, kRidge);
    return expSO3(rot_vec);
}

// ---------- weighted Tsai–Lenz translation over all pairs ----------
static auto estimate_translation_allpairs_weighted(const std::vector<MotionPair>& pairs,
                                                  const Eigen::Matrix3d& rot_X) -> Eigen::Vector3d {
    const int num_pairs = static_cast<int>(pairs.size());
    Eigen::MatrixXd mat_C(3 * num_pairs, 3);
    Eigen::VectorXd vec_w(3 * num_pairs);
    for (int idx = 0; idx < num_pairs; ++idx) {
        const Eigen::Matrix3d& rot_A = pairs[idx].RA;
        const Eigen::Vector3d& tran_A = pairs[idx].tA;
        const Eigen::Vector3d& tran_B = pairs[idx].tB;
        constexpr double kWeight = 1.0;
        mat_C.block<3, 3>(static_cast<Eigen::Index>(3 * idx), 0) = kWeight * (rot_A - Eigen::Matrix3d::Identity());
        vec_w.segment<3>(static_cast<Eigen::Index>(3 * idx)) = kWeight * (rot_X * tran_B - tran_A);
    }
    constexpr double kRidge = 1e-12;
    return ridge_llsq(mat_C, vec_w, kRidge);
}

// ---------- public API: linear init + non-linear refine ----------
auto estimate_handeye_dlt(const std::vector<Eigen::Affine3d>& ref_T_gripper,
                         const std::vector<Eigen::Affine3d>& camera_T_target,
                         double min_angle_deg) -> Eigen::Affine3d {
    auto pairs = build_all_pairs(ref_T_gripper, camera_T_target, min_angle_deg);
    const Eigen::Matrix3d rot_X = estimate_rotation_allpairs_weighted(pairs);
    const Eigen::Vector3d g_t_c = estimate_translation_allpairs_weighted(pairs, rot_X);
    Eigen::Affine3d pose = Eigen::Affine3d::Identity();
    pose.linear() = rot_X;
    pose.translation() = g_t_c;
    return pose;
}

}  // namespace calib
