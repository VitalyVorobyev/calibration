#include "calib/handeye.h"

// std
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <numbers>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "observationutils.h"
#include "handeyeresidual.h"

namespace calib {

// ---------- motion pair packing ----------
static std::vector<MotionPair> build_all_pairs(
    const std::vector<Eigen::Affine3d>& b_T_g,
    const std::vector<Eigen::Affine3d>& c_T_t,
    double min_angle_deg = 1.0,          // discard too-small motions
    bool reject_axis_parallel = true,    // guard against ill-conditioning
    double axis_parallel_eps = 1e-3
) {
    if (b_T_g.size() < 2 || b_T_g.size() != c_T_t.size()) {
        throw std::runtime_error("Inconsistent hand-eye input sizes");
    }
    const size_t n = b_T_g.size();
    const double min_angle = min_angle_deg * std::numbers::pi / 180.0;

    std::vector<MotionPair> pairs;
    pairs.reserve(n*(n-1)/2);

    for (size_t i = 0; i+1 < n; ++i) {
        for (size_t j = i+1; j < n; ++j) {
            Eigen::Affine3d A = b_T_g[i].inverse() * b_T_g[j];
            Eigen::Affine3d B = c_T_t[i] * c_T_t[j].inverse();

            MotionPair mp;
            mp.RA = projectToSO3(A.linear());
            mp.RB = projectToSO3(B.linear());
            mp.tA = A.translation();
            mp.tB = B.translation();

            Eigen::Vector3d alpha = logSO3(mp.RA);
            Eigen::Vector3d beta  = logSO3(mp.RB);
            const double a = alpha.norm();
            const double b = beta.norm();

            // weight by the smaller rotation magnitude; skip if too small
            const double w = std::min(a, b);
            if (w < min_angle) {
                std::cerr << "Skipping pair (" << i << "," << j << ") with too small motion: "
                          << (w * 180.0 / std::numbers::pi) << " deg\n";
                continue;
            }

            if (reject_axis_parallel) {
                const double aa = (a < 1e-9) ? 0.0 : 1.0;
                const double bb = (b < 1e-9) ? 0.0 : 1.0;
                if (aa*bb > 0.0) {
                    double sin_axis = (alpha.normalized().cross(beta.normalized())).norm();
                    if (sin_axis < axis_parallel_eps) {
                        std::cerr << "Skipping pair (" << i << "," << j << ") with near-parallel axes\n";
                        continue; // nearly same axis
                    }
                }
            }

            #if 0
            mp.sqrt_weight = std::sqrt(w);
            #endif

            pairs.push_back(std::move(mp));
        }
    }
    if (pairs.empty()) {
        throw std::runtime_error("No valid motion pairs after filtering. Increase motion or relax thresholds.");
    }
    return pairs;
}

// ---------- weighted Tsai–Lenz rotation over all pairs ----------
static Eigen::Matrix3d estimate_rotation_allpairs_weighted(const std::vector<MotionPair>& pairs) {
    const int m = static_cast<int>(pairs.size());
    Eigen::MatrixXd M(3*m, 3);
    Eigen::VectorXd d(3*m);

    for (int k = 0; k < m; ++k) {
        Eigen::Vector3d alpha = logSO3(pairs[k].RA);
        Eigen::Vector3d beta  = logSO3(pairs[k].RB);
        constexpr double s = 1;  // pairs[k].sqrt_weight;

        M.block<3,3>(3*k, 0) = s * skew(alpha + beta);
        d.segment<3>(3*k)    = s * (beta - alpha);
    }

    Eigen::Vector3d r = ridge_llsq(M, d, 1e-12);
    return expSO3(r);
}

// ---------- weighted Tsai–Lenz translation over all pairs ----------
static Eigen::Vector3d estimate_translation_allpairs_weighted(
    const std::vector<MotionPair>& pairs, const Eigen::Matrix3d& RX)
{
    const int m = static_cast<int>(pairs.size());
    Eigen::MatrixXd C(3*m, 3);
    Eigen::VectorXd w(3*m);

    for (int k = 0; k < m; ++k) {
        const Eigen::Matrix3d& RA = pairs[k].RA;
        const Eigen::Vector3d& tA = pairs[k].tA;
        const Eigen::Vector3d& tB = pairs[k].tB;
        constexpr double s = 1;  // pairs[k].sqrt_weight;

        C.block<3,3>(3*k, 0) = s * (RA - Eigen::Matrix3d::Identity());
        w.segment<3>(3*k)    = s * (RX * tB - tA);
    }

    return ridge_llsq(C, w, 1e-12);
}

// ---------- public API: linear init + non-linear refine ----------
Eigen::Affine3d estimate_hand_eye_tsai_lenz(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg)
{
    auto pairs = build_all_pairs(base_T_gripper, camera_T_target, min_angle_deg);
    const Eigen::Matrix3d RX = estimate_rotation_allpairs_weighted(pairs);
    const Eigen::Vector3d g_t_c = estimate_translation_allpairs_weighted(pairs, RX);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = RX;
    X.translation() = g_t_c;
    return X;
}

Eigen::Affine3d refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Affine3d& X0,
    const RefinementOptions& opts)
{
    auto pairs = build_all_pairs(base_T_gripper, camera_T_target, /*min_angle_deg*/ 0.5);

    // parameters: rotation quaternion + translation
    Eigen::Quaterniond q0(X0.linear());
    double q_param[4] = { q0.w(), q0.x(), q0.y(), q0.z() };
    double t_param[3] = { X0.translation().x(), X0.translation().y(), X0.translation().z() };

    ceres::Problem problem;

    for (const auto& mp : pairs) {
        auto* cost = AX_XBResidual::create(mp);
        ceres::LossFunction* loss = opts.huber_delta > 0 ?
            static_cast<ceres::LossFunction*>(new ceres::HuberLoss(opts.huber_delta)) : nullptr;
        problem.AddResidualBlock(cost, loss, q_param, t_param);
    }
    problem.SetManifold(q_param, new ceres::QuaternionManifold());

    ceres::Solver::Options options;
    options.max_num_iterations = opts.max_iterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = opts.verbose;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (opts.verbose) std::cout << summary.BriefReport() << "\n";

    Eigen::Quaterniond qf(q_param[0], q_param[1], q_param[2], q_param[3]);
    qf.normalize();
    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = qf.toRotationMatrix();
    X.translation() = Eigen::Vector3d(t_param[0], t_param[1], t_param[2]);
    return X;
}

// ---------- convenience: full pipeline ----------
Eigen::Affine3d estimate_and_refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg,
    const RefinementOptions& ro)
{
    auto X0 = estimate_hand_eye_tsai_lenz(base_T_gripper, camera_T_target, min_angle_deg);
    return refine_hand_eye(base_T_gripper, camera_T_target, X0, ro);
}

}  // namespace calib
