#include "calibration/handeye.h"

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

namespace vitavision {

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
            if (w < min_angle) continue;

            if (reject_axis_parallel) {
                const double aa = (a < 1e-9) ? 0.0 : 1.0;
                const double bb = (b < 1e-9) ? 0.0 : 1.0;
                if (aa*bb > 0.0) {
                    double sin_axis = (alpha.normalized().cross(beta.normalized())).norm();
                    if (sin_axis < axis_parallel_eps) continue; // nearly same axis
                }
            }

            mp.sqrt_weight = std::sqrt(w);
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
        const double s = 1;  // pairs[k].sqrt_weight;

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
        const double s = 1;  // pairs[k].sqrt_weight;

        C.block<3,3>(3*k, 0) = s * (RA - Eigen::Matrix3d::Identity());
        w.segment<3>(3*k)    = s * (RX * tB - tA);
    }

    return ridge_llsq(C, w, 1e-12);
}

// ---------- public API: linear init + non-linear refine ----------
Eigen::Affine3d estimate_hand_eye_tsai_lenz_allpairs_weighted(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg)
{
    auto pairs = build_all_pairs(base_T_gripper, camera_T_target, min_angle_deg);
    const Eigen::Matrix3d RX = estimate_rotation_allpairs_weighted(pairs);
    const Eigen::Vector3d tX = estimate_translation_allpairs_weighted(pairs, RX);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = RX;
    X.translation() = tX;
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
    auto X0 = estimate_hand_eye_tsai_lenz_allpairs_weighted(base_T_gripper, camera_T_target, min_angle_deg);
    return refine_hand_eye(base_T_gripper, camera_T_target, X0, ro);
}

struct HandEyeRepParamBlocks final {
    std::array<double, 4> qX;   // g_T_c (camera to gripper)
    std::array<double, 3> tX;
    std::array<double, 4> qBt;  // b_T_t (target to base)
    std::array<double, 3> tBt;
    std::array<double, 9> intr9; // fx,fy,cx,cy,k1,k2,p1,p2,k3
};

static HandEyeRepParamBlocks init_hand_eye_reprojection_param_blocks(
    const Intrinsics& intr,
    const Eigen::Affine3d& g_T_c,
    const PlanarView& view0,
    const Eigen::Affine3d& b_T_g0
) {
    // Parameters: qX,tX, qBt,tBt, intr9
    std::array<double, 4> qX;
    std::array<double, 3> tX;
    populate_quat_tran(g_T_c, qX, tX);

    // Initialize b_T_t roughly from frame 0 chain:  b_T_t ≈ b_T_g * X * c_T_t0
    auto c_T_t0 = estimate_planar_pose_dlt(
        view0, CameraMatrix{intr.fx, intr.fy, intr.cx, intr.cy});
    const Eigen::Affine3d b_T_t0 = b_T_g0 * g_T_c * c_T_t0;
    std::array<double, 4> qBt;
    std::array<double, 3> tBt;
    populate_quat_tran(b_T_t0, qBt, tBt);

    std::array<double, 9> intr9 = {
        intr.fx, intr.fy, intr.cx, intr.cy,
        intr.k1, intr.k2, intr.p1, intr.p2, intr.k3 };

    return { qX, tX, qBt, tBt, intr9 };
}

static ceres::Problem build_hand_eye_reprojection_problem(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<PlanarView> observations,
    const ReprojRefineOptions& options,
    HandEyeRepParamBlocks &blocks
) {
    ceres::Problem problem;
    const size_t nviews = base_T_gripper.size();

    // Add reprojection residuals
    for (size_t k = 0; k < nviews; ++k) {
        const auto& b_T_g = base_T_gripper[k];
        auto* cost = HandEyeViewReprojResidual::create(
            b_T_g, observations[k], 1.0, options.use_distortion);
        ceres::LossFunction* loss = options.huber_delta_px > 0
            ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(options.huber_delta_px))
            : nullptr;
        problem.AddResidualBlock(
            cost,
            loss,
            blocks.qX.data(),
            blocks.tX.data(),
            blocks.qBt.data(),
            blocks.tBt.data(),
            blocks.intr9.data()
        );
    }

    // Manifolds
    problem.SetManifold(blocks.qX.data(), new ceres::QuaternionManifold());
    problem.SetManifold(blocks.qBt.data(), new ceres::QuaternionManifold());

    // Freeze or free intrinsics as requested
    if (!options.refine_intrinsics) {
        problem.SetParameterBlockConstant(blocks.intr9.data());
    } else {
        problem.SetParameterLowerBound(blocks.intr9.data(), 0, 1e-6); // fx > 0
        problem.SetParameterLowerBound(blocks.intr9.data(), 1, 1e-6); // fy > 0
        if (!options.refine_distortion) {
            // Only fx,fy,cx,cy optimize; freeze distortion
            problem.SetManifold(
                blocks.intr9.data(), new ceres::SubsetManifold(9, {4, 5, 6, 7, 8}));
        }
    }

    return problem;
}

std::string solve_hand_eye_reprojection(
    ceres::Problem& problem,
    int max_iterations,
    bool verbose
) {
    ceres::Solver::Options opts;
    opts.max_num_iterations = max_iterations;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = verbose;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    return summary.BriefReport();
}

static HandEyeReprojectionResult compose_results(
    const HandEyeRepParamBlocks& blocks,
    const std::string report
) {
    HandEyeReprojectionResult result;
    result.report = report;

    result.g_T_r = restore_pose(blocks.qX, blocks.tX);  // ref camera to gripper
    result.b_T_t = restore_pose(blocks.qBt, blocks.tBt);  // target to base

    result.intr.fx = blocks.intr9[0];
    result.intr.fy = blocks.intr9[1];
    result.intr.cx = blocks.intr9[2];
    result.intr.cy = blocks.intr9[3];
    result.intr.k1 = blocks.intr9[4];
    result.intr.k2 = blocks.intr9[5];
    result.intr.p1 = blocks.intr9[6];
    result.intr.p2 = blocks.intr9[7];
    result.intr.k3 = blocks.intr9[8];

    // TODO: calculate covariance and reprojection error

    return result;
}

HandEyeReprojectionResult refine_hand_eye_reprojection(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<PlanarView> observations,
    const Intrinsics& init_intr,
    const Eigen::Affine3d& init_gripper_T_ref,
    const ReprojRefineOptions& options
) {
    const size_t nviews = base_T_gripper.size();
    if (nviews == 0) throw std::runtime_error("Empty base_T_gripper");
    if (observations.size() != nviews) throw std::runtime_error("image_points size mismatch");
    for (const auto& obs : observations) {
        if (obs.empty()) throw std::runtime_error("Empty observations");
    }

    HandEyeRepParamBlocks blocks = init_hand_eye_reprojection_param_blocks(
        init_intr, init_gripper_T_ref, observations[0], base_T_gripper[0]);
    ceres::Problem problem = build_hand_eye_reprojection_problem(
        base_T_gripper, observations, options, blocks);
    const std::string report = solve_hand_eye_reprojection(
        problem, options.max_iterations, options.verbose);
    auto result = compose_results(blocks, report);
    result.intr.use_distortion = options.use_distortion;

    return result;
}

}  // namespace vitavision
