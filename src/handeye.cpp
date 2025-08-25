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
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg = 1.0,          // discard too-small motions
    bool reject_axis_parallel = true,    // guard against ill-conditioning
    double axis_parallel_eps = 1e-3
) {
    if (base_T_gripper.size() < 2 || base_T_gripper.size() != camera_T_target.size()) {
        throw std::runtime_error("Inconsistent hand-eye input sizes");
    }
    const size_t n = base_T_gripper.size();
    const double min_angle = min_angle_deg * std::numbers::pi / 180.0;

    std::vector<MotionPair> pairs;
    pairs.reserve(n*(n-1)/2);

    for (size_t i = 0; i+1 < n; ++i) {
        for (size_t j = i+1; j < n; ++j) {
            const Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[j];
            const Eigen::Affine3d B = camera_T_target[i] * camera_T_target[j].inverse();

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
        const double s = pairs[k].sqrt_weight;

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
        const Eigen::Matrix3d& RB = pairs[k].RB;
        const Eigen::Vector3d& tA = pairs[k].tA;
        const Eigen::Vector3d& tB = pairs[k].tB;
        const double s = pairs[k].sqrt_weight;

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
    double min_angle_deg = 1.0,
    const RefinementOptions& ro)
{
    auto X0 = estimate_hand_eye_tsai_lenz_allpairs_weighted(base_T_gripper, camera_T_target, min_angle_deg);
    return refine_hand_eye(base_T_gripper, camera_T_target, X0, ro);
}

struct HandEyeRepParamBlocks final {
    std::array<double, 4> qX;   // ^gT_c
    std::array<double, 3> tX;
    std::array<double, 4> qBt;  // ^bT_t
    std::array<double, 3> tBt;
    std::array<double, 9> intr9; // fx,fy,cx,cy,k1,k2,p1,p2,k3
};

HandEyeRepParamBlocks init_hand_eye_reprojection_param_blocks(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const Intrinsics& intr,
    const Eigen::Affine3d& gripper_T_ref
) {
    // Parameters: qX,tX, qBt,tBt, intr9
    Eigen::Quaterniond qX0(gripper_T_ref.linear());
    std::array<double, 4> qX = { qX0.w(), qX0.x(), qX0.y(), qX0.z() };
    std::array<double, 3> tX = {
        gripper_T_ref.translation().x(),
        gripper_T_ref.translation().y(),
        gripper_T_ref.translation().z()
    };

    // Initialize ^bT_t roughly from frame 0 chain:  ^bT_t ≈ ^bT_g * X * ^cT_t0  (take ^cT_t0 ≈ identity if unknown)
    // If you already know ^cT_t for a frame, plug it here. Otherwise start near base origin.
    Eigen::Quaterniond qBt0 = Eigen::Quaterniond::Identity();
    Eigen::Vector3d tBt0 = Eigen::Vector3d::Zero();
    std::array<double, 4> qBt = { qBt0.w(), qBt0.x(), qBt0.y(), qBt0.z() };
    std::array<double, 3> tBt = { tBt0.x(), tBt0.y(), tBt0.z() };
    std::array<double, 9> intr9 = {
        intr.fx, intr.fy, intr.cx, intr.cy,
        intr.k1, intr.k2, intr.p1, intr.p2, intr.k3 };

    return { qX, tX, qBt, tBt, intr9 };
}

ceres::Problem build_hand_eye_reprojection_problem(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& image_points,
    const Intrinsics& intr,
    const ReprojRefineOptions& options,
    HandEyeRepParamBlocks &param_blocks
) {
    ceres::Problem problem;
    const size_t nviews = base_T_gripper.size();

    // Add reprojection residuals
    for (size_t k = 0; k < nviews; ++k) {
        const auto& b_T_g = base_T_gripper[k];
        for (size_t i = 0; i < object_points.size(); ++i) {
            auto* cost = HandEyeReprojResidual::create(
                b_T_g, object_points[i], image_points[k][i], 1.0, options.use_distortion);
            ceres::LossFunction* loss = options.huber_delta_px > 0
                ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(options.huber_delta_px))
                : nullptr;
            problem.AddResidualBlock(
                cost,
                loss,
                param_blocks.qX.data(),
                param_blocks.tX.data(),
                param_blocks.qBt.data(),
                param_blocks.tBt.data(),
                param_blocks.intr9.data()
            );
        }
    }

    // Manifolds
    problem.SetManifold(param_blocks.qX.data(), new ceres::QuaternionManifold());
    problem.SetManifold(param_blocks.qBt.data(), new ceres::QuaternionManifold());

    // Freeze or free intrinsics as requested
    if (!options.refine_intrinsics) {
        problem.SetParameterBlockConstant(param_blocks.intr9.data());
    } else if (!options.refine_distortion) {
        // Only fx,fy,cx,cy optimize; freeze distortion
        problem.SetParameterBlockVariable(param_blocks.intr9.data());
        problem.SetParameterLowerBound(param_blocks.intr9.data(), 0, 1e-6); // fx>0
        problem.SetParameterLowerBound(param_blocks.intr9.data(), 1, 1e-6); // fy>0
        for (int j=4;j<9;++j) problem.SetParameterLowerBound(param_blocks.intr9.data(), j, param_blocks.intr9[j]); // lock by tying lower=upper
        for (int j=4;j<9;++j) problem.SetParameterUpperBound(param_blocks.intr9.data(), j, param_blocks.intr9[j]);
    } else {
        // full intrinsics incl. distortion
        problem.SetParameterBlockVariable(param_blocks.intr9.data());
        problem.SetParameterLowerBound(param_blocks.intr9.data(), 0, 1e-6);
        problem.SetParameterLowerBound(param_blocks.intr9.data(), 1, 1e-6);
    }

    // Optional soft AX=XB prior (helps when target points geometry is weak)
    if (options.lambda_axxb > 0.0 && nviews >= 2) {
        for (size_t i = 0; i + 1 < nviews; ++i) {
            for (size_t j = i + 1; j < nviews; ++j) {
                Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[j];
                // If you also have camera_T_target estimates, you can build B here.
                // If not, skip B and rely solely on reprojection.
                // For completeness we set B=Identity (no effect). Replace with your ^cT_t estimates if available.
                Eigen::Affine3d B = Eigen::Affine3d::Identity();
                auto* cost = AXXBResidualSoft::create(
                    A.linear(), B.linear(), A.translation(), B.translation(), options.lambda_axxb);
                problem.AddResidualBlock(cost, nullptr, param_blocks.qX.data(), param_blocks.tX.data());
            }
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

static Eigen::Quaterniond array_to_norm_quat(const std::array<double, 4>& arr) {
    Eigen::Quaterniond quat(arr[0], arr[1], arr[2], arr[3]);
    quat.normalize();
    return quat;
}

HandEyeReprojectionResult compose_results(
    const ceres::Problem& problem,
    const HandEyeRepParamBlocks& blocks
) {
    HandEyeReprojectionResult result;

    Eigen::Affine3d r_T_g = Eigen::Affine3d::Identity();
    r_T_g.linear() = array_to_norm_quat(blocks.qX).toRotationMatrix();
    r_T_g.translation() << blocks.tX[0], blocks.tX[1], blocks.tX[2];
    result.r_T_g = r_T_g;

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.linear() = array_to_norm_quat(blocks.qBt).toRotationMatrix();
    b_T_t.translation() << blocks.tBt[0], blocks.tBt[1], blocks.tBt[2];
    result.b_T_t = b_T_t;

    result.intr.fx = blocks.intr9[0];
    result.intr.fy = blocks.intr9[1];
    result.intr.cx = blocks.intr9[2];
    result.intr.cy = blocks.intr9[3];
    result.intr.k1 = blocks.intr9[4];
    result.intr.k2 = blocks.intr9[5];
    result.intr.k3 = blocks.intr9[6];

    // TODO: calculate covariance and reprojection error

    return result;
}

HandEyeReprojectionResult refine_hand_eye_reprojection(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<PlanarView> observables,
    const Intrinsics& init_intr,
    const Eigen::Affine3d& init_gripper_T_ref,
    const ReprojRefineOptions& options
) {
    const size_t nviews = base_T_gripper.size();
    if (nviews == 0) throw std::runtime_error("Empty base_T_gripper");
    if (observables.size() != nviews) throw std::runtime_error("image_points size mismatch");
    for (const auto& obs : observables) {
        if (obs.empty()) throw std::runtime_error("Empty observations");
    }

    HandEyeRepParamBlocks blocks = init_hand_eye_reprojection_param_blocks(
        base_T_gripper, init_intr, init_gripper_T_ref);
    ceres::Problem problem = build_hand_eye_reprojection_problem();
    const std::string report = solve_hand_eye_reprojection(problem, options.max_iterations, options.verbose);
    auto result = compose_results(problem, blocks);
    result.report = report;
    return result;
}

}  // namespace vitavision
