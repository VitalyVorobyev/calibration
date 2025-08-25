#include "calibration/handeyedlt.h"

// std
#include <iostream>

#include "observationutils.h"
#include "handeyeresidual.h"

namespace vitavision {

/**
 * @brief Estimates the hand-eye rotation matrix using a set of motion pairs.
 *
 * This function computes the rotation matrix that aligns the motion of a
 * gripper (base_T_gripper) with the motion of a camera (camera_T_target).
 * It uses the logarithm of rotation matrices to calculate the relative
 * rotations between consecutive poses and solves a least-squares problem
 * to estimate the rotation matrix.
 *
 * @param base_T_gripper A vector of affine transformations representing
 *                       the poses of the gripper relative to the base.
 * @param camera_T_target A vector of affine transformations representing
 *                        the poses of the target relative to the camera.
 *
 * @return The estimated 3x3 rotation matrix.
 *
 * @throws std::runtime_error If the input vectors are empty or have
 *                            inconsistent sizes.
 */
static Eigen::Matrix3d estimate_hand_eye_rotation(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target
) {
    if (base_T_gripper.empty() || base_T_gripper.size() != camera_T_target.size()) {
        std::cerr << "Inconsistent number of poses\n";
        throw std::runtime_error("Inconsistent number of poses");
    }
    const size_t m = base_T_gripper.size() - 1;  // number of motion pairs
    Eigen::MatrixXd M(3*m, 3);
    Eigen::VectorXd d(3*m);

    for (size_t i = 0; i < m; ++i) {
        auto alpha = log_rot(base_T_gripper[i].linear().transpose() * base_T_gripper[i+1].linear());
        auto beta = log_rot(camera_T_target[i].linear() * camera_T_target[i+1].linear().transpose());
        M.block<3,3>(3*i,0) = skew(alpha + beta);
        d.segment<3>(3*i) = beta - alpha;
    }

    Eigen::Vector3d r = solve_llsq(M, d);
    double angle = r.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (angle > 1e-12) {
        R = Eigen::AngleAxisd(angle, r.normalized()).toRotationMatrix();
    }
    return R;
}

/**
 * @brief Estimates the translation component of the hand-eye calibration problem.
 *
 * This function computes the translation vector `t` that aligns the motion of a
 * robotic gripper (base_T_gripper) with the motion of a camera observing a target
 * (camera_T_target), given a known rotation matrix `R`.
 *
 * @param base_T_gripper A vector of affine transformations representing the poses
 *        of the robotic gripper in the base frame. Each transformation corresponds
 *        to a specific time step.
 * @param camera_T_target A vector of affine transformations representing the poses
 *        of the camera relative to the observed target. The size of this vector
 *        must match the size of `base_T_gripper`.
 * @param R A 3x3 rotation matrix representing the rotational component of the
 *        hand-eye calibration.
 * @return Eigen::Vector3d The estimated translation vector `t` that aligns the
 *         motion of the gripper with the motion of the camera.
 *
 * @throws std::runtime_error If the input vectors are empty or their sizes do not match.
 */
static Eigen::Vector3d estimate_hand_eye_translation(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Matrix3d& R
) {
    if (base_T_gripper.empty() || base_T_gripper.size() != camera_T_target.size()) {
        std::cerr << "Inconsistent number of poses\n";
        throw std::runtime_error("Inconsistent number of poses");
    }
    const size_t m = base_T_gripper.size() - 1;  // number of motion pairs

    Eigen::MatrixXd C(3*m, 3);
    Eigen::VectorXd w(3*m);
    for (size_t i = 0; i < m; ++i) {
        Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[i+1];
        Eigen::Affine3d B = camera_T_target[i] * camera_T_target[i+1].inverse();
        C.block<3,3>(3*i,0) = A.linear() - Eigen::Matrix3d::Identity();
        w.segment<3>(3*i) = R * B.translation() - A.translation();
    }
    Eigen::Vector3d t = solve_llsq(C, w);
    return t;
}

/**
 * Compute an initial estimate of the hand-eye transform (camera -> gripper)
 * using the Tsai-Lenz linear method.  The input vectors must contain
 * corresponding poses of the robot end-effector in the base frame and poses of
 * the planar target in the camera frame for the same time instants.
 * @param base_T_gripper A vector of affine transformations representing the poses
 *        of the robotic gripper in the base frame.
 * @param camera_T_target A vector of affine transformations representing the poses
 *        of the camera relative to the observed target.
 * @return The estimated hand-eye transform (camera -> gripper).
 */
Eigen::Affine3d estimate_hand_eye_tsai_lenz(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target) {

    const size_t n = base_T_gripper.size();
    if (n < 2 || n != camera_T_target.size()) {
        std::cerr << "Insufficient data for initial hand-eye estimate\n";
        return Eigen::Affine3d::Identity();
    }
    auto R = estimate_hand_eye_rotation(base_T_gripper, camera_T_target);
    auto t = estimate_hand_eye_translation(base_T_gripper, camera_T_target, R);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = R;
    X.translation() = t;
    return X;
}

// hand_eye_tsai_lenz.hpp
#pragma once

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// std
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <numbers>

// ---------- motion pair packing ----------
struct MotionPair final {
    Eigen::Matrix3d RA, RB;
    Eigen::Vector3d tA, tB;
    double sqrt_weight; // sqrt of weight used to scale rows/residuals
};

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
    return expSO3(r); // NOTE: no transpose here
}

// ---------- weighted Tsai–Lenz translation over all pairs ----------
inline Eigen::Vector3d estimate_translation_allpairs_weighted(
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

// ---------- Ceres residual (AX = XB): rotation log + translation eq ----------
struct AX_XBResidual {
    Eigen::Matrix3d RA_, RB_;
    Eigen::Vector3d tA_, tB_;
    double sqrt_w_;

    AX_XBResidual(const MotionPair& mp) : RA_(mp.RA), RB_(mp.RB), tA_(mp.tA), tB_(mp.tB), sqrt_w_(mp.sqrt_weight) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t, T* residuals) const {
        // q = [w, x, y, z], t = [tx, ty, tz]
        Eigen::Matrix<T,3,3> RX;
        {
            T Rdata[9];
            ceres::QuaternionToRotation(q, Rdata); // column-major
            RX << Rdata[0], Rdata[3], Rdata[6],
                  Rdata[1], Rdata[4], Rdata[7],
                  Rdata[2], Rdata[5], Rdata[8];
        }
        const Eigen::Matrix<T,3,3> RXT = RX.transpose();

        const Eigen::Matrix<T,3,3> RA = RA_.cast<T>();
        const Eigen::Matrix<T,3,3> RB = RB_.cast<T>();
        const Eigen::Matrix<T,3,3> RS = RA * RX * RB * RXT;

        // rot residual: angle-axis of RS
        T RS_cols[9] = {
            RS(0,0), RS(1,0), RS(2,0),
            RS(0,1), RS(1,1), RS(2,1),
            RS(0,2), RS(1,2), RS(2,2)
        };
        T aa[3];
        ceres::RotationMatrixToAngleAxis(RS_cols, aa);

        // trans residual from (RA - I) tX = RX tB - tA
        const Eigen::Matrix<T,3,1> tX(t[0], t[1], t[2]);
        const Eigen::Matrix<T,3,1> tA = tA_.cast<T>();
        const Eigen::Matrix<T,3,1> tB = tB_.cast<T>();
        const Eigen::Matrix<T,3,1> et = (RA - Eigen::Matrix<T,3,3>::Identity()) * tX - (RX * tB - tA);

        const T s = T(sqrt_w_);
        residuals[0] = s * aa[0];
        residuals[1] = s * aa[1];
        residuals[2] = s * aa[2];
        residuals[3] = s * et(0);
        residuals[4] = s * et(1);
        residuals[5] = s * et(2);
        return true;
    }

    static auto create() {

    }
};

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

Eigen::Affine3d refine_hand_eye_ceres(
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
        auto* cost = new ceres::AutoDiffCostFunction<AX_XBResidual, 6, 4, 3>(new AX_XBResidual(mp));
        ceres::LossFunction* loss = opts.huber_delta > 0 ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(opts.huber_delta))
                                                         : nullptr;
        problem.AddResidualBlock(cost, loss, q_param, t_param);
    }

    problem.SetParameterization(q_param, new ceres::QuaternionParameterization());

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
    return refine_hand_eye_ceres(base_T_gripper, camera_T_target, X0, ro);
}

Eigen::Affine3d refine_hand_eye_reprojection(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<std::vector<Eigen::Vector2d>>& image_points,
    const Intrinsics& intr,
    const Eigen::Affine3d& X0,
    const ReprojRefineOptions& options,
    Intrinsics& out_intr,
    Eigen::Affine3d& out_b_T_t
){
    const size_t n = base_T_gripper.size();
    if (n == 0) throw std::runtime_error("Empty base_T_gripper");
    if (image_points.size() != n) throw std::runtime_error("image_points size mismatch");
    for (size_t k=0; k<n; ++k) {
        if (image_points[k].size() != object_points.size())
            throw std::runtime_error("Per-frame image points size mismatch object points size");
    }

    // Parameters: qX,tX, qBt,tBt, intr9
    Eigen::Quaterniond qX0(X0.linear());
    double qX[4] = { qX0.w(), qX0.x(), qX0.y(), qX0.z() };
    double tX[3] = { X0.translation().x(), X0.translation().y(), X0.translation().z() };

    // Initialize ^bT_t roughly from frame 0 chain:  ^bT_t ≈ ^bT_g * X * ^cT_t0  (take ^cT_t0 ≈ identity if unknown)
    // If you already know ^cT_t for a frame, plug it here. Otherwise start near base origin.
    Eigen::Quaterniond qBt0 = Eigen::Quaterniond::Identity();
    Eigen::Vector3d tBt0 = Eigen::Vector3d::Zero();
    double qBt[4] = { qBt0.w(), qBt0.x(), qBt0.y(), qBt0.z() };
    double tBt[3] = { tBt0.x(), tBt0.y(), tBt0.z() };

    double intr_param[9] = { intr.fx, intr.fy, intr.cx, intr.cy,
                             intr.k1, intr.k2, intr.p1, intr.p2, intr.k3 };

    ceres::Problem problem;

    // Add reprojection residuals
    for (size_t k=0; k<n; ++k) {
        const auto& Tbg = base_T_gripper[k];
        for (size_t i=0; i<object_points.size(); ++i) {
            auto* cost = new ceres::AutoDiffCostFunction<ReprojResidual, 2, 4, 3, 4, 3, 9>(
                new ReprojResidual(Tbg, object_points[i], image_points[k][i], 1.0, options.use_distortion)
            );
            ceres::LossFunction* loss = options.huber_delta_px > 0
                ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(options.huber_delta_px))
                : nullptr;
            problem.AddResidualBlock(cost, loss, qX, tX, qBt, tBt, intr_param);
        }
    }

    // Manifolds
    problem.SetParameterization(qX,  new ceres::QuaternionParameterization());
    problem.SetParameterization(qBt, new ceres::QuaternionParameterization());

    // Freeze or free intrinsics as requested
    if (!options.refine_intrinsics) {
        problem.SetParameterBlockConstant(intr_param);
    } else if (!options.refine_distortion) {
        // Only fx,fy,cx,cy optimize; freeze distortion
        problem.SetParameterBlockVariable(intr_param);
        problem.SetParameterLowerBound(intr_param, 0, 1e-6); // fx>0
        problem.SetParameterLowerBound(intr_param, 1, 1e-6); // fy>0
        for (int j=4;j<9;++j) problem.SetParameterLowerBound(intr_param, j, intr_param[j]); // lock by tying lower=upper
        for (int j=4;j<9;++j) problem.SetParameterUpperBound(intr_param, j, intr_param[j]);
    } else {
        // full intrinsics incl. distortion
        problem.SetParameterBlockVariable(intr_param);
        problem.SetParameterLowerBound(intr_param, 0, 1e-6);
        problem.SetParameterLowerBound(intr_param, 1, 1e-6);
    }

    // Optional soft AX=XB prior (helps when target points geometry is weak)
    if (options.lambda_axxb > 0.0 && n >= 2) {
        for (size_t i=0;i+1<n;++i) {
            for (size_t j=i+1;j<n;++j) {
                Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[j];
                // If you also have camera_T_target estimates, you can build B here.
                // If not, skip B and rely solely on reprojection.
                // For completeness we set B=Identity (no effect). Replace with your ^cT_t estimates if available.
                Eigen::Affine3d B = Eigen::Affine3d::Identity();
                auto* cost = new ceres::AutoDiffCostFunction<AXXBResidualSoft, 6, 4, 3>(
                    new AXXBResidualSoft(A.linear(), B.linear(), A.translation(), B.translation(), options.lambda_axxb)
                );
                problem.AddResidualBlock(cost, nullptr, qX, tX);
            }
        }
    }

    // Solve
    ceres::Solver::Options opts;
    opts.max_num_iterations = options.max_iterations;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = options.verbose;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    if (options.verbose) std::cout << summary.BriefReport() << "\n";

    // Pack results
    Eigen::Quaterniond qXf(qX[0], qX[1], qX[2], qX[3]); qXf.normalize();
    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = qXf.toRotationMatrix();
    X.translation() = Eigen::Vector3d(tX[0], tX[1], tX[2]);

    Eigen::Quaterniond qBtf(qBt[0], qBt[1], qBt[2], qBt[3]); qBtf.normalize();
    out_b_T_t = Eigen::Affine3d::Identity();
    out_b_T_t.linear() = qBtf.toRotationMatrix();
    out_b_T_t.translation() = Eigen::Vector3d(tBt[0], tBt[1], tBt[2]);

    out_intr = intr;
    out_intr.fx = intr_param[0]; out_intr.fy = intr_param[1];
    out_intr.cx = intr_param[2]; out_intr.cy = intr_param[3];
    out_intr.k1 = intr_param[4]; out_intr.k2 = intr_param[5];
    out_intr.p1 = intr_param[6]; out_intr.p2 = intr_param[7];
    out_intr.k3 = intr_param[8];
    out_intr.use_distortion = options.use_distortion;

    return X;
}

}  // namespace vitavision
