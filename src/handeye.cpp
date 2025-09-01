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

#include "residuals/handeyeresidual.h"

namespace calib {

// ---------- motion pair packing ----------
Eigen::Affine3d optimize_handeye(
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
    auto X0 = estimate_handeye_dlt(base_T_gripper, camera_T_target, min_angle_deg);
    return optimize_handeye(base_T_gripper, camera_T_target, X0, ro);
}

}  // namespace calib
