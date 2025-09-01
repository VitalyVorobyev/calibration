#include "calib/handeye.h"

// std
#include <array>
#include <vector>
#include <stdexcept>

// ceres
#include <ceres/ceres.h>

#include "observationutils.h"
#include "residuals/handeyeresidual.h"
#include "ceresutils.h"

namespace calib {

struct HandeyeBlocks final : public ProblemParamBlocks {
    std::array<double,4> q;
    std::array<double,3> t;

    static HandeyeBlocks create(const Eigen::Affine3d& X0) {
        HandeyeBlocks b;
        Eigen::Quaterniond q0(X0.linear());
        b.q = {q0.w(), q0.x(), q0.y(), q0.z()};
        b.t = {X0.translation().x(), X0.translation().y(), X0.translation().z()};
        return b;
    }

    std::vector<ParamBlock> get_param_blocks() const override {
        return { {q.data(), q.size(), 3}, {t.data(), t.size(), 3} };
    }

    void populate_result(HandeyeResult& res) const {
        Eigen::Quaterniond qf(q[0], q[1], q[2], q[3]);
        qf.normalize();
        Eigen::Affine3d X = Eigen::Affine3d::Identity();
        X.linear() = qf.toRotationMatrix();
        X.translation() = Eigen::Vector3d(t[0], t[1], t[2]);
        res.g_T_c = X;
    }
};

static ceres::Problem build_problem(const std::vector<MotionPair>& pairs,
                                    const HandeyeOptions& opts,
                                    HandeyeBlocks& blocks) {
    ceres::Problem problem;
    for (const auto& mp : pairs) {
        auto* cost = AX_XBResidual::create(mp);
        ceres::LossFunction* loss = opts.huber_delta > 0 ?
            static_cast<ceres::LossFunction*>(new ceres::HuberLoss(opts.huber_delta)) : nullptr;
        problem.AddResidualBlock(cost, loss, blocks.q.data(), blocks.t.data());
    }
    problem.SetManifold(blocks.q.data(), new ceres::QuaternionManifold());
    return problem;
}

HandeyeResult optimize_handeye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Affine3d& init_gripper_T_ref,
    const HandeyeOptions& opts) {
    auto pairs = build_all_pairs(base_T_gripper, camera_T_target, /*min_angle_deg*/0.5);
    auto blocks = HandeyeBlocks::create(init_gripper_T_ref);
    ceres::Problem problem = build_problem(pairs, opts, blocks);

    HandeyeResult result;
    solve_problem(problem, opts, &result);

    blocks.populate_result(result);
    if (opts.compute_covariance) {
        auto optcov = compute_covariance(blocks, problem);
        if (optcov.has_value()) {
            result.covariance = std::move(optcov.value());
        }
    }
    return result;
}

HandeyeResult estimate_and_refine_hand_eye(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    double min_angle_deg,
    const HandeyeOptions& ro) {
    Eigen::Affine3d X0 = estimate_handeye_dlt(base_T_gripper, camera_T_target, min_angle_deg);
    return optimize_handeye(base_T_gripper, camera_T_target, X0, ro);
}

} // namespace calib
