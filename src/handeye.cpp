#include "calib/handeye.h"

// std
#include <array>
#include <stdexcept>
#include <vector>

// ceres
#include <ceres/ceres.h>

#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/handeyeresidual.h"

namespace calib {

struct HandeyeBlocks final : public ProblemParamBlocks {
    std::array<double, 4> quat;
    std::array<double, 3> tran;

    static auto create(const Eigen::Isometry3d& init_pose) -> HandeyeBlocks {
        HandeyeBlocks blocks;
        Eigen::Quaterniond quat_init(init_pose.linear());
        blocks.quat = {quat_init.w(), quat_init.x(), quat_init.y(), quat_init.z()};
        blocks.tran = {init_pose.translation().x(), init_pose.translation().y(),
                       init_pose.translation().z()};
        return blocks;
    }

    [[nodiscard]]
    auto get_param_blocks() const -> std::vector<ParamBlock> override {
        return {{quat.data(), quat.size(), 3}, {tran.data(), tran.size(), 3}};
    }

    void populate_result(HandeyeResult& result) const {
        Eigen::Quaterniond quat_final(quat[0], quat[1], quat[2], quat[3]);
        quat_final.normalize();
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = quat_final.toRotationMatrix();
        pose.translation() = Eigen::Vector3d(tran[0], tran[1], tran[2]);
        result.g_se3_c = pose;
    }
};

static auto build_problem(const std::vector<MotionPair>& pairs, const HandeyeOptions& opts,
                          HandeyeBlocks& blocks) -> ceres::Problem {
    ceres::Problem problem;
    for (const auto& motion_pair : pairs) {
        auto* cost = AX_XBResidual::create(motion_pair);
        ceres::LossFunction* loss =
            opts.huber_delta > 0
                ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(opts.huber_delta))
                : nullptr;
        problem.AddResidualBlock(cost, loss, blocks.quat.data(), blocks.tran.data());
    }
    problem.SetManifold(blocks.quat.data(), new ceres::QuaternionManifold());
    return problem;
}

auto optimize_handeye(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                      const std::vector<Eigen::Isometry3d>& camera_se3_target,
                      const Eigen::Isometry3d& init_gripper_se3_ref,
                      const HandeyeOptions& opts) -> HandeyeResult {
    constexpr double k_min_angle_deg = 0.5;
    auto pairs = build_all_pairs(base_se3_gripper, camera_se3_target, k_min_angle_deg);
    auto blocks = HandeyeBlocks::create(init_gripper_se3_ref);
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

auto estimate_and_optimize_handeye(const std::vector<Eigen::Isometry3d>& base_se3_gripper,
                                   const std::vector<Eigen::Isometry3d>& camera_se3_target,
                                   double min_angle_deg,
                                   const HandeyeOptions& opts) -> HandeyeResult {
    Eigen::Isometry3d init_pose =
        estimate_handeye_dlt(base_se3_gripper, camera_se3_target, min_angle_deg);
    return optimize_handeye(base_se3_gripper, camera_se3_target, init_pose, opts);
}

}  // namespace calib
