#include "calib/bundle.h"

// std
#include <array>
#include <numeric>

#include "calib/planarpose.h"
#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/bundleresidual.h"

namespace calib {

template <camera_model CameraT>
struct BundleBlocks final : public ProblemParamBlocks {
    static constexpr size_t kIntrSize = CameraTraits<CameraT>::param_count;
    std::array<double, 4> b_q_t;
    std::array<double, 3> b_t_t;
    std::vector<std::array<double, 4>> g_q_c;
    std::vector<std::array<double, 3>> g_t_c;
    std::vector<std::array<double, kIntrSize>> intr;

    explicit BundleBlocks(size_t numcams)
        : b_q_t{0.0, 0.0, 0.0, 1.0},
          b_t_t{0.0, 0.0, 0.0},
          g_q_c(numcams),
          g_t_c(numcams),
          intr(numcams) {}

    static auto create(const std::vector<CameraT>& cameras,
                       const std::vector<Eigen::Affine3d>& g_se3_c,
                       const Eigen::Affine3d& b_se3_t) -> BundleBlocks {
        const size_t num_cams = g_se3_c.size();
        BundleBlocks blocks(num_cams);
        populate_quat_tran(b_se3_t, blocks.b_q_t, blocks.b_t_t);
        for (size_t idx = 0; idx < num_cams; ++idx) {
            populate_quat_tran(g_se3_c[idx], blocks.g_q_c[idx], blocks.g_t_c[idx]);
            CameraTraits<CameraT>::to_array(cameras[idx], blocks.intr[idx]);
        }
        return blocks;
    }

    [[nodiscard]]
    auto get_param_blocks() const -> std::vector<ParamBlock> override {
        std::vector<ParamBlock> blocks;
        for (const auto& intr_block : intr) {
            blocks.emplace_back(intr_block.data(), intr_block.size(), kIntrSize);
        }
        for (const auto& quat_block : g_q_c) {
            blocks.emplace_back(quat_block.data(), quat_block.size(),
                                3);  // 3 dof in unit quaternion
        }
        for (const auto& tran_block : g_t_c) {
            blocks.emplace_back(tran_block.data(), tran_block.size(), 3);
        }
        blocks.emplace_back(b_q_t.data(), b_q_t.size(), 3);  // 3 dof in unit quaternion
        blocks.emplace_back(b_t_t.data(), b_t_t.size(), 3);
        return blocks;
    }

    void populate_results(BundleResult<CameraT>& result) const {
        result.b_se3_t = restore_pose(b_q_t, b_t_t);
        const size_t num_cams = intr.size();
        result.g_se3_c.resize(num_cams);
        result.cameras.resize(num_cams);
        for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
            result.g_se3_c[cam_idx] = restore_pose(g_q_c[cam_idx], g_t_c[cam_idx]);
            result.cameras[cam_idx] =
                CameraTraits<CameraT>::template from_array<double>(intr[cam_idx].data());
        }
    }
};

template <camera_model CameraT>
static auto build_problem(const std::vector<BundleObservation>& observations,
                          const BundleOptions& opts,
                          BundleBlocks<CameraT>& blocks) -> ceres::Problem {
    ceres::Problem problem;
    for (const auto& obs : observations) {
        const size_t cam_idx = obs.camera_index;
        auto* loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
        problem.AddResidualBlock(BundleReprojResidual<CameraT>::create(obs.view, obs.b_se3_g), loss,
                                 blocks.b_q_t.data(), blocks.b_t_t.data(),
                                 blocks.g_q_c[cam_idx].data(), blocks.g_t_c[cam_idx].data(),
                                 blocks.intr[cam_idx].data());
    }

    problem.SetManifold(blocks.b_q_t.data(), new ceres::QuaternionManifold());
    for (size_t cam_idx = 0; cam_idx < blocks.g_q_c.size(); ++cam_idx) {
        problem.SetManifold(blocks.g_q_c[cam_idx].data(), new ceres::QuaternionManifold());
    }

    if (!opts.optimize_target_pose) {
        problem.SetParameterBlockConstant(blocks.b_q_t.data());
        problem.SetParameterBlockConstant(blocks.b_t_t.data());
    }
    if (!opts.optimize_hand_eye) {
        for (auto& quat_block : blocks.g_q_c) {
            problem.SetParameterBlockConstant(quat_block.data());
        }
        for (auto& tran_block : blocks.g_t_c) {
            problem.SetParameterBlockConstant(tran_block.data());
        }
    }
    if (!opts.optimize_intrinsics) {
        for (auto& intr_block : blocks.intr) {
            problem.SetParameterBlockConstant(intr_block.data());
        }
    } else {
        for (size_t cam_idx = 0; cam_idx < blocks.intr.size(); ++cam_idx) {
            problem.SetParameterLowerBound(blocks.intr[cam_idx].data(),
                                           CameraTraits<CameraT>::idx_fx, 0.0);
            problem.SetParameterLowerBound(blocks.intr[cam_idx].data(),
                                           CameraTraits<CameraT>::idx_fy, 0.0);
            if (!opts.optimize_skew) {
                problem.SetManifold(blocks.intr[cam_idx].data(),
                                    new ceres::SubsetManifold(BundleBlocks<CameraT>::kIntrSize,
                                                              {CameraTraits<CameraT>::idx_skew}));
            }
        }
    }
    return problem;
}

template <camera_model CameraT>
void validate_input(const std::vector<BundleObservation>& observations,
                    const std::vector<CameraT>& initial_cameras) {
    const size_t num_cams = initial_cameras.size();
    if (num_cams == 0) {
        throw std::invalid_argument("No camera intrinsics provided");
    }
    if (observations.empty()) {
        throw std::invalid_argument("No observations provided");
    }
}

template <camera_model CameraT>
BundleResult<CameraT> optimize_bundle(const std::vector<BundleObservation>& observations,
                                      const std::vector<CameraT>& initial_cameras,
                                      const std::vector<Eigen::Affine3d>& init_g_se3_c,
                                      const Eigen::Affine3d& init_b_se3_t,
                                      const BundleOptions& opts) {
    validate_input(observations, initial_cameras);

    auto blocks = BundleBlocks<CameraT>::create(initial_cameras, init_g_se3_c, init_b_se3_t);
    ceres::Problem problem = build_problem(observations, opts, blocks);

    BundleResult<CameraT> result;
    solve_problem(problem, opts, &result);

    blocks.populate_results(result);
    if (opts.compute_covariance) {
        auto optcov = compute_covariance(blocks, problem);
        if (optcov.has_value()) {
            result.covariance = std::move(optcov.value());
        }
    }

    return result;
}

template BundleResult<Camera<BrownConradyd>> optimize_bundle(
    const std::vector<BundleObservation>&, const std::vector<Camera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&, const Eigen::Affine3d&, const BundleOptions&);

template BundleResult<ScheimpflugCamera<BrownConradyd>> optimize_bundle(
    const std::vector<BundleObservation>&, const std::vector<ScheimpflugCamera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&, const Eigen::Affine3d&, const BundleOptions&);

}  // namespace calib
