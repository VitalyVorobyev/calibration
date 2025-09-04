#include "calib/bundle.h"

#include <array>
#include <numeric>

#include "calib/planarpose.h"
#include "calib/model/any_camera.h"
#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/bundleresidual.h"

namespace calib {

struct BundleBlocks final : public ProblemParamBlocks {
  size_t intr_size;
  std::array<double, 4> b_q_t;
  std::array<double, 3> b_t_t;
  std::vector<std::array<double, 4>> g_q_c;
  std::vector<std::array<double, 3>> g_t_c;
  std::vector<std::vector<double>> intr;
  std::vector<AnyCamera> cams;

  BundleBlocks(size_t numcams, size_t intrsz, const std::vector<AnyCamera>& cameras)
      : intr_size(intrsz),
        b_q_t{0.0, 0.0, 0.0, 1.0},
        b_t_t{0.0, 0.0, 0.0},
        g_q_c(numcams),
        g_t_c(numcams),
        intr(numcams, std::vector<double>(intrsz)),
        cams(cameras) {}

  static BundleBlocks create(const std::vector<AnyCamera>& cameras,
                             const std::vector<Eigen::Isometry3d>& g_se3_c,
                             const Eigen::Isometry3d& b_se3_t) {
    const size_t num_cams = g_se3_c.size();
    BundleBlocks blocks(num_cams, cameras.front().params().size(), cameras);
    populate_quat_tran(b_se3_t, blocks.b_q_t, blocks.b_t_t);
    for (size_t idx = 0; idx < num_cams; ++idx) {
      populate_quat_tran(g_se3_c[idx], blocks.g_q_c[idx], blocks.g_t_c[idx]);
      std::copy(cameras[idx].params().data(),
                cameras[idx].params().data() + cameras[idx].params().size(),
                blocks.intr[idx].begin());
    }
    return blocks;
  }

  std::vector<ParamBlock> get_param_blocks() const override {
    std::vector<ParamBlock> blocks;
    for (const auto& intr_block : intr)
      blocks.emplace_back(intr_block.data(), intr_block.size(), intr_size);
    for (const auto& quat_block : g_q_c)
      blocks.emplace_back(quat_block.data(), quat_block.size(), 3);
    for (const auto& tran_block : g_t_c)
      blocks.emplace_back(tran_block.data(), tran_block.size(), 3);
    blocks.emplace_back(b_q_t.data(), b_q_t.size(), 3);
    blocks.emplace_back(b_t_t.data(), b_t_t.size(), 3);
    return blocks;
  }

  void populate_results(BundleResult& result) const {
    result.b_se3_t = restore_pose(b_q_t, b_t_t);
    const size_t num_cams = intr.size();
    result.g_se3_c.resize(num_cams);
    result.cameras.resize(num_cams);
    for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
      result.g_se3_c[cam_idx] = restore_pose(g_q_c[cam_idx], g_t_c[cam_idx]);
      AnyCamera cam = cams[cam_idx];
      cam.params() =
          Eigen::Map<const Eigen::VectorXd>(intr[cam_idx].data(), intr_size);
      result.cameras[cam_idx] = cam;
    }
  }
};

static ceres::Problem build_problem(const std::vector<BundleObservation>& observations,
                                    const BundleOptions& opts,
                                    BundleBlocks& blocks) {
  ceres::Problem problem;
  for (const auto& obs : observations) {
    const size_t cam_idx = obs.camera_index;
    auto* loss =
        opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
    problem.AddResidualBlock(
        BundleReprojResidual::create(obs.view, obs.b_se3_g, blocks.cams[cam_idx]),
        loss, blocks.b_q_t.data(), blocks.b_t_t.data(),
        blocks.g_q_c[cam_idx].data(), blocks.g_t_c[cam_idx].data(),
        blocks.intr[cam_idx].data());
  }

  problem.SetManifold(blocks.b_q_t.data(), new ceres::QuaternionManifold());
  for (auto& q : blocks.g_q_c)
    problem.SetManifold(q.data(), new ceres::QuaternionManifold());

  if (!opts.optimize_target_pose) {
    problem.SetParameterBlockConstant(blocks.b_q_t.data());
    problem.SetParameterBlockConstant(blocks.b_t_t.data());
  }
  if (!opts.optimize_hand_eye) {
    for (auto& quat_block : blocks.g_q_c)
      problem.SetParameterBlockConstant(quat_block.data());
    for (auto& tran_block : blocks.g_t_c)
      problem.SetParameterBlockConstant(tran_block.data());
  }
  if (!opts.optimize_intrinsics) {
    for (auto& intr_block : blocks.intr)
      problem.SetParameterBlockConstant(intr_block.data());
  } else {
    for (size_t cam_idx = 0; cam_idx < blocks.intr.size(); ++cam_idx) {
      auto& intr_block = blocks.intr[cam_idx];
      const auto& traits = blocks.cams[cam_idx].traits();
      problem.SetParameterLowerBound(intr_block.data(), traits.idx_fx, 0.0);
      problem.SetParameterLowerBound(intr_block.data(), traits.idx_fy, 0.0);
      if (!opts.optimize_skew) {
        problem.SetManifold(intr_block.data(),
                            new ceres::SubsetManifold(blocks.intr_size,
                                                      {traits.idx_skew}));
      }
    }
  }
  return problem;
}

static void validate_input(const std::vector<BundleObservation>& observations,
                           const std::vector<AnyCamera>& initial_cameras) {
  if (initial_cameras.empty()) {
    throw std::invalid_argument("No camera intrinsics provided");
  }
  if (observations.empty()) {
    throw std::invalid_argument("No observations provided");
  }
}

BundleResult optimize_bundle(const std::vector<BundleObservation>& observations,
                             const std::vector<AnyCamera>& initial_cameras,
                             const std::vector<Eigen::Isometry3d>& init_g_se3_c,
                             const Eigen::Isometry3d& init_b_se3_t,
                             const BundleOptions& opts) {
  validate_input(observations, initial_cameras);

  auto blocks = BundleBlocks::create(initial_cameras, init_g_se3_c, init_b_se3_t);
  ceres::Problem problem = build_problem(observations, opts, blocks);

  BundleResult result;
  solve_problem(problem, opts, &result);

  blocks.populate_results(result);
  if (opts.compute_covariance) {
    auto optcov = compute_covariance(blocks, problem);
    if (optcov.has_value()) result.covariance = std::move(optcov.value());
  }

  return result;
}

}  // namespace calib
