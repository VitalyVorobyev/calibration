#include "calib/extrinsics.h"

#include "calib/distortion.h"
#include "calib/scheimpflug.h"
#include "calib/model/any_camera.h"
#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/extrinsicsresidual.h"

namespace calib {

struct ExtrinsicBlocks final : public ProblemParamBlocks {
  size_t intr_size;
  std::vector<std::array<double, 4>> cam_quat_ref;
  std::vector<std::array<double, 3>> cam_tran_ref;
  std::vector<std::array<double, 4>> ref_quat_tgt;
  std::vector<std::array<double, 3>> ref_tran_tgt;
  std::vector<std::vector<double>> intrinsics;
  std::vector<AnyCamera> cams;

  ExtrinsicBlocks(size_t num_cams, size_t num_views, size_t intr_sz,
                  const std::vector<AnyCamera>& cameras)
      : intr_size(intr_sz),
        cam_quat_ref(num_cams),
        cam_tran_ref(num_cams),
        ref_quat_tgt(num_views),
        ref_tran_tgt(num_views),
        intrinsics(num_cams, std::vector<double>(intr_sz)),
        cams(cameras) {}

  static ExtrinsicBlocks create(const std::vector<AnyCamera>& cameras,
                                const std::vector<Eigen::Isometry3d>& init_cam_se3_ref,
                                const std::vector<Eigen::Isometry3d>& init_ref_se3_tgt) {
    const size_t num_cams = cameras.size();
    const size_t num_views = init_ref_se3_tgt.size();
    ExtrinsicBlocks blocks(num_cams, num_views, cameras.front().params().size(), cameras);
    for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
      populate_quat_tran(init_cam_se3_ref[cam_idx], blocks.cam_quat_ref[cam_idx],
                         blocks.cam_tran_ref[cam_idx]);
      std::copy(cameras[cam_idx].params().data(),
                cameras[cam_idx].params().data() + cameras[cam_idx].params().size(),
                blocks.intrinsics[cam_idx].begin());
    }
    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
      populate_quat_tran(init_ref_se3_tgt[view_idx], blocks.ref_quat_tgt[view_idx],
                         blocks.ref_tran_tgt[view_idx]);
    }
    return blocks;
  }

  std::vector<ParamBlock> get_param_blocks() const override {
    std::vector<ParamBlock> blocks;
    for (const auto& intr : intrinsics)
      blocks.emplace_back(intr.data(), intr.size(), intr_size);
    for (const auto& quat : cam_quat_ref)
      blocks.emplace_back(quat.data(), quat.size(), 3);
    for (const auto& tran : cam_tran_ref)
      blocks.emplace_back(tran.data(), tran.size(), 3);
    for (const auto& quat : ref_quat_tgt)
      blocks.emplace_back(quat.data(), quat.size(), 3);
    for (const auto& tran : ref_tran_tgt)
      blocks.emplace_back(tran.data(), tran.size(), 3);
    return blocks;
  }

  void populate_result(ExtrinsicOptimizationResult& result) const {
    const size_t num_cams = cams.size();
    const size_t num_views = ref_quat_tgt.size();
    result.cameras.resize(num_cams);
    result.c_se3_r.resize(num_cams);
    result.r_se3_t.resize(num_views);
    for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
      AnyCamera cam = cams[cam_idx];
      cam.params() =
          Eigen::Map<const Eigen::VectorXd>(intrinsics[cam_idx].data(), intr_size);
      result.cameras[cam_idx] = cam;
      result.c_se3_r[cam_idx] =
          restore_pose(cam_quat_ref[cam_idx], cam_tran_ref[cam_idx]);
    }
    for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
      result.r_se3_t[view_idx] =
          restore_pose(ref_quat_tgt[view_idx], ref_tran_tgt[view_idx]);
    }
  }
};

static ceres::Problem build_problem(const std::vector<MulticamPlanarView>& views,
                                    const ExtrinsicOptions& options,
                                    ExtrinsicBlocks& blocks) {
  ceres::Problem problem;
  for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
    const auto& multicam_view = views[view_idx];
    for (size_t cam_idx = 0; cam_idx < blocks.cams.size(); ++cam_idx) {
      if (multicam_view[cam_idx].empty()) continue;
      auto loss = options.huber_delta > 0 ? new ceres::HuberLoss(options.huber_delta) : nullptr;
      problem.AddResidualBlock(
          ExtrinsicResidual::create(multicam_view[cam_idx], blocks.cams[cam_idx]),
          loss, blocks.cam_quat_ref[cam_idx].data(),
          blocks.cam_tran_ref[cam_idx].data(), blocks.ref_quat_tgt[view_idx].data(),
          blocks.ref_tran_tgt[view_idx].data(), blocks.intrinsics[cam_idx].data());
    }
  }

  for (auto& cam_q : blocks.cam_quat_ref)
    problem.SetManifold(cam_q.data(), new ceres::QuaternionManifold());
  for (auto& ref_q : blocks.ref_quat_tgt)
    problem.SetManifold(ref_q.data(), new ceres::QuaternionManifold());

  if (!options.optimize_intrinsics) {
    for (auto& intr : blocks.intrinsics) problem.SetParameterBlockConstant(intr.data());
  } else {
    if (!blocks.ref_quat_tgt.empty()) {
      problem.SetParameterBlockConstant(blocks.ref_quat_tgt[0].data());
      problem.SetParameterBlockConstant(blocks.ref_tran_tgt[0].data());
    }
  }

  if (!options.optimize_extrinsics) {
    for (auto& cam_q : blocks.cam_quat_ref) problem.SetParameterBlockConstant(cam_q.data());
    for (auto& cam_t : blocks.cam_tran_ref) problem.SetParameterBlockConstant(cam_t.data());
  } else {
    if (!blocks.cam_quat_ref.empty()) {
      problem.SetParameterBlockConstant(blocks.cam_quat_ref[0].data());
      problem.SetParameterBlockConstant(blocks.cam_tran_ref[0].data());
    }
  }

  for (size_t cam_idx = 0; cam_idx < blocks.cams.size(); ++cam_idx) {
    auto& intr = blocks.intrinsics[cam_idx];
    const auto& traits = blocks.cams[cam_idx].traits();
    problem.SetParameterLowerBound(intr.data(), traits.idx_fx, 0.0);
    problem.SetParameterLowerBound(intr.data(), traits.idx_fy, 0.0);
    if (!options.optimize_skew) {
      problem.SetManifold(intr.data(),
                          new ceres::SubsetManifold(blocks.intr_size,
                                                    {traits.idx_skew}));
    }
  }

  return problem;
}

static void validate_input(const std::vector<AnyCamera>& cams,
                           const std::vector<Eigen::Isometry3d>& init_c_se3_r,
                           const std::vector<Eigen::Isometry3d>& init_r_se3_t,
                           const std::vector<MulticamPlanarView>& views) {
  const size_t num_cams = cams.size();
  const size_t num_views = views.size();
  if (init_c_se3_r.size() != num_cams || init_r_se3_t.size() != num_views) {
    throw std::invalid_argument(
        "Incompatible pose vector sizes for joint optimization");
  }
}

ExtrinsicOptimizationResult optimize_extrinsics(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<AnyCamera>& init_cameras,
    const std::vector<Eigen::Isometry3d>& init_c_se3_r,
    const std::vector<Eigen::Isometry3d>& init_r_se3_t,
    const ExtrinsicOptions& opts) {
  if (init_cameras.empty()) {
    throw std::invalid_argument("No cameras provided");
  }
  validate_input(init_cameras, init_c_se3_r, init_r_se3_t, views);

  auto blocks = ExtrinsicBlocks::create(init_cameras, init_c_se3_r, init_r_se3_t);
  ceres::Problem problem = build_problem(views, opts, blocks);

  ExtrinsicOptimizationResult result;
  solve_problem(problem, opts, &result);
  blocks.populate_result(result);
  if (opts.compute_covariance) {
    auto optcov = compute_covariance(blocks, problem);
    if (optcov.has_value()) result.covariance = std::move(optcov.value());
  }
  return result;
}

}  // namespace calib
