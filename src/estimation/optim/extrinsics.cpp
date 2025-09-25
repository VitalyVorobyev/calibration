#include "calib/estimation/extrinsics.h"

// std
#include <algorithm>
#include <iterator>

#include "calib/models/distortion.h"
#include "calib/models/scheimpflug.h"
#include "detail/ceresutils.h"
#include "detail/observationutils.h"
#include "residuals/extrinsicsresidual.h"

namespace calib {

template <camera_model CameraT>
struct ExtrinsicBlocks final : public ProblemParamBlocks {
    static constexpr size_t intr_size = CameraTraits<CameraT>::param_count;
    std::vector<std::array<double, 4>> cam_quat_ref;
    std::vector<std::array<double, 3>> cam_tran_ref;
    std::vector<std::array<double, 4>> ref_quat_tgt;
    std::vector<std::array<double, 3>> ref_tran_tgt;
    std::vector<std::array<double, intr_size>> intrinsics;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    ExtrinsicBlocks(size_t num_cams, size_t num_views)
        : cam_quat_ref(num_cams),
          cam_tran_ref(num_cams),
          ref_quat_tgt(num_views),
          ref_tran_tgt(num_views),
          intrinsics(num_cams) {}

    static auto create(const std::vector<CameraT>& cameras,
                       const std::vector<Eigen::Isometry3d>& init_cam_se3_ref,
                       const std::vector<Eigen::Isometry3d>& init_ref_se3_tgt) -> ExtrinsicBlocks {
        const size_t num_cams = cameras.size();
        const size_t num_views = init_ref_se3_tgt.size();
        ExtrinsicBlocks blocks(num_cams, num_views);
        for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
            populate_quat_tran(init_cam_se3_ref[cam_idx], blocks.cam_quat_ref[cam_idx],
                               blocks.cam_tran_ref[cam_idx]);
            CameraTraits<CameraT>::to_array(cameras[cam_idx], blocks.intrinsics[cam_idx]);
        }
        for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
            populate_quat_tran(init_ref_se3_tgt[view_idx], blocks.ref_quat_tgt[view_idx],
                               blocks.ref_tran_tgt[view_idx]);
        }
        return blocks;
    }

    [[nodiscard]]
    auto get_param_blocks() const -> std::vector<ParamBlock> override {
        std::vector<ParamBlock> blocks;
        blocks.reserve(intrinsics.size() + cam_quat_ref.size() + cam_tran_ref.size() +
                       ref_quat_tgt.size() + ref_tran_tgt.size());
        std::transform(
            intrinsics.begin(), intrinsics.end(), std::back_inserter(blocks),
            [](const auto& intr) { return ParamBlock(intr.data(), intr.size(), intr_size); });
        std::transform(cam_quat_ref.begin(), cam_quat_ref.end(), std::back_inserter(blocks),
                       [](const auto& quat) { return ParamBlock(quat.data(), quat.size(), 3); });
        std::transform(cam_tran_ref.begin(), cam_tran_ref.end(), std::back_inserter(blocks),
                       [](const auto& tran) { return ParamBlock(tran.data(), tran.size(), 3); });
        std::transform(ref_quat_tgt.begin(), ref_quat_tgt.end(), std::back_inserter(blocks),
                       [](const auto& quat) { return ParamBlock(quat.data(), quat.size(), 3); });
        std::transform(ref_tran_tgt.begin(), ref_tran_tgt.end(), std::back_inserter(blocks),
                       [](const auto& tran) { return ParamBlock(tran.data(), tran.size(), 3); });
        return blocks;
    }

    void populate_result(ExtrinsicOptimizationResult<CameraT>& result) const {
        const size_t num_cams = cam_quat_ref.size();
        const size_t num_views = ref_quat_tgt.size();
        result.cameras.resize(num_cams);
        result.c_se3_r.resize(num_cams);
        result.r_se3_t.resize(num_views);
        for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
            result.cameras[cam_idx] =
                CameraTraits<CameraT>::template from_array<double>(intrinsics[cam_idx].data());
            result.c_se3_r[cam_idx] = restore_pose(cam_quat_ref[cam_idx], cam_tran_ref[cam_idx]);
        }
        for (size_t view_idx = 0; view_idx < num_views; ++view_idx) {
            result.r_se3_t[view_idx] = restore_pose(ref_quat_tgt[view_idx], ref_tran_tgt[view_idx]);
        }
    }
};

template <camera_model CameraT>
static void set_residual_blocks(ceres::Problem& problem,
                                const std::vector<MulticamPlanarView>& views,
                                const std::vector<CameraT>& cameras,
                                const ExtrinsicOptions& options, ExtrinsicBlocks<CameraT>& blocks) {
    for (size_t view_index = 0; view_index < views.size(); ++view_index) {
        const auto& multicam_view = views[view_index];
        for (size_t cam_index = 0; cam_index < cameras.size(); ++cam_index) {
            if (multicam_view[cam_index].empty()) {
                continue;
            }
            auto* loss =
                options.huber_delta > 0 ? new ceres::HuberLoss(options.huber_delta) : nullptr;
            problem.AddResidualBlock(
                ExtrinsicResidual<CameraT>::create(multicam_view[cam_index]), loss,
                blocks.cam_quat_ref[cam_index].data(), blocks.cam_tran_ref[cam_index].data(),
                blocks.ref_quat_tgt[view_index].data(), blocks.ref_tran_tgt[view_index].data(),
                blocks.intrinsics[cam_index].data());
        }
    }
}

template <camera_model CameraT>
static void set_param_constraints(ceres::Problem& problem, const ExtrinsicOptions& options,
                                  ExtrinsicBlocks<CameraT>& blocks) {
    for (auto& cam_quat : blocks.cam_quat_ref) {
        problem.SetManifold(cam_quat.data(), new ceres::QuaternionManifold());
    }
    for (auto& ref_quat : blocks.ref_quat_tgt) {
        problem.SetManifold(ref_quat.data(), new ceres::QuaternionManifold());
    }
    if (!options.optimize_intrinsics) {
        for (auto& intr : blocks.intrinsics) {
            problem.SetParameterBlockConstant(intr.data());
        }
    } else {
        if (!blocks.ref_quat_tgt.empty()) {
            problem.SetParameterBlockConstant(blocks.ref_quat_tgt[0].data());
            problem.SetParameterBlockConstant(blocks.ref_tran_tgt[0].data());
        }
    }
    if (!options.optimize_extrinsics) {
        for (auto& cam_quat : blocks.cam_quat_ref) {
            problem.SetParameterBlockConstant(cam_quat.data());
        }
        for (auto& cam_tran : blocks.cam_tran_ref) {
            problem.SetParameterBlockConstant(cam_tran.data());
        }
    } else {
        if (!blocks.cam_quat_ref.empty()) {
            problem.SetParameterBlockConstant(blocks.cam_quat_ref[0].data());
            problem.SetParameterBlockConstant(blocks.cam_tran_ref[0].data());
        }
    }
    static constexpr size_t intr_size = CameraTraits<CameraT>::param_count;
    for (auto& intr : blocks.intrinsics) {
        problem.SetParameterLowerBound(intr.data(), CameraTraits<CameraT>::idx_fx, 0.0);
        problem.SetParameterLowerBound(intr.data(), CameraTraits<CameraT>::idx_fy, 0.0);
        if (!options.optimize_skew) {
            problem.SetManifold(intr.data(), new ceres::SubsetManifold(
                                                 intr_size, {CameraTraits<CameraT>::idx_skew}));
        }
    }
}

template <camera_model CameraT>
static auto build_problem(const std::vector<MulticamPlanarView>& views,
                          const std::vector<CameraT>& cameras, const ExtrinsicOptions& options,
                          ExtrinsicBlocks<CameraT>& blocks) -> ceres::Problem {
    ceres::Problem problem;
    set_residual_blocks(problem, views, cameras, options, blocks);
    set_param_constraints(problem, options, blocks);
    return problem;
}

template <camera_model CameraT>
static void validate_input(const std::vector<CameraT>& init_cameras,
                           const std::vector<Eigen::Isometry3d>& init_c_se3_r,
                           const std::vector<Eigen::Isometry3d>& init_r_se3_t,
                           const std::vector<MulticamPlanarView>& views) {
    const size_t num_cams = init_cameras.size();
    const size_t num_views = views.size();
    if (init_c_se3_r.size() != num_cams || init_r_se3_t.size() != num_views) {
        throw std::invalid_argument("Incompatible pose vector sizes for joint optimization");
    }
}

template <camera_model CameraT>
ExtrinsicOptimizationResult<CameraT> optimize_extrinsics(
    const std::vector<MulticamPlanarView>& views, const std::vector<CameraT>& init_cameras,
    const std::vector<Eigen::Isometry3d>& init_c_se3_r,
    const std::vector<Eigen::Isometry3d>& init_r_se3_t, const ExtrinsicOptions& opts) {
    validate_input(init_cameras, init_c_se3_r, init_r_se3_t, views);

    auto blocks = ExtrinsicBlocks<CameraT>::create(init_cameras, init_c_se3_r, init_r_se3_t);
    ceres::Problem problem = build_problem(views, init_cameras, opts, blocks);

    ExtrinsicOptimizationResult<CameraT> result;
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

template ExtrinsicOptimizationResult<PinholeCamera<BrownConradyd>> optimize_extrinsics(
    const std::vector<MulticamPlanarView>&, const std::vector<PinholeCamera<BrownConradyd>>&,
    const std::vector<Eigen::Isometry3d>&, const std::vector<Eigen::Isometry3d>&,
    const ExtrinsicOptions&);

template ExtrinsicOptimizationResult<ScheimpflugCamera<PinholeCamera<BrownConradyd>>>
optimize_extrinsics(const std::vector<MulticamPlanarView>&,
                    const std::vector<ScheimpflugCamera<PinholeCamera<BrownConradyd>>>&,
                    const std::vector<Eigen::Isometry3d>&, const std::vector<Eigen::Isometry3d>&,
                    const ExtrinsicOptions&);

}  // namespace calib
