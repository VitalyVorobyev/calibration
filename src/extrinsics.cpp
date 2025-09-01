#include "calib/extrinsics.h"

#include "calib/distortion.h"
#include "calib/scheimpflug.h"

#include "residuals/extrinsicsresidual.h"
#include "observationutils.h"
#include "ceresutils.h"

namespace calib {

template<camera_model CameraT>
struct ExtrinsicBlocks final : public ProblemParamBlocks {
    static constexpr size_t IntrSize = CameraTraits<CameraT>::param_count;
    std::vector<std::array<double, 4>> c_q_r;
    std::vector<std::array<double, 3>> c_t_r;
    std::vector<std::array<double, 4>> r_q_t;
    std::vector<std::array<double, 3>> r_t_t;
    std::vector<std::array<double, IntrSize>> intr;

    ExtrinsicBlocks(size_t numcams, size_t numviews) :
        c_q_r(numcams), c_t_r(numcams), r_q_t(numviews), r_t_t(numviews),
        intr(numcams) {}

    static ExtrinsicBlocks create(
        const std::vector<CameraT>& cameras,
        const std::vector<Eigen::Affine3d>& init_c_T_r,
        const std::vector<Eigen::Affine3d>& init_r_T_t
    ) {
        const size_t num_cams = cameras.size();
        const size_t num_views = init_r_T_t.size();
        ExtrinsicBlocks blocks(num_cams, num_views);

        for (size_t i = 0; i < num_cams; ++i) {
            populate_quat_tran(init_c_T_r[i], blocks.c_q_r[i], blocks.c_t_r[i]);
            CameraTraits<CameraT>::to_array(cameras[i], blocks.intr[i]);
        }
        for (size_t v = 0; v < num_views; ++v) {
            populate_quat_tran(init_r_T_t[v], blocks.r_q_t[v], blocks.r_t_t[v]);
        }
        return blocks;
    }

    std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        for (const auto& i : intr) blocks.emplace_back(i.data(), i.size(), IntrSize);
        for (const auto& i : c_q_r) blocks.emplace_back(i.data(), i.size(), 3);  // 3 dof in unit quaternion
        for (const auto& i : c_t_r) blocks.emplace_back(i.data(), i.size(), 3);
        for (const auto& i : r_q_t) blocks.emplace_back(i.data(), i.size(), 3);  // 3 dof in unit quaternion
        for (const auto& i : r_t_t) blocks.emplace_back(i.data(), i.size(), 3);
        return blocks;
    }

    void populate_result(ExtrinsicOptimizationResult<CameraT>& result) const {
        const size_t num_cams = c_q_r.size();
        const size_t num_views = r_q_t.size();

        result.cameras.resize(num_cams);
        result.c_T_r.resize(num_cams);
        result.r_T_t.resize(num_views);

        for (size_t i = 0; i < num_cams; ++i) {
            result.cameras[i] = CameraTraits<CameraT>::template from_array<double>(intr[i].data());
            result.c_T_r[i] = restore_pose(c_q_r[i], c_t_r[i]);
        }
        for (size_t v = 0; v < num_views; ++v) {
            result.r_T_t[v] = restore_pose(r_q_t[v], r_t_t[v]);
        }
    }
};

template<camera_model CameraT>
static ceres::Problem build_problem(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<CameraT>& cameras,
    const ExtrinsicOptions& options,
    ExtrinsicBlocks<CameraT>& blocks
) {
    ceres::Problem p;
    for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
        const auto& multicam_view = views[view_idx];
        for (size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
            if (multicam_view[cam_idx].empty()) continue;

            auto loss = options.huber_delta > 0 ? new ceres::HuberLoss(options.huber_delta) : nullptr;
            p.AddResidualBlock(
                ExtrinsicResidual<CameraT>::create(multicam_view[cam_idx]),
                loss,
                blocks.c_q_r[cam_idx].data(), blocks.c_t_r[cam_idx].data(),
                blocks.r_q_t[view_idx].data(), blocks.r_t_t[view_idx].data(),
                blocks.intr[cam_idx].data());
        }
    }

    for (auto& c_q_r : blocks.c_q_r) p.SetManifold(c_q_r.data(), new ceres::QuaternionManifold());
    for (auto& r_q_t : blocks.r_q_t) p.SetManifold(r_q_t.data(), new ceres::QuaternionManifold());

    if (!options.optimize_intrinsics) {
        for (auto& intr : blocks.intr) p.SetParameterBlockConstant(intr.data());
    } else {
        if (!blocks.r_q_t.empty()) {
            p.SetParameterBlockConstant(blocks.r_q_t[0].data());
            p.SetParameterBlockConstant(blocks.r_t_t[0].data());
        }
    }

    if (!options.optimize_extrinsics) {
        for (auto& c_q_r : blocks.c_q_r) p.SetParameterBlockConstant(c_q_r.data());
        for (auto& c_t_r : blocks.c_t_r) p.SetParameterBlockConstant(c_t_r.data());
    } else {
        // Fix the reference frame to avoid gauge ambiguity
        // Fix the first camera pose (reference camera)
        if (!blocks.c_q_r.empty()) {
            p.SetParameterBlockConstant(blocks.c_q_r[0].data());
            p.SetParameterBlockConstant(blocks.c_t_r[0].data());
        }
    }

    static constexpr size_t IntrSize = CameraTraits<CameraT>::param_count;
    for (auto& intr : blocks.intr) {
        p.SetParameterLowerBound(intr.data(), CameraTraits<CameraT>::idx_fx, 0.0);
        p.SetParameterLowerBound(intr.data(), CameraTraits<CameraT>::idx_fy, 0.0);
        if (!options.optimize_skew) {
            p.SetManifold(intr.data(),
                          new ceres::SubsetManifold(IntrSize, {CameraTraits<CameraT>::idx_skew}));
        }
    }

    return p;
}

template<camera_model CameraT>
static void validate_input(const std::vector<CameraT>& init_cameras,
                           const std::vector<Eigen::Affine3d>& init_c_T_r,
                           const std::vector<Eigen::Affine3d>& init_r_T_t,
                           const std::vector<MulticamPlanarView>& views) {
    const size_t num_cams = init_cameras.size();
    const size_t num_views = views.size();
    if (init_c_T_r.size() != num_cams ||
        init_r_T_t.size() != num_views) {
        throw std::invalid_argument("Incompatible pose vector sizes for joint optimization");
    }
}

template<camera_model CameraT>
ExtrinsicOptimizationResult<CameraT> optimize_extrinsics(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<CameraT>& init_cameras,
    const std::vector<Eigen::Affine3d>& init_c_T_r,
    const std::vector<Eigen::Affine3d>& init_r_T_t,
    const ExtrinsicOptions& opts
) {
    validate_input(init_cameras, init_c_T_r, init_r_T_t, views);

    auto blocks = ExtrinsicBlocks<CameraT>::create(init_cameras, init_c_T_r, init_r_T_t);
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

template ExtrinsicOptimizationResult<Camera<BrownConradyd>> optimize_extrinsics(
    const std::vector<MulticamPlanarView>&,
    const std::vector<Camera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&,
    const std::vector<Eigen::Affine3d>&,
    const ExtrinsicOptions&);

template ExtrinsicOptimizationResult<ScheimpflugCamera<BrownConradyd>> optimize_extrinsics(
    const std::vector<MulticamPlanarView>&,
    const std::vector<ScheimpflugCamera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&,
    const std::vector<Eigen::Affine3d>&,
    const ExtrinsicOptions&);

}  // namespace calib
