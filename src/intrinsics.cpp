#include "calib/intrinsics.h"

#include <algorithm>

#include "calib/distortion.h"
#include "calib/scheimpflug.h"
#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/intrinsicresidual.h"

namespace calib {

template <camera_model CameraT>
struct IntrinsicBlocks final : public ProblemParamBlocks {
    static constexpr size_t intr_size = CameraTraits<CameraT>::param_count;
    std::vector<std::array<double, 4>> c_quat_t;
    std::vector<std::array<double, 3>> c_tra_t;
    std::array<double, intr_size> intr{};

    explicit IntrinsicBlocks(size_t numviews) : c_quat_t(numviews), c_tra_t(numviews) {}

    static IntrinsicBlocks create(const CameraT& camera,
                                  const std::vector<Eigen::Isometry3d>& init_c_se3_t) {
        const size_t num_views = init_c_se3_t.size();
        IntrinsicBlocks blocks(num_views);

        CameraTraits<CameraT>::to_array(camera, blocks.intr);
        for (size_t v = 0; v < num_views; ++v) {
            populate_quat_tran(init_c_se3_t[v], blocks.c_quat_t[v], blocks.c_tra_t[v]);
        }
        return blocks;
    }

    [[nodiscard]] std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        blocks.emplace_back(intr.data(), intr.size(), intr_size);

        // Reserve space for efficiency
        blocks.reserve(1 + c_quat_t.size() + c_tra_t.size());

        // Add quaternion blocks using std::transform
        std::transform(c_quat_t.begin(), c_quat_t.end(), std::back_inserter(blocks),
                       [](const auto& i) { return ParamBlock{i.data(), i.size(), 3}; });

        // Add translation blocks using std::transform
        std::transform(c_tra_t.begin(), c_tra_t.end(), std::back_inserter(blocks),
                       [](const auto& i) { return ParamBlock{i.data(), i.size(), 3}; });

        return blocks;
    }

    void populate_result(IntrinsicsOptimizationResult<CameraT>& result) const {
        const size_t num_views = c_quat_t.size();
        result.c_se3_t.resize(num_views);

        result.camera = CameraTraits<CameraT>::template from_array<double>(intr.data());
        for (size_t v = 0; v < num_views; ++v) {
            result.c_se3_t[v] = restore_pose(c_quat_t[v], c_tra_t[v]);
        }
    }
};

template <camera_model CameraT>
static ceres::Problem build_problem(const std::vector<PlanarView>& views,
                                    const IntrinsicsOptions& opts,
                                    IntrinsicBlocks<CameraT>& blocks) {
    ceres::Problem p;
    for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
        const auto& view = views[view_idx];
        auto* loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
        p.AddResidualBlock(IntrinsicResidual<CameraT>::create(view), loss,
                           blocks.c_quat_t[view_idx].data(), blocks.c_tra_t[view_idx].data(),
                           blocks.intr.data());
    }

    for (auto& c_quat_t : blocks.c_quat_t) {
        p.SetManifold(c_quat_t.data(), new ceres::QuaternionManifold());
    }

    p.SetParameterLowerBound(blocks.intr.data(), CameraTraits<CameraT>::idx_fx, 0.0);
    p.SetParameterLowerBound(blocks.intr.data(), CameraTraits<CameraT>::idx_fy, 0.0);
    if (!opts.optimize_skew) {
        p.SetManifold(blocks.intr.data(),
                      new ceres::SubsetManifold(IntrinsicBlocks<CameraT>::intr_size,
                                                {CameraTraits<CameraT>::idx_skew}));
    }

    return p;
}

static void validate_input(const std::vector<PlanarView>& views) {
    if (views.size() < 4) {
        throw std::invalid_argument("Insufficient views for calibration (at least 4 required).");
    }
}

template <camera_model CameraT>
IntrinsicsOptimizationResult<CameraT> optimize_intrinsics(
    const std::vector<PlanarView>& views, const CameraT& init_camera,
    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts) {
    validate_input(views);

    auto blocks = IntrinsicBlocks<CameraT>::create(init_camera, init_c_se3_t);
    ceres::Problem problem = build_problem(views, opts, blocks);

    IntrinsicsOptimizationResult<CameraT> result;
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

template IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> optimize_intrinsics(
    const std::vector<PlanarView>& views, const PinholeCamera<BrownConradyd>& init_camera,
    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts);

template IntrinsicsOptimizationResult<ScheimpflugCamera<PinholeCamera<BrownConradyd>>>
optimize_intrinsics(const std::vector<PlanarView>& views,
                    const ScheimpflugCamera<PinholeCamera<BrownConradyd>>& init_camera,
                    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts);

}  // namespace calib
