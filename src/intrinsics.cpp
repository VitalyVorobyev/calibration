#include "calib/intrinsics.h"

#include "calib/distortion.h"
#include "calib/scheimpflug.h"
#include "calib/model/any_camera.h"
#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/intrinsicresidual.h"

namespace calib {

struct IntrinsicBlocks final : public ProblemParamBlocks {
    std::vector<std::array<double, 4>> c_q_t;
    std::vector<std::array<double, 3>> c_t_t;
    std::vector<double> intr;
    AnyCamera cam;

    IntrinsicBlocks(const AnyCamera& camera, size_t numviews)
        : c_q_t(numviews), c_t_t(numviews), intr(camera.params().size()), cam(camera) {}

    static IntrinsicBlocks create(const AnyCamera& camera,
                                  const std::vector<Eigen::Isometry3d>& init_c_se3_t) {
        const size_t num_views = init_c_se3_t.size();
        IntrinsicBlocks blocks(camera, num_views);
        std::copy(camera.params().data(), camera.params().data() + camera.params().size(),
                  blocks.intr.begin());
        for (size_t v = 0; v < num_views; ++v) {
            populate_quat_tran(init_c_se3_t[v], blocks.c_q_t[v], blocks.c_t_t[v]);
        }
        return blocks;
    }

    std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        blocks.emplace_back(intr.data(), intr.size(), intr.size());
        for (const auto& i : c_q_t) blocks.emplace_back(i.data(), i.size(), 3);
        for (const auto& i : c_t_t) blocks.emplace_back(i.data(), i.size(), 3);
        return blocks;
    }

    void populate_result(IntrinsicsOptimizationResult& result) const {
        const size_t num_views = c_q_t.size();
        result.c_se3_t.resize(num_views);
        result.camera = cam;
        result.camera.params() =
            Eigen::Map<const Eigen::VectorXd>(intr.data(), intr.size());
        for (size_t v = 0; v < num_views; ++v) {
            result.c_se3_t[v] = restore_pose(c_q_t[v], c_t_t[v]);
        }
    }
};

static ceres::Problem build_problem(const std::vector<PlanarView>& views,
                                    const IntrinsicsOptions& opts,
                                    IntrinsicBlocks& blocks) {
    ceres::Problem p;
    for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
        const auto& view = views[view_idx];
        auto loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
        p.AddResidualBlock(IntrinsicResidual::create(view, blocks.cam), loss,
                           blocks.c_q_t[view_idx].data(), blocks.c_t_t[view_idx].data(),
                           blocks.intr.data());
    }

    for (auto& c_q_t : blocks.c_q_t) {
        p.SetManifold(c_q_t.data(), new ceres::QuaternionManifold());
    }

    p.SetParameterLowerBound(blocks.intr.data(), blocks.cam.traits().idx_fx, 0.0);
    p.SetParameterLowerBound(blocks.intr.data(), blocks.cam.traits().idx_fy, 0.0);
    if (!opts.optimize_skew) {
        p.SetManifold(blocks.intr.data(), new ceres::SubsetManifold(
                                                static_cast<int>(blocks.intr.size()),
                                                {blocks.cam.traits().idx_skew}));
    }

    return p;
}

static void validate_input(const std::vector<PlanarView>& views) {
    if (views.size() < 4) {
        throw std::invalid_argument("Insufficient views for calibration (at least 4 required).");
    }
}

IntrinsicsOptimizationResult optimize_intrinsics(
    const std::vector<PlanarView>& views, const AnyCamera& init_camera,
    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts) {
    validate_input(views);

    auto blocks = IntrinsicBlocks::create(init_camera, init_c_se3_t);
    ceres::Problem problem = build_problem(views, opts, blocks);

    IntrinsicsOptimizationResult result;
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

}  // namespace calib
