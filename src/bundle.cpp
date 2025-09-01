#include "calib/bundle.h"

// std
#include <numeric>
#include <array>

#include "calib/planarpose.h"

#include "residuals/bundleresidual.h"
#include "observationutils.h"
#include "ceresutils.h"

namespace calib {

template<camera_model CameraT>
struct BundleBlocks final : public ProblemParamBlocks {
    static constexpr size_t IntrSize = CameraTraits<CameraT>::param_count;
    std::array<double, 4> b_q_t;
    std::array<double, 3> b_t_t;
    std::vector<std::array<double, 4>> g_q_c;
    std::vector<std::array<double, 3>> g_t_c;
    std::vector<std::array<double, IntrSize>> intr;

    BundleBlocks(size_t numcams): g_q_c(numcams), g_t_c(numcams), intr(numcams) {}

    static BundleBlocks create(
        const std::vector<CameraT>& cameras,
        const std::vector<Eigen::Affine3d>& g_T_c,
        const Eigen::Affine3d& b_T_t)
    {
        const size_t numcams = g_T_c.size();
        BundleBlocks blocks(numcams);
        populate_quat_tran(b_T_t, blocks.b_q_t, blocks.b_t_t);
        for (size_t i = 0; i < numcams; ++i) {
            populate_quat_tran(g_T_c[i], blocks.g_q_c[i], blocks.g_t_c[i]);
            CameraTraits<CameraT>::to_array(cameras[i], blocks.intr[i]);
        }
        return blocks;
    }

    std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        for (const auto& i : intr) blocks.emplace_back(i.data(), i.size(), IntrSize);
        for (const auto& i : g_q_c) blocks.emplace_back(i.data(), i.size(), 3);  // 3 dof in unit quaternion
        for (const auto& i : g_t_c) blocks.emplace_back(i.data(), i.size(), 3);
        blocks.emplace_back(b_q_t.data(), b_q_t.size(), 3);  // 3 dof in unit quaternion
        blocks.emplace_back(b_t_t.data(), b_t_t.size(), 3);
        return blocks;
    }

    void populate_results(BundleResult<CameraT>& result) const {
        result.b_T_t = restore_pose(b_q_t, b_t_t);
        const size_t num_cams = intr.size();
        result.g_T_c.resize(num_cams);
        result.cameras.resize(num_cams);

        for (size_t c = 0; c < num_cams; ++c) {
            result.g_T_c[c] = restore_pose(g_q_c[c], g_t_c[c]);
            result.cameras[c] = CameraTraits<CameraT>::template from_array<double>(intr[c].data());
        }
    }
};

template<camera_model CameraT>
static ceres::Problem build_problem(
    const std::vector<BundleObservation>& observations,
    const BundleOptions& opts,
    BundleBlocks<CameraT>& blocks)
{
    ceres::Problem p;
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        auto loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
        p.AddResidualBlock(
            BundleReprojResidual<CameraT>::create(obs.view, obs.b_T_g),
            loss,
            blocks.b_q_t.data(), blocks.b_t_t.data(),
            blocks.g_q_c[cam].data(), blocks.g_t_c[cam].data(),
            blocks.intr[cam].data());
    }

    p.SetManifold(blocks.b_q_t.data(), new ceres::QuaternionManifold());
    for (size_t cam = 0; cam < blocks.g_q_c.size(); ++cam) {
        p.SetManifold(blocks.g_q_c[cam].data(), new ceres::QuaternionManifold());
    }

    if (!opts.optimize_target_pose) {
        p.SetParameterBlockConstant(blocks.b_q_t.data());
        p.SetParameterBlockConstant(blocks.b_t_t.data());
    }
    if (!opts.optimize_hand_eye) {
        for (auto& e : blocks.g_q_c) p.SetParameterBlockConstant(e.data());
        for (auto& e : blocks.g_t_c) p.SetParameterBlockConstant(e.data());
    }
    if (!opts.optimize_intrinsics) {
        for (auto& e : blocks.intr) p.SetParameterBlockConstant(e.data());
    } else {
        for (size_t c = 0; c < blocks.intr.size(); ++c) {
            p.SetParameterLowerBound(blocks.intr[c].data(), CameraTraits<CameraT>::idx_fx, 0.0);
            p.SetParameterLowerBound(blocks.intr[c].data(), CameraTraits<CameraT>::idx_fy, 0.0);
            if (!opts.optimize_skew) {
                p.SetManifold(blocks.intr[c].data(),
                              new ceres::SubsetManifold(BundleBlocks<CameraT>::IntrSize,
                                                        {CameraTraits<CameraT>::idx_skew}));
            }
        }
    }
    return p;
}

template<camera_model CameraT>
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

template<camera_model CameraT>
BundleResult<CameraT> optimize_bundle(
    const std::vector<BundleObservation>& observations,
    const std::vector<CameraT>& initial_cameras,
    const std::vector<Eigen::Affine3d>& init_g_T_c,
    const Eigen::Affine3d& init_b_T_t,
    const BundleOptions& opts)
{
    validate_input(observations, initial_cameras);

    auto blocks = BundleBlocks<CameraT>::create(initial_cameras, init_g_T_c, init_b_T_t);
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
    const std::vector<BundleObservation>&,
    const std::vector<Camera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&,
    const Eigen::Affine3d&,
    const BundleOptions&);

template BundleResult<ScheimpflugCamera<BrownConradyd>> optimize_bundle(
    const std::vector<BundleObservation>&,
    const std::vector<ScheimpflugCamera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&,
    const Eigen::Affine3d&,
    const BundleOptions&);

}  // namespace calib
