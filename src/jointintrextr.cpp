#include "calib/jointintrextr.h"

#include "calib/distortion.h"
#include "calib/scheimpflug.h"

#include "observationutils.h"

#include "residuals/jointextrintrresidual.h"
#include "ceresutils.h"

namespace calib {

template<camera_model CameraT>
struct JointBlocks final : public ProblemParamBlocks {
    static constexpr size_t IntrSize = CameraTraits<CameraT>::param_count;
    std::vector<std::array<double, 4>> c_q_r;
    std::vector<std::array<double, 3>> c_t_r;
    std::vector<std::array<double, 4>> r_q_t;
    std::vector<std::array<double, 3>> r_t_t;
    std::vector<std::array<double, IntrSize>> intr;

    JointBlocks(size_t numcams, size_t numviews) :
        c_q_r(numcams), c_t_r(numcams), r_q_t(numviews), r_t_t(numviews),
        intr(numcams) {}

    static JointBlocks create(
        const std::vector<CameraT>& cameras,
        const std::vector<Eigen::Affine3d>& init_c_T_r,
        const std::vector<Eigen::Affine3d>& init_r_T_t
    ) {
        const size_t num_cams = cameras.size();
        const size_t num_views = init_r_T_t.size();
        JointBlocks blocks(num_cams, num_views);

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
        ParamBlock blocks;
        for (const auto& i : intr) blocks.emplace_back(i.data(), i.size(), IntrSize);
        for (const auto& i : c_q_r) blocks.emplace_back(i.data(), i.size(), 3);  // 3 dof in unit quaternion
        for (const auto& i : c_t_r) blocks.emplace_back(i.data(), i.size(), 3);
        for (const auto& i : r_q_t) blocks.emplace_back(i.data(), i.size(), 3);  // 3 dof in unit quaternion
        for (const auto& i : r_t_t) blocks.emplace_back(i.data(), i.size(), 3);
        return blocks;
    }

    size_t total_params() const override {
        return intr.size() * IntrSize +
            c_q_r.size() * 3 + c_t_r.size() * 3 +
            r_q_t.size() * 3 + r_t_t.size() * 3;
    }

    void populate_result(JointOptimizationResult<CameraT>& )
};

template<camera_model CameraT>
static ceres::Problem build_problem(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<CameraT>& cameras,
    const OptimOptions& options,
    JointBlocks<CameraT>& blocks
) {
    ceres::Problem problem;
}

template<camera_model CameraT>
static void recover_parameters(
    const JointBlocks<CameraT>& blocks,
    JointOptimizationResult<CameraT>& result
) {
    const size_t num_cams = cam_poses.size();
    const size_t num_views = targ_poses.size();

    result.intrinsics.resize(num_cams);
    result.camera_poses.resize(num_cams);
    for (size_t i = 0; i < num_cams; ++i) {
        result.intrinsics[i] = {intr[i][0], intr[i][1], intr[i][2], intr[i][3], intr[i][4]};
        result.camera_poses[i] = pose2affine(cam_poses[i].data());
    }
    result.target_poses.resize(num_views);
    for (size_t v = 0; v < num_views; ++v) {
        result.target_poses[v] = pose2affine(targ_poses[v].data());
    }
}

template<camera_model CameraT>
static void validate_input(const std::vector<CameraT>& init_cameras,
                            const std::vector<Eigen::Affine3d>& init_c_T_r,
                            const std::vector<Eigen::Affine3d>& init_r_T_t,
                            const std::vector<ExtrinsicPlanarView>& views) {
    const size_t num_cams = init_cameras.size();
    const size_t num_views = views.size();
    if (init_c_T_r.size() != num_cams ||
        init_r_T_t.size() != num_views) {
        throw std::invalid_argument("Incompatible pose vector sizes for joint optimization");
    }
}

template<camera_model CameraT>
JointOptimizationResult<CameraT> optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<CameraT>& init_cameras,
    const std::vector<Eigen::Affine3d>& init_c_T_r,
    const std::vector<Eigen::Affine3d>& init_r_T_t,
    const OptimOptions& opts
) {
    validate_input(init_cameras, init_c_T_r, init_r_T_t, views);

    JointBlocks<CameraT> blocks = initialize_blocks(init_cameras, init_c_T_r, init_r_T_t);
    ceres::Problem problem = build_problem(views, init_cameras, opts, blocks);

    JointOptimizationResult<CameraT> result;
    solve_problem(problem, opts, result);

    recover_parameters(blocks, result);
    if (opts.compute_covariance) {
        compute_covariance(problem, blocks, result);
    }

    return result;
}

template JointOptimizationResult<Camera<BrownConradyd>> optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>&,
    const std::vector<Camera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&,
    const std::vector<Eigen::Affine3d>&,
    const OptimOptions&);

template JointOptimizationResult<ScheimpflugCamera<BrownConradyd>> optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>&,
    const std::vector<ScheimpflugCamera<BrownConradyd>>&,
    const std::vector<Eigen::Affine3d>&,
    const std::vector<Eigen::Affine3d>&,
    const OptimOptions&);

}  // namespace calib
