#include "calibration/jointintrextr.h"

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "observationutils.h"

namespace vitavision {

using Pose6 = Eigen::Matrix<double, 6, 1>;

struct JointResidual {
    PlanarView obs_;
    int num_radial_;

    JointResidual(PlanarView obs, const Eigen::VectorXd& dist)
        : obs_(std::move(obs)),
          num_radial_(std::max<int>(0, static_cast<int>(dist.size()) - 2)) {}

    template <typename T>
    bool operator()(const T* intr, const T* cam_pose6, const T* target_pose6, T* residuals) const {
        std::vector<Observation<T>> o(obs_.size());

        auto pose_cam = pose2affine(cam_pose6);
        auto pose_target = pose2affine(target_pose6);
        planar_observables_to_observables(obs_, o, pose_cam * pose_target);

        auto dr = fit_distortion_full(o, intr[0], intr[1], intr[2], intr[3], num_radial_);
        if (!dr.has_value()) throw std::runtime_error("Distortion fitting failed");
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    static auto* create(PlanarView obs, const Eigen::VectorXd& dist) {
        auto* functor = new JointResidual(obs, dist);
        auto* cost = new ceres::AutoDiffCostFunction<JointResidual,
                                                     ceres::DYNAMIC,
                                                     4,6,6>(functor,
                                                            static_cast<int>(obs.size())*2);
        return cost;
    }
};

static void setup_joint_problem(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& initial_cameras,
    std::vector<std::array<double, 4>>& intr,
    std::vector<Pose6>& cam_poses,
    std::vector<Pose6>& targ_poses,
    ceres::Problem& problem
) {
    const size_t num_cams = initial_cameras.size();
    const size_t num_views = views.size();

    for (size_t i = 0; i < num_cams; ++i) {
        problem.AddParameterBlock(intr[i].data(), 4);
        problem.AddParameterBlock(cam_poses[i].data(), 6);
    }

    // fix SE(3) gauge ambiguity
    if (!cam_poses.empty()) {
        problem.SetParameterBlockConstant(cam_poses[0].data());
    }
    for (size_t v = 0; v < num_views; ++v) {
        problem.AddParameterBlock(targ_poses[v].data(), 6);
    }
    // Fix the first target pose to remove the overall scale ambiguity between
    // target translation and the focal length. Without anchoring one target
    // pose, the optimisation can trade off focal length against scene scale
    // and still achieve zero reprojection error.  Keeping the first target
    // constant defines the world scale and allows intrinsics to be recovered.
    if (!targ_poses.empty()) {
        problem.SetParameterBlockConstant(targ_poses[0].data());
    }

    // Residuals
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        for (size_t c = 0; c < num_cams && c < view.size(); ++c) {
            const auto& obs = view[c];
            if (obs.empty()) continue;
            auto* cost = JointResidual::create(obs, initial_cameras[c].distortion.forward);
            problem.AddResidualBlock(cost, nullptr,
                                     intr[c].data(), cam_poses[c].data(), targ_poses[v].data());
        }
    }
}

static void extract_joint_solution(
    const std::vector<std::array<double, 4>>& intr,
    const std::vector<Pose6>& cam_poses,
    const std::vector<Pose6>& targ_poses,
    JointOptimizationResult& result
) {
    const size_t num_cams = cam_poses.size();
    const size_t num_views = targ_poses.size();

    result.intrinsics.resize(num_cams);
    result.camera_poses.resize(num_cams);
    for (size_t i = 0; i < num_cams; ++i) {
        result.intrinsics[i] = {intr[i][0], intr[i][1], intr[i][2], intr[i][3]};
        result.camera_poses[i] = pose2affine(cam_poses[i].data());
    }
    result.target_poses.resize(num_views);
    for (size_t v = 0; v < num_views; ++v) {
        result.target_poses[v] = pose2affine(targ_poses[v].data());
    }
}

static std::pair<double, size_t> compute_joint_residual_stats(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras,
    JointOptimizationResult& result
) {
    const size_t num_views = views.size();
    const size_t num_cams = cameras.size();

    std::vector<std::vector<Observation<double>>> per_cam_obs(num_cams);
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        for (size_t c = 0; c < num_cams && c < view.size(); ++c) {
            const auto& obs = view[c];
            for (const auto& ob : obs) {
                Eigen::Vector3d P = result.camera_poses[c] * result.target_poses[v]
                                    * Eigen::Vector3d(ob.object_xy.x(), ob.object_xy.y(), 0.0);
                double xn = P.x() / P.z();
                double yn = P.y() / P.z();
                per_cam_obs[c].push_back(Observation<double>{xn, yn,
                                                             ob.image_uv.x(),
                                                             ob.image_uv.y()});
            }
        }
    }

    result.distortions.assign(num_cams, Eigen::VectorXd());
    double ssr = 0.0; size_t count = 0;
    for (size_t c = 0; c < num_cams; ++c) {
        if (per_cam_obs[c].empty()) continue;
        int num_radial = std::max<int>(0, static_cast<int>(cameras[c].distortion.forward.size()) - 2);
        auto dr = fit_distortion_full(per_cam_obs[c],
                                      result.intrinsics[c].fx,
                                      result.intrinsics[c].fy,
                                      result.intrinsics[c].cx,
                                      result.intrinsics[c].cy,
                                      num_radial);
        if (dr) {
            result.distortions[c] = dr->distortion;
            ssr += dr->residuals.squaredNorm();
            count += dr->residuals.size();
        }
    }

    return { ssr, count };
}

static void compute_joint_covariance(
    ceres::Problem& problem,
    const std::vector<std::array<double, 4>>& intr,
    const std::vector<Pose6>& cam_poses,
    const std::vector<Pose6>& targ_poses,
    double sigma2,
    JointOptimizationResult& result
) {
    const size_t num_views = targ_poses.size();
    const size_t num_cams = cam_poses.size();

    result.intrinsic_covariances.assign(num_cams, Eigen::Matrix4d::Zero());
    result.camera_covariances.assign(num_cams, Eigen::Matrix<double,6,6>::Zero());
    result.target_covariances.assign(num_views, Eigen::Matrix<double,6,6>::Zero());

    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> blocks;
    for (size_t i = 0; i < num_cams; ++i) {
        blocks.emplace_back(intr[i].data(), intr[i].data());
        if (!problem.IsParameterBlockConstant(cam_poses[i].data())) {
            blocks.emplace_back(cam_poses[i].data(), cam_poses[i].data());
        }
    }
    for (size_t v = 0; v < num_views; ++v) {
        blocks.emplace_back(targ_poses[v].data(), targ_poses[v].data());
    }
    if (cov.Compute(blocks, &problem)) {
        for (size_t i = 0; i < num_cams; ++i) {
            double Cov4x4[16];
            cov.GetCovarianceBlock(intr[i].data(), intr[i].data(), Cov4x4);
            Eigen::Map<Eigen::Matrix4d> C4(Cov4x4);
            result.intrinsic_covariances[i] = sigma2 * C4;
            if (!problem.IsParameterBlockConstant(cam_poses[i].data())) {
                double Cov6x6[36];
                cov.GetCovarianceBlock(cam_poses[i].data(), cam_poses[i].data(), Cov6x6);
                Eigen::Map<Eigen::Matrix<double,6,6>> C6(Cov6x6);
                result.camera_covariances[i] = sigma2 * C6;
            }
        }
        for (size_t v = 0; v < num_views; ++v) {
            double Cov6x6[36];
            cov.GetCovarianceBlock(targ_poses[v].data(), targ_poses[v].data(), Cov6x6);
            Eigen::Map<Eigen::Matrix<double,6,6>> C6(Cov6x6);
            result.target_covariances[v] = sigma2 * C6;
        }
    }
}

static void initialize_pose_vectors(const std::vector<Eigen::Affine3d>& initial_camera_poses,
                                    const std::vector<Eigen::Affine3d>& initial_target_poses,
                                    std::vector<Pose6>& cam_poses,
                                    std::vector<Pose6>& targ_poses) {
    for (size_t i = 0; i < initial_camera_poses.size(); ++i) {
        ceres::RotationMatrixToAngleAxis(initial_camera_poses[i].rotation().data(), cam_poses[i].data());
        cam_poses[i][3] = initial_camera_poses[i].translation().x();
        cam_poses[i][4] = initial_camera_poses[i].translation().y();
        cam_poses[i][5] = initial_camera_poses[i].translation().z();
    }
    for (size_t j = 0; j < initial_target_poses.size(); ++j) {
        ceres::RotationMatrixToAngleAxis(initial_target_poses[j].rotation().data(), targ_poses[j].data());
        targ_poses[j][3] = initial_target_poses[j].translation().x();
        targ_poses[j][4] = initial_target_poses[j].translation().y();
        targ_poses[j][5] = initial_target_poses[j].translation().z();
    }
}

// Joint optimization of camera intrinsics, extrinsic poses and target poses
JointOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose
) {
    JointOptimizationResult result;
    const size_t num_cams = initial_cameras.size();
    const size_t num_views = views.size();
    if (initial_camera_poses.size() != num_cams ||
        initial_target_poses.size() != num_views) {
        throw std::invalid_argument("Incompatible pose vector sizes for joint optimization");
    }

    // Parameter arrays
    std::vector<std::array<double,4>> intr(num_cams);
    for (size_t i = 0; i < num_cams; ++i) {
        intr[i] = {initial_cameras[i].K.fx,
                   initial_cameras[i].K.fy,
                   initial_cameras[i].K.cx,
                   initial_cameras[i].K.cy};
    }
    std::vector<Pose6> cam_poses(num_cams);
    std::vector<Pose6> targ_poses(num_views);
    initialize_pose_vectors(initial_camera_poses, initial_target_poses, cam_poses, targ_poses);

    ceres::Problem problem;
    setup_joint_problem(views, initial_cameras, intr, cam_poses, targ_poses, problem);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = verbose;
    constexpr double eps = 1e-6;
    opts.function_tolerance = eps;
    opts.gradient_tolerance = eps;
    opts.parameter_tolerance = eps;
    opts.max_num_iterations = 1000;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    result.summary = summary.BriefReport();

    // Extract solution
    extract_joint_solution(intr, cam_poses, targ_poses, result);

    // Residual statistics and distortion update
    const auto [ssr, count] = compute_joint_residual_stats(views, initial_cameras, result);
    if (count > 0) {
        result.reprojection_error = std::sqrt(ssr / count);
    }

    const int num_params = static_cast<int>(4*num_cams + ((num_cams?num_cams-1:0)+num_views)*6);
    const int dof = std::max(1, static_cast<int>(count) - num_params);
    const double sigma2 = ssr / dof;

    // Covariances
    compute_joint_covariance(problem, intr, cam_poses, targ_poses, sigma2, result);
    return result;
}

}  // namespace vitavision
