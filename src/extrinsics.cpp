#include "calibration/extrinsics.h"

// std
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vitavision {

using Pose6 = Eigen::Matrix<double, 6, 1>;

struct ExtrinsicResidual {
    std::vector<PlanarObservation> obs_;
    Camera cam_;

    ExtrinsicResidual(std::vector<PlanarObservation> obs, const Camera& cam)
        : obs_(std::move(obs)), cam_(cam) {}

    template <typename T>
    bool operator()(const T* cam_pose6, const T* target_pose6, T* residuals) const {
        Eigen::Matrix<T,3,3> R_cam, R_target;
        ceres::AngleAxisToRotationMatrix(cam_pose6, R_cam.data());
        ceres::AngleAxisToRotationMatrix(target_pose6, R_target.data());
        Eigen::Matrix<T,3,1> t_cam{cam_pose6[3], cam_pose6[4], cam_pose6[5]};
        Eigen::Matrix<T,3,1> t_target{target_pose6[3], target_pose6[4], target_pose6[5]};

        Eigen::Matrix<T,3,3> R = R_cam * R_target;
        Eigen::Matrix<T,3,1> t = R_cam * t_target + t_cam;

        const int N = static_cast<int>(obs_.size());
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T,3,1> P{T(ob.object_xy.x()), T(ob.object_xy.y()), T(0)};
            Eigen::Matrix<T,3,1> Pc = R * P + t;
            T xn = Pc.x() / Pc.z();
            T yn = Pc.y() / Pc.z();
            Eigen::Matrix<T,2,1> xyn{xn, yn};
            auto uv = cam_.projectNormalized(xyn);
            residuals[2*i]   = uv.x() - T(ob.image_uv.x());
            residuals[2*i+1] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }
};

static Eigen::Affine3d pose6_to_affine(const Pose6& p) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(p.head<3>().data(), R.data());
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = R;
    T.translation() = p.tail<3>();
    return T;
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

static void setup_problem(const std::vector<ExtrinsicPlanarView>& views,
                          const std::vector<Camera>& cameras,
                          std::vector<Pose6>& cam_poses,
                          std::vector<Pose6>& targ_poses,
                          ceres::Problem& problem) {
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();

    for (size_t i = 0; i < num_cams; ++i) {
        problem.AddParameterBlock(cam_poses[i].data(), 6);
    }
    if (!cam_poses.empty()) {
        problem.SetParameterBlockConstant(cam_poses[0].data());
    }
    for (size_t j = 0; j < num_views; ++j) {
        problem.AddParameterBlock(targ_poses[j].data(), 6);
    }

    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        if (view.observations.size() != num_cams) continue;
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = view.observations[c];
            if (obs.empty()) continue;
            auto* functor = new ExtrinsicResidual(obs, cameras[c]);
            auto* cost = new ceres::AutoDiffCostFunction<ExtrinsicResidual,
                                                         ceres::DYNAMIC, 6, 6>(
                functor, static_cast<int>(obs.size()) * 2);
            problem.AddResidualBlock(cost, nullptr, cam_poses[c].data(), targ_poses[v].data());
        }
    }
}

static void extract_solution(const std::vector<Pose6>& cam_poses,
                             const std::vector<Pose6>& targ_poses,
                             ExtrinsicOptimizationResult& result) {
    result.camera_poses.resize(cam_poses.size());
    for (size_t i = 0; i < cam_poses.size(); ++i) {
        result.camera_poses[i] = pose6_to_affine(cam_poses[i]);
    }
    result.target_poses.resize(targ_poses.size());
    for (size_t j = 0; j < targ_poses.size(); ++j) {
        result.target_poses[j] = pose6_to_affine(targ_poses[j]);
    }
}

static std::pair<double, size_t> compute_residual_stats(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras,
    const ExtrinsicOptimizationResult& result) {
    double ssr = 0.0;
    size_t count = 0;
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        if (view.observations.size() != num_cams) continue;
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = view.observations[c];
            for (const auto& ob : obs) {
                Eigen::Vector3d P = result.camera_poses[c] * result.target_poses[v]
                                    * Eigen::Vector3d(ob.object_xy.x(), ob.object_xy.y(), 0.0);
                Eigen::Vector2d xyn{P.x()/P.z(), P.y()/P.z()};
                Eigen::Vector2d pred = cameras[c].projectNormalized(xyn);
                Eigen::Vector2d diff = pred - ob.image_uv;
                ssr += diff.squaredNorm();
                count += 2;
            }
        }
    }
    return {ssr, count};
}

static void compute_covariances(ceres::Problem& problem,
                                const std::vector<Pose6>& cam_poses,
                                const std::vector<Pose6>& targ_poses,
                                double sigma2,
                                ExtrinsicOptimizationResult& result) {
    const size_t num_cams = cam_poses.size();
    const size_t num_views = targ_poses.size();
    result.camera_covariances.assign(num_cams, Eigen::Matrix<double,6,6>::Zero());
    result.target_covariances.assign(num_views, Eigen::Matrix<double,6,6>::Zero());

    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> blocks;
    for (size_t i = 0; i < num_cams; ++i) {
        if (!problem.IsParameterBlockConstant(cam_poses[i].data())) {
            blocks.emplace_back(cam_poses[i].data(), cam_poses[i].data());
        }
    }
    for (size_t j = 0; j < num_views; ++j) {
        blocks.emplace_back(targ_poses[j].data(), targ_poses[j].data());
    }
    if (blocks.empty() || !cov.Compute(blocks, &problem)) {
        return; // covariances remain zero
    }
    for (size_t i = 0; i < num_cams; ++i) {
        if (problem.IsParameterBlockConstant(cam_poses[i].data())) continue;
        double Cov6x6[36];
        cov.GetCovarianceBlock(cam_poses[i].data(), cam_poses[i].data(), Cov6x6);
        Eigen::Map<Eigen::Matrix<double,6,6>> C(Cov6x6);
        result.camera_covariances[i] = sigma2 * C;
    }
    for (size_t j = 0; j < num_views; ++j) {
        double Cov6x6[36];
        cov.GetCovarianceBlock(targ_poses[j].data(), targ_poses[j].data(), Cov6x6);
        Eigen::Map<Eigen::Matrix<double,6,6>> C(Cov6x6);
        result.target_covariances[j] = sigma2 * C;
    }
}

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose) {
    ExtrinsicOptimizationResult result;
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    if (initial_camera_poses.size() != num_cams ||
        initial_target_poses.size() != num_views) {
        return result;
    }

    std::vector<Pose6> cam_poses(num_cams);
    std::vector<Pose6> targ_poses(num_views);
    initialize_pose_vectors(initial_camera_poses, initial_target_poses, cam_poses, targ_poses);

    ceres::Problem problem;
    setup_problem(views, cameras, cam_poses, targ_poses, problem);

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

    extract_solution(cam_poses, targ_poses, result);

    auto [ssr, count] = compute_residual_stats(views, cameras, result);
    if (count > 0) {
        result.reprojection_error = std::sqrt(ssr / count);
    }

    const int num_params = static_cast<int>((num_cams ? num_cams - 1 : 0) + num_views) * 6;
    const int dof = std::max(1, static_cast<int>(count) - num_params);
    const double sigma2 = ssr / dof;

    compute_covariances(problem, cam_poses, targ_poses, sigma2, result);

    return result;
}

} // namespace vitavision

