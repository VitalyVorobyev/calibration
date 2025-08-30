#include "calib/extrinsics.h"

// std
#include <numeric>
#include <algorithm>
#include <array>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "observationutils.h"

namespace calib {

using Pose6 = Eigen::Matrix<double, 6, 1>;

InitialExtrinsicGuess make_initial_extrinsic_guess(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<DualDistortion>>& cameras
) {
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    std::vector<std::vector<Eigen::Affine3d>> cam_to_target(
        num_views, std::vector<Eigen::Affine3d>(num_cams, Eigen::Affine3d::Identity()));

    // Estimate per-camera target poses via DLT
    for (size_t v = 0; v < num_views; ++v) {
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = views[v];
            if (c >= obs.size()) continue;
            const auto& ob_c = obs[c];
            if (ob_c.size() < 4) continue;
            std::vector<Eigen::Vector2d> obj_xy, img_uv;
            obj_xy.reserve(ob_c.size());
            img_uv.reserve(ob_c.size());
            for (const auto& o : ob_c) {
                obj_xy.push_back(o.object_xy);
                img_uv.push_back(o.image_uv);
            }
            cam_to_target[v][c] = estimate_planar_pose_dlt(obj_xy, img_uv, cameras[c].K);
        }
    }

    InitialExtrinsicGuess guess;
    guess.camera_poses.assign(num_cams, Eigen::Affine3d::Identity());
    guess.target_poses.assign(num_views, Eigen::Affine3d::Identity());

    // Compute camera poses relative to first camera (reference)
    for (size_t c = 1; c < num_cams; ++c) {
        std::vector<Eigen::Affine3d> rels;
        for (size_t v = 0; v < num_views; ++v) {
            if (c >= views[v].size()) continue;
            const auto& obs0 = views[v][0];
            const auto& obsC = views[v][c];
            if (obs0.size() < 4 || obsC.size() < 4) continue;
            rels.push_back(cam_to_target[v][c] * cam_to_target[v][0].inverse());
        }
        if (!rels.empty()) {
            guess.camera_poses[c] = average_affines(rels);
        }
    }

    // Compute target poses in reference frame
    for (size_t v = 0; v < num_views; ++v) {
        std::vector<Eigen::Affine3d> tposes;
        for (size_t c = 0; c < num_cams; ++c) {
            if (c >= views[v].size()) continue;
            const auto& ob_c = views[v][c];
            if (ob_c.size() < 4) continue;
            tposes.push_back(guess.camera_poses[c].inverse() * cam_to_target[v][c]);
        }
        if (!tposes.empty()) {
            guess.target_poses[v] = average_affines(tposes);
        }
    }

    return guess;
}

struct ExtrinsicResidual final {
    PlanarView obs_;
    const Camera<DualDistortion> cam_;

    ExtrinsicResidual(PlanarView obs, const Camera<DualDistortion>& cam)
        : obs_(std::move(obs)), cam_(cam) {}

    template <typename T>
    bool operator()(const T* cam_pose6, const T* target_pose6, T* residuals) const {
        auto pose_cam = pose2affine(cam_pose6);
        auto pose_target = pose2affine(target_pose6);
        Eigen::Transform<T, 3, Eigen::Affine> pose = pose_cam * pose_target;

        const int N = static_cast<int>(obs_.size());
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T,3,1> P{T(ob.object_xy.x()), T(ob.object_xy.y()), T(0)};
            P = pose * P;
            T xn = P.x() / P.z();
            T yn = P.y() / P.z();
            Eigen::Matrix<T,2,1> xyn{xn, yn};
            auto uv = cam_.project(xyn);
            residuals[2*i]   = uv.x() - T(ob.image_uv.x());
            residuals[2*i+1] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }

    static auto* create(const PlanarView& obs, const Camera<DualDistortion> cam) {
        auto* functor = new ExtrinsicResidual(obs, cam);
        auto* cost = new ceres::AutoDiffCostFunction<ExtrinsicResidual,
            ceres::DYNAMIC, 6, 6>(
        functor, static_cast<int>(obs.size()) * 2);

        return cost;
    }
};

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
                          const std::vector<Camera<DualDistortion>>& cameras,
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
        if (view.size() != num_cams) continue;
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = view[c];
            if (obs.empty()) continue;
            auto* cost = ExtrinsicResidual::create(obs, cameras[c]);
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
    const std::vector<Camera<DualDistortion>>& cameras,
    const ExtrinsicOptimizationResult& result) {
    double ssr = 0.0;
    size_t count = 0;
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        if (view.size() != num_cams) continue;
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = view[c];
            for (const auto& ob : obs) {
                Eigen::Vector3d P = result.camera_poses[c] * result.target_poses[v]
                                    * Eigen::Vector3d(ob.object_xy.x(), ob.object_xy.y(), 0.0);
                Eigen::Vector2d xyn{P.x()/P.z(), P.y()/P.z()};
                Eigen::Vector2d pred = cameras[c].project(xyn);
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
    const std::vector<Camera<DualDistortion>>& cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose
) {
    ExtrinsicOptimizationResult result;
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    if (initial_camera_poses.size() != num_cams ||
        initial_target_poses.size() != num_views) {
        throw std::invalid_argument("Incompatible pose vector sizes: "
                                    "cameras: " + std::to_string(num_cams) +
                                    ", views: " + std::to_string(num_views) +
                                    ", initial_camera_poses: " + std::to_string(initial_camera_poses.size()) +
                                    ", initial_target_poses: " + std::to_string(initial_target_poses.size()) +
                                    " are not compatible.");
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

} // namespace calib
