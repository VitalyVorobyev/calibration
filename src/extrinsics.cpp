#include "calibration/extrinsics.h"

// std
#include <numeric>
#include <algorithm>
#include <array>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "observationutils.h"

namespace vitavision {

template<typename T>
using Pose6 = Eigen::Matrix<T, 6, 1>;

InitialExtrinsicGuess make_initial_extrinsic_guess(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<double>>& cameras
) {
    const size_t num_cams = cameras.size();
    const size_t num_views = views.size();
    std::vector<std::vector<Eigen::Affine3d>> cam_to_target(
        num_views, std::vector<Eigen::Affine3d>(num_cams, Eigen::Affine3d::Identity()));

    // Estimate per-camera target poses via DLT
    for (size_t v = 0; v < num_views; ++v) {
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = views[v].observations;
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
            cam_to_target[v][c] = estimate_planar_pose_dlt(obj_xy, img_uv, cameras[c].intrinsics);
        }
    }

    InitialExtrinsicGuess guess;
    guess.camera_poses.assign(num_cams, Eigen::Affine3d::Identity());
    guess.target_poses.assign(num_views, Eigen::Affine3d::Identity());

    // Compute camera poses relative to first camera (reference)
    for (size_t c = 1; c < num_cams; ++c) {
        std::vector<Eigen::Affine3d> rels;
        for (size_t v = 0; v < num_views; ++v) {
            if (c >= views[v].observations.size()) continue;
            const auto& obs0 = views[v].observations[0];
            const auto& obsC = views[v].observations[c];
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
            if (c >= views[v].observations.size()) continue;
            const auto& ob_c = views[v].observations[c];
            if (ob_c.size() < 4) continue;
            tposes.push_back(guess.camera_poses[c].inverse() * cam_to_target[v][c]);
        }
        if (!tposes.empty()) {
            guess.target_poses[v] = average_affines(tposes);
        }
    }

    return guess;
}

struct ExtrinsicResidual {
    std::vector<PlanarObservation> obs_;
    Camera<double> cam_;
    std::vector<double> dist_;

    ExtrinsicResidual(std::vector<PlanarObservation> obs, const Camera<double>& cam)
        : obs_(std::move(obs)), cam_(cam), dist_(cam_.distortion.size()) {
            for (int i = 0; i < cam_.distortion.size(); ++i) {
                dist_[static_cast<size_t>(i)] = cam_.distortion[i];
            }
        }

    template <typename T>
    bool operator()(const T* cam_pose6, const T* target_pose6, T* residuals) const {
        Eigen::Transform<T, 3, Eigen::Affine> T_cam, T_target;
        ceres::AngleAxisToRotationMatrix(cam_pose6, T_cam.linear().data());
        ceres::AngleAxisToRotationMatrix(target_pose6, T_target.linear().data());
        T_cam.translation() << cam_pose6[3], cam_pose6[4], cam_pose6[5];
        T_target.translation() << target_pose6[3], target_pose6[4], target_pose6[5];

        Eigen::Matrix<T,Eigen::Dynamic,1> dist(dist_.size());
        for (size_t i = 0; i < dist_.size(); ++i) {
            dist(static_cast<int>(i)) = T(dist_[i]);
        }

        Camera<T> cam {
            CameraMatrix<T> {
                T(cam_.intrinsics.fx),
                T(cam_.intrinsics.fy),
                T(cam_.intrinsics.cx),
                T(cam_.intrinsics.cy)
            },
            dist,
            T_cam
        };

        const int N = static_cast<int>(obs_.size());
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T,3,1> P{T(ob.object_xy.x()), T(ob.object_xy.y()), T(0)};
            P = T_target * P;
            auto uv = cam.project(P);
            residuals[2*i]   = uv.x() - T(ob.image_uv.x());
            residuals[2*i+1] = uv.y() - T(ob.image_uv.y());
        }
        return true;
    }
};

struct JointResidual {
    std::vector<PlanarObservation> obs_;
    std::vector<double> dist_;
    int num_radial_;

    JointResidual(std::vector<PlanarObservation> obs, const Eigen::VectorXd& dist)
        : obs_(std::move(obs)),
          dist_(static_cast<size_t>(dist.size())),
          num_radial_(std::max<int>(0, static_cast<int>(dist.size()) - 2)) {
            for (int i = 0; i < dist.size(); ++i) {
                dist_[static_cast<size_t>(i)] = dist(i);
            }
          }

    template <typename T>
    bool operator()(const T* intr, const T* cam_pose6, const T* target_pose6, T* residuals) const {
        Eigen::Transform<T, 3, Eigen::Affine> T_cam, T_target;
        ceres::AngleAxisToRotationMatrix(cam_pose6, T_cam.linear().data());
        ceres::AngleAxisToRotationMatrix(target_pose6, T_target.linear().data());
        T_cam.translation() << cam_pose6[3], cam_pose6[4], cam_pose6[5];
        T_target.translation() << target_pose6[3], target_pose6[4], target_pose6[5];

        Eigen::Matrix<T,Eigen::Dynamic,1> dist(dist_.size());
        for (size_t i = 0; i < dist_.size(); ++i) {
            dist(static_cast<int>(i)) = T(dist_[i]);
        }

        Camera<T> cam {
            CameraMatrix<T> {intr[0], intr[1], intr[2], intr[3]},
            dist,
            T_cam
        };

        const int N = static_cast<int>(obs_.size());
        #if 1
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T,3,1> P{T(ob.object_xy.x()), T(ob.object_xy.y()), T(0)};
            P = T_target * P;
            auto uv = cam.project(P);
            residuals[2*i]   = uv.x() - T(ob.image_uv.x());
            residuals[2*i+1] = uv.y() - T(ob.image_uv.y());
        }
        #else
        static thread_local std::vector<Observation<T>> o;
        if (o.size() != static_cast<size_t>(N)) o.resize(N);
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T,3,1> P{T(ob.object_xy.x()), T(ob.object_xy.y()), T(0)};
            Eigen::Matrix<T,3,1> Pc = R * P + t;
            T xn = Pc.x() / Pc.z();
            T yn = Pc.y() / Pc.z();
            o[i] = Observation<T>{xn, yn, T(ob.image_uv.x()), T(ob.image_uv.y())};
        }

        auto dr = fit_distortion_full(o, intr[0], intr[1], intr[2], intr[3], num_radial_);
        if (!dr) {
            throw std::runtime_error("Failed to fit distortion");
        } else {
            const auto& r = dr->residuals;
            for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        }
        #endif
        return true;
    }
};

template<typename T>
static Eigen::Transform<T, 3, Eigen::Affine> pose6_to_affine(const Pose6<T>& p) {
    Eigen::Transform<T, 3, Eigen::Affine> pose = Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    ceres::AngleAxisToRotationMatrix(p.template head<3>().data(), pose.linear().data());
    pose.translation() = p.template tail<3>();
    return pose;
}

static void initialize_pose_vectors(const std::vector<Eigen::Affine3d>& initial_camera_poses,
                                    const std::vector<Eigen::Affine3d>& initial_target_poses,
                                    std::vector<Pose6<double>>& cam_poses,
                                    std::vector<Pose6<double>>& targ_poses) {
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

static void initialize_joint_vectors(const std::vector<Camera<double>>& initial_cameras,
                                     const std::vector<Eigen::Affine3d>& initial_camera_poses,
                                     const std::vector<Eigen::Affine3d>& initial_target_poses,
                                     std::vector<std::array<double,4>>& intr,
                                     std::vector<Pose6<double>>& cam_poses,
                                     std::vector<Pose6<double>>& targ_poses) {
    for (size_t i = 0; i < initial_cameras.size(); ++i) {
        intr[i] = {initial_cameras[i].intrinsics.fx,
                   initial_cameras[i].intrinsics.fy,
                   initial_cameras[i].intrinsics.cx,
                   initial_cameras[i].intrinsics.cy};
    }
    initialize_pose_vectors(initial_camera_poses, initial_target_poses, cam_poses, targ_poses);
}

static void setup_problem(const std::vector<ExtrinsicPlanarView>& views,
                          const std::vector<Camera<double>>& cameras,
                          std::vector<Pose6<double>>& cam_poses,
                          std::vector<Pose6<double>>& targ_poses,
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

static void extract_solution(const std::vector<Pose6<double>>& cam_poses,
                             const std::vector<Pose6<double>>& targ_poses,
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
    const std::vector<Camera<double>>& cameras,
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
                Eigen::Vector2d pred = cameras[c].project_normalized(xyn);
                Eigen::Vector2d diff = pred - ob.image_uv;
                ssr += diff.squaredNorm();
                count += 2;
            }
        }
    }
    return {ssr, count};
}

static void compute_covariances(ceres::Problem& problem,
                                const std::vector<Pose6<double>>& cam_poses,
                                const std::vector<Pose6<double>>& targ_poses,
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
    const std::vector<Camera<double>>& cameras,
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

    std::vector<Pose6<double>> cam_poses(num_cams);
    std::vector<Pose6<double>> targ_poses(num_views);
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

static void setup_joint_problem(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<double>>& initial_cameras,
    std::vector<std::array<double, 4>>& intr,
    std::vector<Pose6<double>>& cam_poses,
    std::vector<Pose6<double>>& targ_poses,
    ceres::Problem& problem
) {
    const size_t num_cams = initial_cameras.size();
    const size_t num_views = views.size();

    for (size_t i = 0; i < num_cams; ++i) {
        problem.AddParameterBlock(intr[i].data(), 4);
        problem.AddParameterBlock(cam_poses[i].data(), 6);
    }
    if (!cam_poses.empty()) {
        problem.SetParameterBlockConstant(cam_poses[0].data());
    }
    for (size_t v = 0; v < num_views; ++v) {
        problem.AddParameterBlock(targ_poses[v].data(), 6);
    }

    // Residuals
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        for (size_t c = 0; c < num_cams && c < view.observations.size(); ++c) {
            const auto& obs = view.observations[c];
            if (obs.empty()) continue;
            auto* functor = new JointResidual(obs, initial_cameras[c].distortion);
            auto* cost = new ceres::AutoDiffCostFunction<JointResidual,
                                                         ceres::DYNAMIC,
                                                         4,6,6>(functor,
                                                                static_cast<int>(obs.size())*2);
            problem.AddResidualBlock(cost, nullptr,
                                     intr[c].data(), cam_poses[c].data(), targ_poses[v].data());
        }
    }
}

static void extract_joint_solution(
    const std::vector<std::array<double, 4>>& intr,
    const std::vector<Pose6<double>>& cam_poses,
    const std::vector<Pose6<double>>& targ_poses,
    JointOptimizationResult& result
) {
    const size_t num_cams = cam_poses.size();
    const size_t num_views = targ_poses.size();

    result.intrinsics.resize(num_cams);
    result.camera_poses.resize(num_cams);
    for (size_t i = 0; i < num_cams; ++i) {
        result.intrinsics[i] = {intr[i][0], intr[i][1], intr[i][2], intr[i][3]};
        result.camera_poses[i] = pose6_to_affine(cam_poses[i]);
    }
    result.target_poses.resize(num_views);
    for (size_t v = 0; v < num_views; ++v) {
        result.target_poses[v] = pose6_to_affine(targ_poses[v]);
    }
}

static std::pair<double, size_t> compute_joint_residual_stats(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<double>>& cameras,
    JointOptimizationResult& result
) {
    const size_t num_views = views.size();
    const size_t num_cams = cameras.size();

    std::vector<std::vector<Observation<double>>> per_cam_obs(num_cams);
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        for (size_t c = 0; c < num_cams && c < view.observations.size(); ++c) {
            const auto& obs = view.observations[c];
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
        int num_radial = std::max<int>(0, static_cast<int>(cameras[c].distortion.size()) - 2);
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
    const std::vector<Pose6<double>>& cam_poses,
    const std::vector<Pose6<double>>& targ_poses,
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

// Joint optimization of camera intrinsics, extrinsic poses and target poses
JointOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera<double>>& initial_cameras,
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
    std::vector<Pose6<double>> cam_poses(num_cams);
    std::vector<Pose6<double>> targ_poses(num_views);
    initialize_joint_vectors(
        initial_cameras, initial_camera_poses, initial_target_poses, intr, cam_poses, targ_poses);

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

} // namespace vitavision
