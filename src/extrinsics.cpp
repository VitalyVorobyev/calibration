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
    const CameraMatrix kmtx_;
    Eigen::VectorXd dist_;

    ExtrinsicResidual(std::vector<PlanarObservation> obs,
                      const CameraMatrix& intr,
                      const Eigen::VectorXd& dist)
        : obs_(std::move(obs)),
          kmtx_(intr),
          dist_(dist) {}

    template<typename T>
    bool operator()(const T* cam_pose6, const T* target_pose6, T* residuals) const {
        Eigen::Matrix<T, 3, 3> R_cam, R_target;
        ceres::AngleAxisToRotationMatrix(cam_pose6, R_cam.data());
        ceres::AngleAxisToRotationMatrix(target_pose6, R_target.data());
        Eigen::Matrix<T, 3, 1> t_cam{cam_pose6[3], cam_pose6[4], cam_pose6[5]};
        Eigen::Matrix<T, 3, 1> t_target{target_pose6[3], target_pose6[4], target_pose6[5]};

        Eigen::Matrix<T, 3, 3> R = R_cam * R_target;
        Eigen::Matrix<T, 3, 1> t = R_cam * t_target + t_cam;

        const int N = static_cast<int>(obs_.size());
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T, 3, 1> P{T(ob.object_xy.x()), T(ob.object_xy.y()), T(0)};
            Eigen::Matrix<T, 3, 1> Pc = R * P + t;
            T xn = Pc.x() / Pc.z();
            T yn = Pc.y() / Pc.z();
            Eigen::Matrix<T, 2, 1> xyn {xn, yn};
            Eigen::Matrix<T, 2, 1> d = apply_distortion(xyn, dist_);
            auto uv = kmtx_.denormalize(d);
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

static Eigen::Vector2d project_point(const Eigen::Vector2d& obj,
                                     const Eigen::Affine3d& cam_pose,
                                     const Eigen::Affine3d& target_pose,
                                     const CameraMatrix& K,
                                     const Eigen::VectorXd& dist) {
    Eigen::Vector3d P = cam_pose * target_pose * Eigen::Vector3d(obj.x(), obj.y(), 0.0);
    Eigen::Vector2d xyn { P.x() / P.z(), P.y() / P.z() };
    xyn = apply_distortion(xyn, dist);
    return K.denormalize(xyn);
}

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<CameraMatrix>& intrinsics,
    const std::vector<Eigen::VectorXd>& distortions,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose
) {
    ExtrinsicOptimizationResult result;
    const size_t num_cams = intrinsics.size();
    const size_t num_views = views.size();
    if (distortions.size() != num_cams ||
        initial_camera_poses.size() != num_cams ||
        initial_target_poses.size() != num_views) {
        return result;
    }

    std::vector<Pose6> cam_poses(num_cams);
    std::vector<Pose6> targ_poses(num_views);

    for (size_t i = 0; i < num_cams; ++i) {
        ceres::RotationMatrixToAngleAxis(initial_camera_poses[i].rotation().data(), cam_poses[i].data());
        cam_poses[i][3] = initial_camera_poses[i].translation().x();
        cam_poses[i][4] = initial_camera_poses[i].translation().y();
        cam_poses[i][5] = initial_camera_poses[i].translation().z();
    }

    for (size_t j = 0; j < num_views; ++j) {
        ceres::RotationMatrixToAngleAxis(initial_target_poses[j].rotation().data(), targ_poses[j].data());
        targ_poses[j][3] = initial_target_poses[j].translation().x();
        targ_poses[j][4] = initial_target_poses[j].translation().y();
        targ_poses[j][5] = initial_target_poses[j].translation().z();
    }

    ceres::Problem problem;
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
            auto* functor = new ExtrinsicResidual(obs, intrinsics[c], distortions[c]);
            auto* cost = new ceres::AutoDiffCostFunction<ExtrinsicResidual,
                                                         ceres::DYNAMIC, 6, 6>(
                functor, static_cast<int>(obs.size()) * 2);
            problem.AddResidualBlock(cost, nullptr, cam_poses[c].data(), targ_poses[v].data());
        }
    }

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

    result.camera_poses.resize(num_cams);
    for (size_t i = 0; i < num_cams; ++i) {
        result.camera_poses[i] = pose6_to_affine(cam_poses[i]);
    }
    result.target_poses.resize(num_views);
    for (size_t j = 0; j < num_views; ++j) {
        result.target_poses[j] = pose6_to_affine(targ_poses[j]);
    }

    double ssr = 0.0;
    size_t count = 0;
    for (size_t v = 0; v < num_views; ++v) {
        const auto& view = views[v];
        if (view.observations.size() != num_cams) continue;
        for (size_t c = 0; c < num_cams; ++c) {
            const auto& obs = view.observations[c];
            for (const auto& ob : obs) {
                Eigen::Vector2d pred = project_point(ob.object_xy,
                                                     result.camera_poses[c],
                                                     result.target_poses[v],
                                                     intrinsics[c],
                                                     distortions[c]);
                Eigen::Vector2d diff = pred - ob.image_uv;
                ssr += diff.squaredNorm();
                count += 2;
            }
        }
    }
    if (count > 0) {
        result.reprojection_error = std::sqrt(ssr / count);
    }

    return result;
}

} // namespace vitavision
