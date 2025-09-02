#include "calib/planarpose.h"

// std
#include <algorithm>
#include <array>
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calib/distortion.h"
#include "calib/homography.h"
#include "ceresutils.h"
#include "observationutils.h"

namespace calib {

// Decompose homography in normalized camera coords: H = [r1 r2 t]
Eigen::Isometry3d pose_from_homography_normalized(const Eigen::Matrix3d& H) {
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);

    double s = std::sqrt(h1.norm() * h2.norm());
    if (s < 1e-12) s = 1.0;
    Eigen::Vector3d r1 = h1 / s;
    Eigen::Vector3d r2 = h2 / s;
    Eigen::Vector3d r3 = r1.cross(r2);

    // Orthonormalize to the nearest rotation
    Eigen::Matrix3d Rinit;
    Rinit.col(0) = r1;
    Rinit.col(1) = r2;
    Rinit.col(2) = r3;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rinit, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = svd.matrixU() * V.transpose();
    }
    Eigen::Vector3d t = h3 / s;
    if (R(2, 2) < 0) {  // enforce cheirality (Z forward)
        R = -R;
        t = -t;
    }

    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = R;
    pose.translation() = t;
    return pose;
}

Eigen::Isometry3d estimate_planar_pose_dlt(const PlanarView& obs, const CameraMatrix& intrinsics) {
    if (obs.size() < 4) {
        return Eigen::Isometry3d::Identity();
    }

    std::vector<Eigen::Vector2d> obj_xy, img_uv;
    for (const auto& o : obs) {
        obj_xy.push_back(o.object_xy);
        img_uv.push_back(o.image_uv);
    }

    return estimate_planar_pose_dlt(obj_xy, img_uv, intrinsics);
}

// Convenience: one-shot planar pose from pixels & K
// Returns true on success; outputs R (world->cam) and t
auto estimate_planar_pose_dlt(const std::vector<Eigen::Vector2d>& obj_xy,
                              const std::vector<Eigen::Vector2d>& img_uv,
                              const CameraMatrix& intrinsics) -> Eigen::Isometry3d {
    if (obj_xy.size() < 4 || obj_xy.size() != img_uv.size()) {
        return Eigen::Isometry3d::Identity();
    }

    std::vector<Eigen::Vector2d> img_norm(img_uv.size());
    std::transform(img_uv.begin(), img_uv.end(), img_norm.begin(),
                   [&intrinsics](const Eigen::Vector2d& pix) { return intrinsics.normalize(pix); });

    Eigen::Matrix3d H = estimate_homography_dlt(obj_xy, img_norm);
    return pose_from_homography_normalized(H);
}

using Pose6 = Eigen::Matrix<double, 6, 1>;

struct PlanarPoseBlocks final : public ProblemParamBlocks {
    std::array<double, 6> pose6;
    std::vector<ParamBlock> get_param_blocks() const override {
        return {{pose6.data(), pose6.size(), 6}};
    }
};

// Residual functor used with AutoDiffCostFunction for planar pose
// estimation.  For a given pose (angle-axis + translation) it builds the
// variable projection system to eliminate distortion coefficients.
struct PlanarPoseVPResidual {
    PlanarView obs_;
    double K_[5];  // fx, fy, cx, cy, skew
    int num_radial_;

    PlanarPoseVPResidual(PlanarView obs, int num_radial, const CameraMatrix& intrinsics)
        : obs_(std::move(obs)),
          K_{intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.skew},
          num_radial_(num_radial) {}

    template <typename T>
    bool operator()(const T* pose6, T* residuals) const {
        const T fx = T(K_[0]);
        const T fy = T(K_[1]);
        const T cx = T(K_[2]);
        const T cy = T(K_[3]);
        const T skew_param = T(K_[4]);

        std::vector<Observation<T>> o(obs_.size());
        std::transform(obs_.begin(), obs_.end(), o.begin(),
                       [pose6](const PlanarObservation& s) { return to_observation(s, pose6); });

        auto dr = fit_distortion_full(o, fx, fy, cx, cy, skew_param, num_radial_);
        if (!dr) return false;
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    // Helper used after optimization to compute best distortion coefficients.
    Eigen::VectorXd SolveDistortionFor(const Pose6& pose6) const {
        std::vector<Observation<double>> o(obs_.size());
        std::transform(obs_.begin(), obs_.end(), o.begin(), [pose6](const PlanarObservation& s) {
            return to_observation(s, pose6.data());
        });

        auto d = fit_distortion(o, K_[0], K_[1], K_[2], K_[3], K_[4], num_radial_);
        return d ? d->distortion : Eigen::VectorXd{};
    }
};

static Eigen::Isometry3d axisangle_to_pose(const Pose6& pose6) {
    Eigen::Matrix3d rotation_matrix;
    ceres::AngleAxisToRotationMatrix(pose6.head<3>().data(), rotation_matrix.data());

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = rotation_matrix;
    transform.translation() = pose6.tail<3>();

    return transform;
}

PlanarPoseResult optimize_planar_pose(const std::vector<Eigen::Vector2d>& obj_xy,
                                      const std::vector<Eigen::Vector2d>& img_uv,
                                      const CameraMatrix& intrinsics,
                                      const PlanarPoseOptions& opts) {
    PlanarPoseResult result;

    auto init_pose = estimate_planar_pose_dlt(obj_xy, img_uv, intrinsics);
    PlanarPoseBlocks blocks;
    ceres::RotationMatrixToAngleAxis(reinterpret_cast<const double*>(init_pose.rotation().data()),
                                     blocks.pose6.data());
    blocks.pose6[3] = init_pose.translation().x();
    blocks.pose6[4] = init_pose.translation().y();
    blocks.pose6[5] = init_pose.translation().z();

    PlanarView view(obj_xy.size());
    std::transform(obj_xy.begin(), obj_xy.end(), img_uv.begin(), view.begin(),
                   [](const Eigen::Vector2d& xy, const Eigen::Vector2d& uv) {
                       return PlanarObservation{xy, uv};
                   });

    auto* functor = new PlanarPoseVPResidual(view, opts.num_radial, intrinsics);
    auto* cost = new ceres::AutoDiffCostFunction<PlanarPoseVPResidual, ceres::DYNAMIC, 6>(
        functor, static_cast<int>(view.size()) * 2);

    ceres::Problem problem;
    problem.AddResidualBlock(
        cost, opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr,
        blocks.pose6.data());

    solve_problem(problem, opts, &result);

    // Compute residuals for statistics and covariance
    const int m = static_cast<int>(view.size()) * 2;
    std::vector<double> residuals(m);
    const double* parameter_blocks[] = {blocks.pose6.data()};
    cost->Evaluate(parameter_blocks, residuals.data(), nullptr);

    const double ssr = std::accumulate(residuals.begin(), residuals.end(), 0.0,
                                       [](double sum, double r) { return sum + r * r; });
    result.reprojection_error = std::sqrt(ssr / m);

    if (opts.compute_covariance) {
        auto optcov = compute_covariance(blocks, problem, ssr, residuals.size());
        if (optcov.has_value()) {
            result.covariance = std::move(optcov.value());
        }
    }

    result.pose = axisangle_to_pose(Eigen::Map<const Pose6>(blocks.pose6.data()));
    result.distortion = functor->SolveDistortionFor(Eigen::Map<const Pose6>(blocks.pose6.data()));

    return result;
}

}  // namespace calib
