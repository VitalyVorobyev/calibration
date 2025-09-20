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
Eigen::Isometry3d pose_from_homography_normalized(const Eigen::Matrix3d& hmtx) {
    Eigen::Vector3d hcol1 = hmtx.col(0);
    Eigen::Vector3d hcol2 = hmtx.col(1);
    Eigen::Vector3d hcol3 = hmtx.col(2);

    double s = std::sqrt(hcol1.norm() * hcol2.norm());
    if (s < 1e-12) {
        s = 1.0;
    }
    Eigen::Vector3d rcol1 = hcol1 / s;
    Eigen::Vector3d rcol2 = hcol2 / s;
    Eigen::Vector3d rcol3 = rcol1.cross(rcol2);

    // Orthonormalize to the nearest rotation
    Eigen::Matrix3d r_init;
    r_init.col(0) = rcol1;
    r_init.col(1) = rcol2;
    r_init.col(2) = rcol3;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(r_init, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
    if (rotation.determinant() < 0) {
        Eigen::Matrix3d vmtx = svd.matrixV();
        vmtx.col(2) *= -1.0;
        rotation = svd.matrixU() * vmtx.transpose();
    }
    Eigen::Vector3d translation = hcol3 / s;
    if (rotation(2, 2) < 0) {  // enforce cheirality (Z forward)
        rotation = -rotation;
        translation = -translation;
    }

    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = rotation;
    pose.translation() = translation;
    return pose;
}

// TODO: return std::optional
auto estimate_planar_pose(PlanarView view, const CameraMatrix& intrinsics) -> Eigen::Isometry3d {
    if (view.size() < 4) {
        return Eigen::Isometry3d::Identity();
    }

    std::for_each(view.begin(), view.end(), [&intrinsics](PlanarObservation& item) {
        item.image_uv = intrinsics.normalize(item.image_uv);
    });

    const auto hres = estimate_homography(view);
    if (!hres.success) {
        std::cerr << "Failed to estimate homography\n";
        return Eigen::Isometry3d::Identity();
    }
    return pose_from_homography_normalized(hres.hmtx);
}

using Pose6 = Eigen::Matrix<double, 6, 1>;

struct PlanarPoseBlocks final : public ProblemParamBlocks {
    std::array<double, 6> pose6;
    [[nodiscard]] std::vector<ParamBlock> get_param_blocks() const override {
        return {{pose6.data(), pose6.size(), 6}};
    }
};

// Residual functor used with AutoDiffCostFunction for planar pose
// estimation.  For a given pose (angle-axis + translation) it builds the
// variable projection system to eliminate distortion coefficients.
struct PlanarPoseVPResidual final {
    PlanarView obs_;
    const CameraMatrix intrinsics_;
    int num_radial_;

    PlanarPoseVPResidual(PlanarView obs, int num_radial, CameraMatrix intrinsics)
        : obs_(std::move(obs)), intrinsics_(intrinsics), num_radial_(num_radial) {}

    template <typename T>
    bool operator()(const T* pose6, T* residuals) const {
        const CameraMatrixT<T> intrinsics{T(intrinsics_.fx), T(intrinsics_.fy), T(intrinsics_.cx),
                                          T(intrinsics_.cy), T(intrinsics_.skew)};

        std::vector<Observation<T>> o(obs_.size());
        std::transform(obs_.begin(), obs_.end(), o.begin(),
                       [pose6](const PlanarObservation& s) { return to_observation(s, pose6); });

        auto dr = fit_distortion_full(o, intrinsics, num_radial_);
        if (!dr) {
            return false;
        }
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) {
            residuals[i] = r[i];
        }
        return true;
    }

    // Helper used after optimization to compute best distortion coefficients.
    [[nodiscard]] Eigen::VectorXd solve_distortion_for(const Pose6& pose6) const {
        std::vector<Observation<double>> o(obs_.size());
        std::transform(obs_.begin(), obs_.end(), o.begin(), [pose6](const PlanarObservation& s) {
            return to_observation(s, pose6.data());
        });

        const CameraMatrix intrinsics{intrinsics_.fx, intrinsics_.fy, intrinsics_.cx,
                                      intrinsics_.cy, intrinsics_.skew};
        auto d = fit_distortion(o, intrinsics, num_radial_);
        return d ? d->distortion : Eigen::VectorXd{};
    }
};

static auto axisangle_to_pose(const Pose6& pose6) -> Eigen::Isometry3d {
    Eigen::Matrix3d rotation_matrix;
    ceres::AngleAxisToRotationMatrix(pose6.head<3>().data(), rotation_matrix.data());

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = rotation_matrix;
    transform.translation() = pose6.tail<3>();

    return transform;
}

auto optimize_planar_pose(const PlanarView& view, const CameraMatrix& intrinsics,
                          const Eigen::Isometry3d& init_pose, const PlanarPoseOptions& opts)
    -> PlanarPoseResult {
    PlanarPoseResult result;
    PlanarPoseBlocks blocks;
    ceres::RotationMatrixToAngleAxis(reinterpret_cast<const double*>(init_pose.rotation().data()),
                                     blocks.pose6.data());
    blocks.pose6[3] = init_pose.translation().x();
    blocks.pose6[4] = init_pose.translation().y();
    blocks.pose6[5] = init_pose.translation().z();

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
    const std::array<const double*, 1> parameter_blocks = {blocks.pose6.data()};
    cost->Evaluate(parameter_blocks.data(), residuals.data(), nullptr);

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
    result.distortion = functor->solve_distortion_for(Eigen::Map<const Pose6>(blocks.pose6.data()));

    return result;
}

}  // namespace calib
