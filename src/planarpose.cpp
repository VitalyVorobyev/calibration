#include "calibration/planarpose.h"

// std
#include <algorithm>
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/homography.h"
#include "calibration/distortion.h"

#include "observationutils.h"

namespace vitavision {

// Decompose homography in normalized camera coords: H = [r1 r2 t]
Eigen::Affine3d pose_from_homography_normalized(const Eigen::Matrix3d& H) {
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
    if (R(2, 2) < 0) { // enforce cheirality (Z forward)
        R = -R; t = -t;
    }

    auto pose = Eigen::Affine3d::Identity();
    pose.linear() = R;
    pose.translation() = t;
    return pose;
}

// Convenience: one-shot planar pose from pixels & K
// Returns true on success; outputs R (world->cam) and t
Eigen::Affine3d estimate_planar_pose_dlt(const std::vector<Eigen::Vector2d>& obj_xy,
                                         const std::vector<Eigen::Vector2d>& img_uv,
                                         const CameraMatrix<double>& intrinsics) {
    if (obj_xy.size() < 4 || obj_xy.size() != img_uv.size()) {
        return Eigen::Affine3d::Identity();
    }

    std::vector<Eigen::Vector2d> img_norm(img_uv.size());
    std::transform(img_uv.begin(), img_uv.end(), img_norm.begin(),
        [&intrinsics](const Eigen::Vector2d& pix) {
            return intrinsics.normalize(pix);
        });

    Eigen::Matrix3d H = estimate_homography_dlt(obj_xy, img_norm);
    return pose_from_homography_normalized(H);
}

// Residual functor used with AutoDiffCostFunction for planar pose
// estimation.  For a given pose (angle-axis + translation) it builds the
// variable projection system to eliminate distortion coefficients.
struct PlanarPoseVPResidual {
    std::vector<PlanarObservation> obs_;
    double K_[4]; // fx, fy, cx, cy
    int num_radial_;

    PlanarPoseVPResidual(std::vector<PlanarObservation> obs,
                         int num_radial,
                         const CameraMatrix<double>& intrinsics)
        : obs_(std::move(obs)),
          K_{intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy},
          num_radial_(num_radial) {}

    template <typename T>
    bool operator()(const T* pose6, T* residuals) const {
        const T fx = T(K_[0]);
        const T fy = T(K_[1]);
        const T cx = T(K_[2]);
        const T cy = T(K_[3]);

        static thread_local std::vector<Observation<T>> o;
        if (o.size() != obs_.size()) o.resize(obs_.size());

        std::transform(obs_.begin(), obs_.end(), o.begin(),
            [pose6](const PlanarObservation& s) { return to_observation(s, pose6); });

        auto dr = fit_distortion_full(o, fx, fy, cx, cy, num_radial_);
        if (!dr) return false;
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    // Helper used after optimization to compute best distortion coefficients.
    Eigen::VectorXd SolveDistortionFor(const Pose6<double>& pose6) const {
        std::vector<Observation<double>> o(obs_.size());
        std::transform(obs_.begin(), obs_.end(), o.begin(),
            [pose6](const PlanarObservation& s) { return to_observation(s, pose6.data()); });

        auto d = fit_distortion(o, K_[0], K_[1], K_[2], K_[3], num_radial_);
        return d ? d->distortion : Eigen::VectorXd{};
    }
};

static Eigen::Affine3d axisangle_to_pose(const Pose6<double>& pose6) {
    Eigen::Matrix3d rotation_matrix;
    ceres::AngleAxisToRotationMatrix(pose6.head<3>().data(), rotation_matrix.data());

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.linear() = rotation_matrix;
    transform.translation() = pose6.tail<3>();

    return transform;
}

PlanarPoseFitResult optimize_planar_pose(
    const std::vector<Eigen::Vector2d>& obj_xy,
    const std::vector<Eigen::Vector2d>& img_uv,
    const CameraMatrix<double>& intrinsics,
    int num_radial,
    bool verbose
) {
    PlanarPoseFitResult result;

    // Step 1: Estimate initial pose using DLT
    auto init_pose = estimate_planar_pose_dlt(obj_xy, img_uv, intrinsics);

    // Step 2: Optimize pose using non-linear least squares
    Pose6<double> pose6;
    ceres::RotationMatrixToAngleAxis(reinterpret_cast<const double*>(init_pose.rotation().data()), pose6.data());
    pose6[3] = init_pose.translation().x();
    pose6[4] = init_pose.translation().y();
    pose6[5] = init_pose.translation().z();

    std::vector<PlanarObservation> view(obj_xy.size());
    std::transform(obj_xy.begin(), obj_xy.end(), img_uv.begin(), view.begin(),
        [](const Eigen::Vector2d& xy, const Eigen::Vector2d& uv) {
            return PlanarObservation{xy, uv};
        });

    ceres::Problem p;
    auto* functor = new PlanarPoseVPResidual(view, num_radial, intrinsics);
    auto* cost = new ceres::AutoDiffCostFunction<PlanarPoseVPResidual,
                                                 ceres::DYNAMIC, 6>(functor,
                                                                    static_cast<int>(view.size()) * 2);
    p.AddResidualBlock(cost, /*loss=*/nullptr, pose6.data());

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = verbose;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;

    ceres::Solver::Summary sum;
    ceres::Solve(opts, &p, &sum);
    result.summary = sum.BriefReport();

    // Best-fit distortion for the refined pose (if you want it)
    result.distortion = functor->SolveDistortionFor(pose6); // [k1..kK, p1, p2]

    // Residual stats & covariance (6x6 on pose)
    const int m = static_cast<int>(view.size()) * 2;
    std::vector<double> r(m);

    const double* parameter_blocks[] = {pose6.data()};
    cost->Evaluate(parameter_blocks, r.data(), nullptr);

    double ssr = 0.0;
    for (double e : r) ssr += e*e;
    const int dof = std::max(1, m - 6);
    const double sigma2 = ssr / dof;
    result.reprojection_error = std::sqrt(ssr / m);

    // Covariance block on pose
    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> blocks = { {pose6.data(), pose6.data()} };
    if (!cov.Compute(blocks, &p)) {
        std::cerr << "Covariance computation failed.\n";
        return result;
    }

    double Cov6x6[36];
    cov.GetCovarianceBlock(pose6.data(), pose6.data(), Cov6x6);

    // Scale by residual variance (unit weights)
    Eigen::Map<Eigen::Matrix<double, 6, 6>> Cpose(Cov6x6);
    Cpose *= sigma2;
    result.covariance = Cpose;
    result.pose = axisangle_to_pose(pose6);

    return result;
}

}  // namespace vitavision
