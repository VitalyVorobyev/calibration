#include "calibration/calib.h"

// std
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/distortion.h"
#include "calibration/planarpose.h"  // for estimate_planar_pose_dlt

namespace vitavision {

struct CalibObs { Eigen::Vector2d XY; Eigen::Vector2d uv; };
using Pose6 = Eigen::Matrix<double,6,1>;

// Variable projection residual for full camera calibration.
struct CalibVPResidual {
    std::vector<std::vector<CalibObs>> views_; // observations per view
    int num_radial_;
    size_t total_obs_;

    CalibVPResidual(std::vector<std::vector<CalibObs>> views, int num_radial)
        : views_(std::move(views)), num_radial_(num_radial) {
        total_obs_ = 0;
        for (const auto& v : views_) total_obs_ += v.size();
    }

    template<typename T>
    bool operator()(T const* const* params, T* residuals) const {
        const T* intr = params[0];
        std::vector<Observation<T>> obs;
        obs.reserve(total_obs_);
        for (size_t i = 0; i < views_.size(); ++i) {
            const T* pose6 = params[1 + i];
            for (const auto& ob : views_[i]) {
                Eigen::Matrix<T,3,1> P(T(ob.XY.x()), T(ob.XY.y()), T(0.0));
                Eigen::Matrix<T,3,1> Pc;
                ceres::AngleAxisRotatePoint(pose6, P.data(), Pc.data());
                Pc += Eigen::Matrix<T,3,1>(pose6[3], pose6[4], pose6[5]);
                T invZ = T(1.0) / Pc.z();
                Observation<T> o;
                o.x = Pc.x() * invZ;
                o.y = Pc.y() * invZ;
                o.u = T(ob.uv.x());
                o.v = T(ob.uv.y());
                obs.push_back(o);
            }
        }
        auto [_, r] = fit_distortion_full(obs, intr[0], intr[1], intr[2], intr[3], num_radial_);
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    DistortionWithResiduals<double> SolveFull(const double* intr, const std::vector<const double*>& pose_ptrs) const {
        std::vector<Observation<double>> obs;
        obs.reserve(total_obs_);
        for (size_t i = 0; i < views_.size(); ++i) {
            const double* pose6 = pose_ptrs[i];
            for (const auto& ob : views_[i]) {
                Eigen::Vector3d P(ob.XY.x(), ob.XY.y(), 0.0);
                Eigen::Vector3d Pc;
                ceres::AngleAxisRotatePoint(pose6, P.data(), Pc.data());
                Pc += Eigen::Vector3d(pose6[3], pose6[4], pose6[5]);
                double invZ = 1.0 / Pc.z();
                Observation<double> o;
                o.x = Pc.x() * invZ;
                o.y = Pc.y() * invZ;
                o.u = ob.uv.x();
                o.v = ob.uv.y();
                obs.push_back(o);
            }
        }
        return fit_distortion_full(obs, intr[0], intr[1], intr[2], intr[3], num_radial_);
    }
};

static Eigen::Affine3d axisangle_to_pose(const Pose6& pose6) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(pose6.head<3>().data(), R.data());
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = R;
    T.translation() = pose6.tail<3>();
    return T;
}

CameraCalibrationResult calibrate_camera_planar(
    const std::vector<PlanarView>& views,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verbose) {
    CameraCalibrationResult result;
    const size_t V = views.size();
    if (V == 0) return result;

    // Prepare observations per view
    std::vector<std::vector<CalibObs>> obs_views(V);
    size_t total_obs = 0;
    for (size_t i = 0; i < V; ++i) {
        const auto& obj = views[i].object_xy;
        const auto& img = views[i].image_uv;
        size_t n = std::min(obj.size(), img.size());
        obs_views[i].reserve(n);
        for (size_t j = 0; j < n; ++j) {
            obs_views[i].push_back({obj[j], img[j]});
        }
        total_obs += n;
    }

    double intr[4] = {initial_guess.fx, initial_guess.fy, initial_guess.cx, initial_guess.cy};
    std::vector<Pose6> poses(V);
    for (size_t i = 0; i < V; ++i) {
        Eigen::Affine3d pose = estimate_planar_pose_dlt(views[i].object_xy, views[i].image_uv, initial_guess);
        ceres::RotationMatrixToAngleAxis(pose.linear().data(), poses[i].data());
        poses[i][3] = pose.translation().x();
        poses[i][4] = pose.translation().y();
        poses[i][5] = pose.translation().z();
    }

    auto* functor = new CalibVPResidual(obs_views, num_radial);
    auto* cost = new ceres::DynamicAutoDiffCostFunction<CalibVPResidual>(functor);
    cost->AddParameterBlock(4);
    for (size_t i = 0; i < V; ++i) cost->AddParameterBlock(6);
    cost->SetNumResiduals(static_cast<int>(total_obs * 2));

    ceres::Problem problem;
    std::vector<double*> param_blocks;
    param_blocks.push_back(intr);
    for (size_t i = 0; i < V; ++i) param_blocks.push_back(poses[i].data());
    problem.AddResidualBlock(cost, nullptr, param_blocks);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = verbose;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    result.summary = summary.BriefReport();

    // Compute best distortion and residuals
    std::vector<const double*> pose_ptrs(V);
    for (size_t i = 0; i < V; ++i) pose_ptrs[i] = poses[i].data();
    auto dr = functor->SolveFull(intr, pose_ptrs);
    result.distortion = dr.distortion;

    // Reprojection errors
    int m = static_cast<int>(total_obs * 2);
    double ssr = dr.residuals.squaredNorm();
    result.reprojection_error = std::sqrt(ssr / m);
    result.view_errors.resize(V);
    int idx = 0;
    for (size_t i = 0; i < V; ++i) {
        int ni = static_cast<int>(obs_views[i].size());
        double s = 0.0;
        for (int j = 0; j < ni * 2; ++j) {
            double r = dr.residuals[idx++];
            s += r * r;
        }
        result.view_errors[i] = std::sqrt(s / (ni * 2));
    }

    // Populate outputs
    result.intrinsics.fx = intr[0];
    result.intrinsics.fy = intr[1];
    result.intrinsics.cx = intr[2];
    result.intrinsics.cy = intr[3];
    result.poses.resize(V);
    for (size_t i = 0; i < V; ++i) result.poses[i] = axisangle_to_pose(poses[i]);

    // Covariance computation
    const size_t total_params = 4 + 6 * V;
    std::vector<const double*> blocks = param_blocks;
    std::vector<int> block_sizes; block_sizes.push_back(4); for (size_t i=0;i<V;++i) block_sizes.push_back(6);
    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> cov_blocks;
    for (size_t i = 0; i < blocks.size(); ++i)
        for (size_t j = 0; j <= i; ++j)
            cov_blocks.emplace_back(blocks[i], blocks[j]);

    if (cov.Compute(cov_blocks, &problem)) {
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(total_params, total_params);
        size_t oi = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            int si = block_sizes[i];
            size_t oj = 0;
            for (size_t j = 0; j <= i; ++j) {
                int sj = block_sizes[j];
                std::vector<double> tmp(si * sj);
                cov.GetCovarianceBlock(blocks[i], blocks[j], tmp.data());
                for (int r = 0; r < si; ++r)
                    for (int c = 0; c < sj; ++c) {
                        C(oi + r, oj + c) = tmp[r * sj + c];
                        if (j < i) C(oj + c, oi + r) = tmp[r * sj + c];
                    }
                oj += sj;
            }
            oi += si;
        }
        int dof = std::max(1, m - static_cast<int>(total_params));
        double sigma2 = ssr / dof;
        C *= sigma2;
        result.covariance = C;
    }

    return result;
}

} // namespace vitavision

