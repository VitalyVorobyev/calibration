#include "calib/linescan.h"

// std
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ceres
#include <ceres/ceres.h>

#include "calib/distortion.h"
#include "calib/homography.h"
#include "calib/planarpose.h"

namespace calib {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// Simple SVD-based plane initialization
static Eigen::Vector4d fit_plane_svd(const std::vector<Vec3>& pts) {
    Vec3 centroid =
        std::accumulate(pts.cbegin(), pts.cend(), Vec3{Vec3::Zero()},
                        [](const Vec3& avec, const Vec3& bvec) -> Vec3 { return avec + bvec; });
    centroid /= static_cast<double>(pts.size());

    Eigen::MatrixXd amtx(static_cast<Eigen::Index>(pts.size()), 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        amtx.row(static_cast<Eigen::Index>(i)) = (pts[i] - centroid).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(amtx, Eigen::ComputeFullV);
    Vec3 normal = svd.matrixV().col(2);
    double d = -normal.dot(centroid);
    return {normal.x(), normal.y(), normal.z(), d};
}

struct PlaneResidual final {
    explicit PlaneResidual(Vec3 p) : p_(std::move(p)) {}

    template <typename T>
    bool operator()(const T* plane, T* residual) const {
        // plane = [nx, ny, nz, d]
        T nx = plane[0];
        T ny = plane[1];
        T nz = plane[2];
        T d = plane[3];
        T denom = ceres::sqrt(nx * nx + ny * ny + nz * nz);
        residual[0] = (nx * T(p_.x()) + ny * T(p_.y()) + nz * T(p_.z()) + d) / denom;
        return true;
    }

    Vec3 p_;

    static auto create(const Vec3& p) {
        auto* cost = new ceres::AutoDiffCostFunction<PlaneResidual, 1, 4>(new PlaneResidual(p));
        return cost;
    }
};

// Validates that the observations meet minimum requirements
static void validate_observations(const std::vector<LineScanObservation>& views) {
    if (views.size() < 2) {
        throw std::invalid_argument("At least 2 views are required");
    }

    if (std::any_of(views.begin(), views.end(), [](const auto& v) {
            return v.target_xy.size() < 4 || v.target_xy.size() != v.target_uv.size();
        })) {
        throw std::invalid_argument("Each view requires >=4 target correspondences");
    }
}

// Processes a single view to extract 3D points
static auto process_view(const LineScanObservation& view,
                                      const Camera<DualDistortion>& camera) -> std::vector<Vec3> {
    std::vector<Vec3> points;
    PlanarView pview(view.target_xy.size());

    // Normalize and undistort target pixel coordinates
    std::transform(
        view.target_xy.begin(), view.target_xy.end(), view.target_uv.begin(), pview.begin(),
        [&camera](const Vec2& xy, const Vec2& uv) {
            return PlanarObservation{xy, camera.unproject(uv)};
        });

    // Homography from normalized pixels to plane
    // TODO: consider homography optimization
    auto h_norm_to_obj = estimate_homography(pview);
    if (!h_norm_to_obj.success) {
        std::cout << "Failed to estimate homography for view\n";
        return points;
    }

    // Pose of plane (world->camera)
    Eigen::Isometry3d pose = estimate_planar_pose_dlt(pview, camera.kmtx);

    // Reproject laser pixels to plane and transform to camera coordinates
    for (const auto& lpix : view.laser_uv) {
        Vec2 norm = camera.unproject(lpix);
        Eigen::Vector3d hp = h_norm_to_obj.hmtx * Eigen::Vector3d(norm.x(), norm.y(), 1.0);
        Vec2 plane_xy = hp.hnormalized();
        Vec3 obj_pt(plane_xy.x(), plane_xy.y(), 0.0);
        Vec3 cam_pt = pose * obj_pt;
        points.push_back(cam_pt);
    }

    return points;
}

// Fits a plane to 3D points using Ceres optimization
static Eigen::Vector4d fit_plane(const std::vector<Vec3>& points, std::string& summary) {
    if (points.size() < 3) {
        throw std::invalid_argument("Not enough laser points to fit a plane");
    }

    Eigen::Vector4d init = fit_plane_svd(points);
    std::array<double, 4> params = {init[0], init[1], init[2], init[3]};

    ceres::Problem problem;
    for (const auto& p : points) {
        auto* cost = PlaneResidual::create(p);
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) - ownership transferred to
        // ceres::Problem
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), params.data());
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary solver_summary;
    ceres::Solve(opts, &problem, &solver_summary);
    summary = solver_summary.BriefReport();

    // Normalize the plane parameters
    double nrm = std::sqrt(params[0] * params[0] + params[1] * params[1] + params[2] * params[2]);
    return {params[0] / nrm, params[1] / nrm, params[2] / nrm, params[3] / nrm};
}

// Computes statistics for the fitted plane
static void compute_plane_statistics(const std::vector<Vec3>& points, const Eigen::Vector4d& plane,
                                     LineScanCalibrationResult& result) {
    // Denormalized plane parameters for covariance computation
    double nrm = plane.head<3>().norm();
    std::array<double, 4> params = {plane[0] * nrm, plane[1] * nrm, plane[2] * nrm, plane[3] * nrm};

    // Compute residual stats
    double ssr = 0.0;
    for (const auto& p : points) {
        double r = (plane[0] * p.x() + plane[1] * p.y() + plane[2] * p.z() + plane[3]);
        ssr += r * r;
    }
    int m = static_cast<int>(points.size());
    int dof = std::max(1, m - 3);
    double sigma2 = ssr / dof;
    result.rms_error = std::sqrt(ssr / m);

    // Create problem for covariance computation
    ceres::Problem problem;
    for (const auto& p : points) {
        auto* cost = PlaneResidual::create(p);
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) - ownership transferred to
        // ceres::Problem
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), params.data());
    }

    // Covariance of plane params
    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> blocks = {{params.data(), params.data()}};
    if (cov.Compute(blocks, &problem)) {
        std::array<double, 16> cov4{};
        cov.GetCovarianceBlock(params.data(), params.data(), cov4.data());
        Eigen::Map<Eigen::Matrix<double, 4, 4>> c(cov4.data());
        c *= sigma2;
        result.covariance = c;
    } else {
        result.covariance.setZero();
    }
}

// Builds homography from normalized pixels to plane coordinates
static Mat3 build_plane_homography(const Eigen::Vector4d& plane) {
    Vec3 nvec = plane.head<3>();
    Vec3 p0 = -plane[3] * nvec;  // closest point to camera
    Vec3 tmp = (std::abs(nvec.z()) < 0.9) ? Vec3::UnitZ() : Vec3::UnitX();
    Vec3 e1 = nvec.cross(tmp).normalized();
    Vec3 e2 = nvec.cross(e1).normalized();
    Mat3 h_plane_to_norm;
    h_plane_to_norm.col(0) = e1;
    h_plane_to_norm.col(1) = e2;
    h_plane_to_norm.col(2) = p0;
    return h_plane_to_norm.inverse();
}

LineScanCalibrationResult calibrate_laser_plane(const std::vector<LineScanObservation>& views,
                                                const PinholeCamera<DualDistortion>& camera) {
    // Validate input observations
    validate_observations(views);

    LineScanCalibrationResult result;
    std::vector<Vec3> all_points;

    // Process each view to get 3D points
    for (const auto& view : views) {
        std::vector<Vec3> view_points = process_view(view, camera);
        all_points.insert(all_points.end(), view_points.begin(), view_points.end());
    }

    // Fit plane to all points
    result.plane = fit_plane(all_points, result.summary);

    // Compute statistics
    compute_plane_statistics(all_points, result.plane, result);

    // Build homography
    result.homography = build_plane_homography(result.plane);

    return result;
}

}  // namespace calib
