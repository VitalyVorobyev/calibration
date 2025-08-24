#include "calibration/linescan.h"

// std
#include <numeric>
#include <stdexcept>
#include <cmath>

// ceres
#include <ceres/ceres.h>

#include "calibration/homography.h"
#include "calibration/planarpose.h"

namespace vitavision {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

namespace {

// Simple SVD-based plane initialization
static Eigen::Vector4d fit_plane_svd(const std::vector<Vec3>& pts) {
    Vec3 centroid = Vec3::Zero();
    for (const auto& p : pts) centroid += p;
    centroid /= static_cast<double>(pts.size());

    Eigen::MatrixXd A(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        A.row(i) = (pts[i] - centroid).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Vec3 normal = svd.matrixV().col(2);
    double d = -normal.dot(centroid);
    return {normal.x(), normal.y(), normal.z(), d};
}

struct PlaneResidual {
    PlaneResidual(const Vec3& p) : p_(p) {}

    template <typename T>
    bool operator()(const T* plane, T* residual) const {
        // plane = [nx, ny, nz, d]
        T nx = plane[0];
        T ny = plane[1];
        T nz = plane[2];
        T d  = plane[3];
        T denom = ceres::sqrt(nx*nx + ny*ny + nz*nz);
        residual[0] = (nx*T(p_.x()) + ny*T(p_.y()) + nz*T(p_.z()) + d) / denom;
        return true;
    }

    Vec3 p_;
};

} // namespace

LineScanCalibrationResult calibrate_laser_plane(
    const std::vector<LineScanObservation>& views,
    const CameraMatrix& intrinsics) {

    LineScanCalibrationResult result;
    std::vector<Vec3> all_points;

    for (size_t i = 0; i < views.size(); ++i) {
        const auto& v = views[i];
        if (v.target_xy.size() < 4 || v.target_xy.size() != v.target_uv.size()) {
            throw std::invalid_argument("Each view requires >=4 target correspondences");
        }

        // Normalize pixel coordinates
        std::vector<Vec2> img_norm(v.target_uv.size());
        std::transform(v.target_uv.begin(), v.target_uv.end(), img_norm.begin(),
            [&intrinsics](const Vec2& uv) { return intrinsics.normalize(uv); });

        // Homography from plane to normalized pixels
        Mat3 H_obj_to_norm = fit_homography(v.target_xy, img_norm);
        Mat3 H_norm_to_obj = H_obj_to_norm.inverse();

        // Pose of plane (world->camera)
        Eigen::Affine3d pose = estimate_planar_pose_dlt(v.target_xy, v.target_uv, intrinsics);

        // Reproject laser pixels to plane and transform to camera coordinates
        for (const auto& lpix : v.laser_uv) {
            Vec2 norm = intrinsics.normalize(lpix);
            Eigen::Vector3d hp = H_norm_to_obj * Eigen::Vector3d(norm.x(), norm.y(), 1.0);
            Vec2 plane_xy = hp.hnormalized();
            Vec3 obj_pt(plane_xy.x(), plane_xy.y(), 0.0);
            Vec3 cam_pt = pose * obj_pt;
            all_points.push_back(cam_pt);
        }
    }

    if (all_points.size() < 3) {
        throw std::invalid_argument("Not enough laser points to fit a plane");
    }

    Eigen::Vector4d init = fit_plane_svd(all_points);
    double params[4] = {init[0], init[1], init[2], init[3]};

    ceres::Problem problem;
    for (const auto& p : all_points) {
        auto* cost = new ceres::AutoDiffCostFunction<PlaneResidual,1,4>(new PlaneResidual(p));
        problem.AddResidualBlock(cost, nullptr, params);
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    result.summary = summary.BriefReport();

    // Compute residual stats
    double ssr = 0.0;
    double nrm = std::sqrt(params[0]*params[0]+params[1]*params[1]+params[2]*params[2]);
    for (const auto& p : all_points) {
        double r = (params[0]*p.x() + params[1]*p.y() + params[2]*p.z() + params[3]) / nrm;
        ssr += r*r;
    }
    int m = static_cast<int>(all_points.size());
    int dof = std::max(1, m - 3);
    double sigma2 = ssr / dof;
    result.rms_error = std::sqrt(ssr / m);

    // Covariance of plane params
    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> blocks = { {params, params} };
    if (cov.Compute(blocks, &problem)) {
        double Cov4[16];
        cov.GetCovarianceBlock(params, params, Cov4);
        Eigen::Map<Eigen::Matrix<double,4,4>> C(Cov4);
        C *= sigma2;
        result.covariance = C;
    } else {
        result.covariance.setZero();
    }

    // Normalise plane parameters
    nrm = std::sqrt(params[0]*params[0]+params[1]*params[1]+params[2]*params[2]);
    result.plane = Eigen::Vector4d(params[0]/nrm, params[1]/nrm, params[2]/nrm, params[3]/nrm);

    // Build homography from normalized pixels to plane coordinates
    Vec3 nvec = result.plane.head<3>();
    Vec3 p0 = -result.plane[3] * nvec; // closest point to camera
    Vec3 tmp = (std::abs(nvec.z()) < 0.9) ? Vec3::UnitZ() : Vec3::UnitX();
    Vec3 e1 = nvec.cross(tmp).normalized();
    Vec3 e2 = nvec.cross(e1).normalized();
    Mat3 H_plane_to_norm;
    H_plane_to_norm.col(0) = e1;
    H_plane_to_norm.col(1) = e2;
    H_plane_to_norm.col(2) = p0;
    result.homography = H_plane_to_norm.inverse();

    return result;
}

} // namespace vitavision

