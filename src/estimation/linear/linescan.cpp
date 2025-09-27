#include "calib/estimation/linear/linescan.h"

// std
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace calib {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

static void validate_observations(const std::vector<LineScanView>& views) {
    if (views.size() < 2) {
        throw std::invalid_argument("At least 2 views are required");
    }
    if (std::any_of(views.begin(), views.end(),
                    [](const auto& v) { return v.target_view.size() < 4; })) {
        throw std::invalid_argument("Each view requires >=4 target correspondences");
    }
}

static Eigen::Vector4d fit_plane_svd(const std::vector<Vec3>& pts) {
    Vec3 centroid = std::accumulate(pts.cbegin(), pts.cend(), Vec3{Vec3::Zero()},
                                    [](const Vec3& a, const Vec3& b) { return a + b; });
    centroid /= static_cast<double>(pts.size());
    Eigen::MatrixXd A(static_cast<Eigen::Index>(pts.size()), 3);
    for (size_t i = 0; i < pts.size(); ++i)
        A.row(static_cast<Eigen::Index>(i)) = (pts[i] - centroid).transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Vec3 n = svd.matrixV().col(2);
    double d = -n.dot(centroid);
    const double nrm = n.norm();
    return {n.x() / nrm, n.y() / nrm, n.z() / nrm, d / nrm};
}

static std::vector<Vec3> process_view(LineScanView view,
                                      const PinholeCamera<DualDistortion>& camera) {
    std::for_each(
        view.target_view.begin(), view.target_view.end(),
        [&camera](PlanarObservation& item) { item.image_uv = camera.unproject(item.image_uv); });

    auto hres = estimate_homography(view.target_view);
    if (!hres.success) return {};

    Eigen::Isometry3d pose = pose_from_homography_normalized(hres.hmtx);
    Mat3 h_norm_to_obj = hres.hmtx.inverse();
    h_norm_to_obj /= h_norm_to_obj(2, 2);

    std::vector<Vec3> points;
    points.reserve(view.laser_uv.size());
    for (const auto& lpix : view.laser_uv) {
        Vec2 norm = camera.unproject(lpix);
        Eigen::Vector3d hp = h_norm_to_obj * Eigen::Vector3d(norm.x(), norm.y(), 1.0);
        Vec2 plane_xy = hp.hnormalized();
        Vec3 obj_pt(plane_xy.x(), plane_xy.y(), 0.0);
        Vec3 cam_pt = pose * obj_pt;
        points.push_back(cam_pt);
    }
    return points;
}

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

auto calibrate_laser_plane(const std::vector<LineScanView>& views,
                           const PinholeCamera<DualDistortion>& camera)
    -> LineScanCalibrationResult {
    validate_observations(views);

    std::vector<Vec3> all_points;
    for (const auto& view : views) {
        auto pts = process_view(view, camera);
        all_points.insert(all_points.end(), pts.begin(), pts.end());
    }
    if (all_points.size() < 3) {
        throw std::invalid_argument("Not enough laser points to fit a plane");
    }

    LineScanCalibrationResult result;
    result.plane = fit_plane_svd(all_points);

    // RMS
    double ssr = 0.0;
    for (const auto& p : all_points) {
        const double r = result.plane.head<3>().dot(p) + result.plane[3];
        ssr += r * r;
    }
    result.rms_error = std::sqrt(ssr / static_cast<double>(all_points.size()));

    // Homography (plane frame definition)
    result.homography = build_plane_homography(result.plane);

    result.covariance.setZero();
    result.summary = "linear_svd";
    return result;
}

}  // namespace calib
