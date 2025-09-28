#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "calib/estimation/linear/linescan.h"
#include "calib/pipeline/linescan.h"

using namespace calib;

namespace {

constexpr double k_eps = 1e-12;
constexpr double k_half = 0.5;
constexpr double k_samples_per_unit = 200.0;

using Line3 = Eigen::ParametrizedLine<double, 3>;
using Plane = Eigen::Hyperplane<double, 3>;

static auto make_target_plane_in_camera(const Eigen::Isometry3d& c_se3_t) -> Plane {
    const Eigen::Vector3d p0 = c_se3_t * Eigen::Vector3d(0, 0, 0);
    const Eigen::Vector3d p1 = c_se3_t * Eigen::Vector3d(1, 0, 0);
    const Eigen::Vector3d p2 = c_se3_t * Eigen::Vector3d(0, 1, 0);
    return Plane::Through(p0, p1, p2);
}

static auto plane_plane_intersection(const Plane& a, const Plane& b) -> std::optional<Line3> {
    const Eigen::Vector3d na = a.normal();
    const Eigen::Vector3d nb = b.normal();
    const double da = a.offset();
    const double db = b.offset();
    const Eigen::Vector3d dir = na.cross(nb);
    if (dir.squaredNorm() < k_eps) return std::nullopt;
    Eigen::Matrix3d A;
    A.row(0) = na.transpose();
    A.row(1) = nb.transpose();
    A.row(2) = dir.transpose();
    const Eigen::Vector3d rhs(-da, -db, 0.0);
    const Eigen::Vector3d p = A.fullPivLu().solve(rhs);
    return Line3(p, dir.normalized());
}

static auto to_target_frame(const Line3& line_c, const Eigen::Isometry3d& t_se3_c) -> Line3 {
    return {t_se3_c * line_c.origin(), t_se3_c.linear() * line_c.direction()};
}

static auto clip_to_square_xy(const Eigen::Vector3d& p, const Eigen::Vector3d& d)
    -> std::optional<std::pair<double, double>> {
    auto clip1d = [](double p0, double v, double lo, double hi, double& smin, double& smax) {
        if (std::abs(v) < k_eps) return (p0 >= lo - 1e-14 && p0 <= hi + 1e-14);
        double s0 = (lo - p0) / v;
        double s1 = (hi - p0) / v;
        if (s0 > s1) std::swap(s0, s1);
        smin = std::max(smin, s0);
        smax = std::min(smax, s1);
        return smin < smax;
    };
    double smin = -std::numeric_limits<double>::infinity();
    double smax = std::numeric_limits<double>::infinity();
    if (!clip1d(p.x(), d.x(), -k_half, k_half, smin, smax)) return std::nullopt;
    if (!clip1d(p.y(), d.y(), -k_half, k_half, smin, smax)) return std::nullopt;
    if (!(smin < smax)) return std::nullopt;
    return std::make_pair(smin, smax);
}

static auto sample_segment_xy_on_plane(const Eigen::Vector3d& p, const Eigen::Vector3d& d,
                                       double smin, double smax) -> std::vector<Eigen::Vector3d> {
    const Eigen::Vector2d a = (p + smin * d).head<2>();
    const Eigen::Vector2d b = (p + smax * d).head<2>();
    const double L = (b - a).norm();
    const int n = std::max(2, static_cast<int>(std::ceil(L * k_samples_per_unit)));
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = (n == 1) ? 0.0 : double(i) / double(n - 1);
        const double s = smin + t * (smax - smin);
        Eigen::Vector3d x = p + s * d;
        x.z() = 0.0;
        pts.emplace_back(x);
    }
    return pts;
}

template <class CameraT>
static auto project_points_target_to_pixels(const CameraT& cam, const Eigen::Isometry3d& pose_t2c,
                                            const std::vector<Eigen::Vector3d>& pts_t)
    -> std::vector<Eigen::Vector2d> {
    std::vector<Eigen::Vector2d> uv;
    uv.reserve(pts_t.size());
    for (const auto& x_t : pts_t) {
        const Eigen::Vector3d x_c = pose_t2c * x_t;
        uv.emplace_back(cam.project(x_c));
    }
    return uv;
}

template <class CameraT>
static void fill_target_view_from_xy(const Eigen::Isometry3d& pose_t2c, const CameraT& cam,
                                     const std::vector<Eigen::Vector2d>& xy, PlanarView& out_view) {
    out_view.resize(xy.size());
    std::transform(xy.begin(), xy.end(), out_view.begin(),
                   [&](const Eigen::Vector2d& p) -> PlanarObservation {
                       const Eigen::Vector3d x_c = pose_t2c * Eigen::Vector3d(p.x(), p.y(), 0.0);
                       return {p, cam.project(x_c)};
                   });
}

static auto create_view(const Eigen::Isometry3d& c_se3_t, const Plane& laser_plane,
                        const PinholeCamera<DualDistortion>& camera) -> LineScanView {
    const std::vector<Eigen::Vector2d> object_xy = {
        {-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};
    LineScanView view;
    fill_target_view_from_xy(c_se3_t, camera, object_xy, view.target_view);
    const auto target_plane_c = make_target_plane_in_camera(c_se3_t);
    const auto line_c_opt = plane_plane_intersection(target_plane_c, laser_plane);
    if (!line_c_opt) return view;
    const Eigen::Isometry3d t_se3_c = c_se3_t.inverse();
    const Line3 line_t = to_target_frame(*line_c_opt, t_se3_c);
    const auto interval = clip_to_square_xy(line_t.origin(), line_t.direction());
    if (!interval) return view;
    const auto pts_t = sample_segment_xy_on_plane(line_t.origin(), line_t.direction(),
                                                  interval->first, interval->second);
    view.laser_uv = project_points_target_to_pixels(camera, c_se3_t, pts_t);
    return view;
}

}  // namespace

TEST(LinescanFacade, CalibratesFromViews) {
    CameraMatrix kmtx{400.0, 402.0, 0.0, 0.0};
    Eigen::VectorXd dist = Eigen::VectorXd::Zero(5);
    PinholeCamera<BrownConradyd> cam_b(kmtx, dist);
    PinholeCamera<DualDistortion> cam_d(kmtx, DualDistortion{Eigen::VectorXd::Zero(2)});

    const Eigen::Vector3d n = Eigen::Vector3d(0.1, 1.0, -0.05).normalized();
    constexpr double d = 0.4;
    Eigen::Hyperplane<double, 3> laser(n, d);

    Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
    pose1.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);

    Eigen::Isometry3d pose2 = Eigen::Isometry3d::Identity();
    pose2.linear() = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    pose2.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);

    auto v1 = create_view(pose1, laser, cam_d);
    auto v2 = create_view(pose2, laser, cam_d);

    pipeline::LinescanCalibrationFacade facade;
    auto run = facade.calibrate(cam_b, {v1, v2});
    ASSERT_TRUE(run.success);
    EXPECT_EQ(run.used_views, 2U);
    EXPECT_NEAR(run.result.plane[0], n.x(), 1e-3);
    EXPECT_NEAR(run.result.plane[1], n.y(), 1e-3);
    EXPECT_NEAR(run.result.plane[2], n.z(), 1e-3);
    EXPECT_NEAR(run.result.plane[3], d, 1e-2);
}
