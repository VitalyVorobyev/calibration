#include "calib/pipeline/linescan.h"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace calib;

// ---- Helpers ---------------------------------------------------------------
constexpr double kEps = 1e-12;
constexpr double kHalf = 0.5;              // square is [-0.5, 0.5]^2
constexpr double kSamplesPerUnit = 400.0;  // along the target segment

using Line3 = Eigen::ParametrizedLine<double, 3>;
using Plane = Eigen::Hyperplane<double, 3>;

static auto make_target_plane_in_camera(const Eigen::Isometry3d& c_se3_t) -> Plane {
    const Eigen::Vector3d p0 = c_se3_t * Eigen::Vector3d(0, 0, 0);
    const Eigen::Vector3d p1 = c_se3_t * Eigen::Vector3d(1, 0, 0);
    const Eigen::Vector3d p2 = c_se3_t * Eigen::Vector3d(0, 1, 0);
    return Plane::Through(p0, p1, p2);
}

// Returns nullopt if planes are (near) parallel.
static auto plane_plane_intersection(const Plane& a, const Plane& b) -> std::optional<Line3> {
    const Eigen::Vector3d na = a.normal();
    const Eigen::Vector3d nb = b.normal();
    const double da = a.offset();  // plane: n·x + d = 0
    const double db = b.offset();

    const Eigen::Vector3d dir = na.cross(nb);
    if (dir.squaredNorm() < kEps) return std::nullopt;

    // Solve for a point p on both planes with the extra constraint dir·p = 0
    // to make the 3x3 system well-posed:
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

// Clip parametric line X(s) = p + s*d to the axis-aligned square [-kHalf,kHalf]^2 in XY.
// Returns {smin,smax} or nullopt if it misses.
static auto clip_to_square_xy(const Eigen::Vector3d& p, const Eigen::Vector3d& d)
    -> std::optional<std::pair<double, double>> {
    auto clip_1d = [&](double p0, double v, double lo, double hi, double& smin,
                       double& smax) -> bool {
        if (std::abs(v) < kEps) {
            return (p0 >= lo - 1e-14 && p0 <= hi + 1e-14);
        }
        double s0 = (lo - p0) / v;
        double s1 = (hi - p0) / v;
        if (s0 > s1) std::swap(s0, s1);
        smin = std::max(smin, s0);
        smax = std::min(smax, s1);
        return smin < smax;
    };

    double smin = -std::numeric_limits<double>::infinity();
    double smax = std::numeric_limits<double>::infinity();
    if (!clip_1d(p.x(), d.x(), -kHalf, kHalf, smin, smax)) return std::nullopt;
    if (!clip_1d(p.y(), d.y(), -kHalf, kHalf, smin, smax)) return std::nullopt;
    if (!(smin < smax)) return std::nullopt;
    return std::make_pair(smin, smax);
}

static auto sample_segment_xy_on_plane(const Eigen::Vector3d& p, const Eigen::Vector3d& d,
                                       double smin, double smax) -> std::vector<Eigen::Vector3d> {
    const Eigen::Vector2d a = (p + smin * d).head<2>();
    const Eigen::Vector2d b = (p + smax * d).head<2>();
    const double L = (b - a).norm();
    const int N = std::max(2, static_cast<int>(std::ceil(L * kSamplesPerUnit)));

    std::vector<Eigen::Vector3d> pts;
    pts.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        const double s = smin + t * (smax - smin);
        Eigen::Vector3d X = p + s * d;
        X.z() = 0.0;  // enforce the target plane z=0
        pts.emplace_back(X);
    }
    return pts;
}

template <class CameraT>
static auto project_points_target_to_pixels(const CameraT& cam, const Eigen::Isometry3d& pose_t2c,
                                            const std::vector<Eigen::Vector3d>& pts_t)
    -> std::vector<Eigen::Vector2d> {
    std::vector<Eigen::Vector2d> uv;
    uv.reserve(pts_t.size());
    for (const auto& X_t : pts_t) {
        const Eigen::Vector3d X_c = pose_t2c * X_t;
        uv.emplace_back(cam.project(X_c));
    }
    return uv;
}

template <class CameraT>
static void fill_target_view_from_xy(const Eigen::Isometry3d& pose_t2c, const CameraT& cam,
                                     const std::vector<Eigen::Vector2d>& xy, PlanarView& out_view) {
    out_view.resize(xy.size());
    std::transform(xy.begin(), xy.end(), out_view.begin(),
                   [&](const Eigen::Vector2d& p) -> PlanarObservation {
                       const Eigen::Vector3d X_c = pose_t2c * Eigen::Vector3d(p.x(), p.y(), 0.0);
                       return {p, cam.project(X_c)};
                   });
}

// ---- Orchestrator ----------------------------------------------------------

static auto create_view(const Eigen::Isometry3d& c_se3_t,
                        const Eigen::Hyperplane<double, 3>& laser_plane,
                        const PinholeCamera<DualDistortion>& camera) -> LineScanView {
    // 1) Target feature points in target frame
    const std::vector<Eigen::Vector2d> object_xy = {
        {-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};

    // 2) Init view and fill planar features
    LineScanView view;
    fill_target_view_from_xy(c_se3_t, camera, object_xy, view.target_view);

    // 3) Make target plane in camera frame
    const auto target_plane_c = make_target_plane_in_camera(c_se3_t);

    // 4) Intersect planes -> line in camera frame
    const auto line_c_opt = plane_plane_intersection(target_plane_c, laser_plane);
    if (!line_c_opt) {
        view.laser_uv.clear();
        return view;
    }

    // 5) Move line into target frame (for easy clipping)
    const Eigen::Isometry3d t_se3_c = c_se3_t.inverse();
    const Line3 line_t = to_target_frame(*line_c_opt, t_se3_c);

    // 6) Clip to square ROI in target XY
    const auto interval = clip_to_square_xy(line_t.origin(), line_t.direction());
    if (!interval) {
        view.laser_uv.clear();
        return view;
    }

    // 7) Sample along the segment on the target plane
    const auto pts_t = sample_segment_xy_on_plane(line_t.origin(), line_t.direction(),
                                                  interval->first, interval->second);

    // 8) Project to pixels
    view.laser_uv = project_points_target_to_pixels(camera, c_se3_t, pts_t);

    return view;
}

TEST(LineScanCalibration, PlaneFitFailsSingleView) {
    CameraMatrix kmtx{1.0, 1.0, 0.0, 0.0};
    auto dist = Eigen::VectorXd::Zero(5);
    PinholeCamera<DualDistortion> camera(kmtx, dist);

    const LineScanView view =
        create_view(Eigen::Isometry3d::Identity(),
                    Eigen::Hyperplane<double, 3>(Eigen::Vector3d(0, 1, 0), -0.5), camera);

    ASSERT_THROW(calibrate_laser_plane({view}, camera), std::invalid_argument);
}

TEST(LineScanCalibration, PlaneFitMultipleViews) {
    CameraMatrix kmtx{1.0, 1.0, 0.0, 0.0};
    auto dist = Eigen::VectorXd::Zero(5);
    PinholeCamera<DualDistortion> camera(kmtx, dist);

    const Eigen::Vector3d plane_norm = Eigen::Vector3d(0.1, 1, -0.1).normalized();
    constexpr double plane_d = 0.5;
    Eigen::Hyperplane<double, 3> plane(plane_norm, plane_d);

    Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
    pose1.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);

    Eigen::Isometry3d pose2 = Eigen::Isometry3d::Identity();
    pose2.linear() = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    pose2.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);

    auto v1 = create_view(pose1, plane, camera);
    auto v2 = create_view(pose2, plane, camera);

    auto res = calibrate_laser_plane({v1, v2}, camera);
    EXPECT_NEAR(res.plane[0], plane_norm.x(), 1e-6);
    EXPECT_NEAR(res.plane[1], plane_norm.y(), 1e-6);
    EXPECT_NEAR(res.plane[2], plane_norm.z(), 1e-6);
    EXPECT_NEAR(res.plane[3], plane_d, 1e-6);
    EXPECT_NEAR(res.rms_error, 0.0, 1e-9);
}
