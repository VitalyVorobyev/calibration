#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration/linescan.h"

using namespace vitavision;

TEST(LineScanCalibration, PlaneFitFailsSingleView) {
    CameraMatrix K{1.0, 1.0, 0.0, 0.0};
    Eigen::VectorXd dist;
    Camera camera(K, dist);

    LineScanObservation view;
    view.target_xy = {
        {-0.5, -0.5},
        { 0.5, -0.5},
        { 0.5,  0.5},
        {-0.5,  0.5}
    };
    view.target_uv = view.target_xy;

    // Laser plane y = 0.5 -> normal (0,1,0), d = -0.5
    for (double x = -0.4; x <= 0.4; x += 0.2) {
        view.laser_uv.emplace_back(x, 0.5);
    }

    ASSERT_THROW(calibrate_laser_plane({view}, camera), std::invalid_argument);
}

static LineScanObservation create_view(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    LineScanObservation v;
    v.target_xy = {
        {-0.5, -0.5},
        { 0.5, -0.5},
        { 0.5,  0.5},
        {-0.5,  0.5}
    };
    for (const auto& pt : v.target_xy) {
        Eigen::Vector3d Pc = R * Eigen::Vector3d(pt.x(), pt.y(), 0.0) + t;
        v.target_uv.emplace_back(Pc.x() / Pc.z(), Pc.y() / Pc.z());
    }
    for (double x = -0.4; x <= 0.4; x += 0.2) {
        double y = (0.5 - t.y() - R(1,0) * x) / R(1,1);
        Eigen::Vector3d Pc = R * Eigen::Vector3d(x, y, 0.0) + t;
        v.laser_uv.emplace_back(Pc.x() / Pc.z(), Pc.y() / Pc.z());
    }
    return v;
}

TEST(LineScanCalibration, PlaneFitMultipleViews) {
    CameraMatrix K{1.0, 1.0, 0.0, 0.0};
    Eigen::VectorXd dist;

    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1(0.0, 0.0, 1.0);
    Eigen::Matrix3d R2 = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Vector3d t2(0.0, 0.0, 1.0);

    auto v1 = create_view(R1, t1);
    auto v2 = create_view(R2, t2);

    auto res = calibrate_laser_plane({v1, v2}, K, dist);
    EXPECT_NEAR(res.plane[0], 0.0, 1e-6);
    EXPECT_NEAR(res.plane[1], 1.0, 1e-6);
    EXPECT_NEAR(res.plane[2], 0.0, 1e-6);
    EXPECT_NEAR(res.plane[3], -0.5, 1e-6);
    EXPECT_NEAR(res.rms_error, 0.0, 1e-9);
}
