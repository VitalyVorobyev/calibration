#include <gtest/gtest.h>
#include <Eigen/Core>

#include "calibration/linescan.h"

using namespace vitavision;

TEST(LineScanCalibration, PlaneFit) {
    CameraMatrix K{1.0, 1.0, 0.0, 0.0};

    LineScanObservation view;
    // four target correspondences (square)
    view.target_xy = {
        {-0.5, -0.5},
        { 0.5, -0.5},
        { 0.5,  0.5},
        {-0.5,  0.5}
    };
    view.target_uv = view.target_xy; // with identity intrinsics this matches

    // Laser plane y = 0.5 -> normal (0,1,0), d = -0.5
    for (double x = -0.4; x <= 0.4; x += 0.2) {
        view.laser_uv.emplace_back(x, 0.5);
    }

    auto res = calibrate_laser_plane({view}, K);
    for (const auto& lpix : view.laser_uv) {
        Eigen::Vector3d hp = res.homography * Eigen::Vector3d(lpix.x(), lpix.y(), 1.0);
        Eigen::Vector2d xy = hp.hnormalized();
        EXPECT_NEAR(xy.x(), lpix.x(), 1e-6);
        EXPECT_NEAR(xy.y(), -1.0, 1e-6);
    }
    EXPECT_NEAR(res.plane[0], 0.0, 1e-6);
    EXPECT_NEAR(res.plane[1], 1.0, 1e-6);
    EXPECT_NEAR(res.plane[2], 0.0, 1e-6);
    EXPECT_NEAR(res.plane[3], -0.5, 1e-6);
    EXPECT_NEAR(res.rms_error, 0.0, 1e-9);
}

