#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "calib/estimation/linear/linescan.h"
#include "calib/estimation/linear/planefit.h"
#include "calib/models/pinhole.h"

using namespace calib;

TEST(LinescanUtils, FitPlaneSVDDetectsPlane) {
    std::vector<Eigen::Vector3d> pts;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            pts.emplace_back(static_cast<double>(i), static_cast<double>(j), 0.0);
        }
    }
    Eigen::Vector4d plane = fit_plane_svd(pts);
    EXPECT_NEAR(plane[0], 0.0, 1e-12);
    EXPECT_NEAR(plane[1], 0.0, 1e-12);
    EXPECT_NEAR(plane[2], 1.0, 1e-12);
    EXPECT_NEAR(plane[3], 0.0, 1e-12);
}

TEST(LinescanUtils, PlaneRMSZeroForExactPoints) {
    std::vector<Eigen::Vector3d> pts = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
                                        Eigen::Vector3d(0, 1, 0)};
    Eigen::Vector4d plane = fit_plane_svd(pts);
    EXPECT_NEAR(plane_rms(pts, plane), 0.0, 1e-12);
}

TEST(LinescanUtils, PointsFromViewProduces3DPoints) {
    CameraMatrix kmtx{400.0, 400.0, 0.0, 0.0, 0.0};
    PinholeCamera<DualDistortion> camera(kmtx, DualDistortion{Eigen::VectorXd::Zero(2)});

    LineScanView view;
    view.target_view = {{Eigen::Vector2d(-0.5, -0.5), Eigen::Vector2d(-200, -200)},
                        {Eigen::Vector2d(0.5, -0.5), Eigen::Vector2d(200, -200)},
                        {Eigen::Vector2d(0.5, 0.5), Eigen::Vector2d(200, 200)},
                        {Eigen::Vector2d(-0.5, 0.5), Eigen::Vector2d(-200, 200)}};
    view.laser_uv = {Eigen::Vector2d(-50, 0), Eigen::Vector2d(0, 0), Eigen::Vector2d(50, 0)};

    auto pts = points_from_view(view, camera);
    EXPECT_EQ(pts.size(), view.laser_uv.size());
}
