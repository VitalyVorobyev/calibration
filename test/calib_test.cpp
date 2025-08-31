#include "calib/calib.h"

// std
#include <random>

// gtest
#include <gtest/gtest.h>

using namespace calib;

static void distort_and_project(const Eigen::Vector3d& P,
                                const Eigen::Affine3d& pose,
                                const CameraMatrix& intr,
                                const std::vector<double>& k_radial,
                                double p1, double p2,
                                Eigen::Vector2d& uv) {
    Eigen::Vector3d Pc = pose * P;
    double x = Pc.x() / Pc.z();
    double y = Pc.y() / Pc.z();
    double r2 = x*x + y*y;
    double radial = 1.0;
    double rpow = r2;
    for (double k : k_radial) { radial += k * rpow; rpow *= r2; }
    double x_t = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double y_t = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    uv.x() = intr.fx * x_t + intr.cx;
    uv.y() = intr.fy * y_t + intr.cy;
}

TEST(CameraCalibrationTest, PlanarViewsExact) {
    CameraMatrix intr_true{800.0, 820.0, 640.0, 360.0};
    std::vector<double> k_rad = {-0.20, 0.03};
    double p1 = 0.001, p2 = -0.0005;

    // Planar grid points
    std::vector<Eigen::Vector2d> obj_xy;
    for (int i=-5;i<=5;i+=2) {
        for (int j=-5;j<=5;j+=2) {
            obj_xy.emplace_back(0.04*i, 0.04*j);
        }
    }

    // Generate several poses
    std::vector<PlanarView> views(4);
    std::vector<Eigen::Affine3d> poses_true(4, Eigen::Affine3d::Identity());
    poses_true[0].translation() = Eigen::Vector3d(0.1, -0.1, 2.0);
    poses_true[1].linear() = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    poses_true[1].translation() = Eigen::Vector3d(-0.2, 0.1, 1.8);
    poses_true[2].linear() = Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitX()).toRotationMatrix();
    poses_true[2].translation() = Eigen::Vector3d(0.05, 0.2, 2.2);
    poses_true[3].linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    poses_true[3].translation() = Eigen::Vector3d(-0.1, -0.15, 1.9);

    for (size_t v = 0; v < views.size(); ++v) {
        views[v].resize(obj_xy.size());
        for (size_t i = 0; i < obj_xy.size(); ++i) {
            Eigen::Vector3d P(obj_xy[i].x(), obj_xy[i].y(), 0.0);
            views[v][i].object_xy = obj_xy[i];
            distort_and_project(P, poses_true[v], intr_true, k_rad, p1, p2, views[v][i].image_uv);
        }
    }

    CameraMatrix guess{780.0, 800.0, 630.0, 350.0};
    auto res = calibrate_camera_planar(views, 2, guess, true);

    EXPECT_NEAR(res.intrinsics.fx, intr_true.fx, 1e-4);
    EXPECT_NEAR(res.intrinsics.fy, intr_true.fy, 1e-4);
    EXPECT_NEAR(res.intrinsics.cx, intr_true.cx, 1e-4);
    EXPECT_NEAR(res.intrinsics.cy, intr_true.cy, 1e-4);

    EXPECT_NEAR(res.distortion[0], k_rad[0], 1e-4);
    EXPECT_NEAR(res.distortion[1], k_rad[1], 1e-4);
    EXPECT_NEAR(res.distortion[2], p1, 1e-6);
    EXPECT_NEAR(res.distortion[3], p2, 1e-6);

    EXPECT_EQ(res.view_errors.size(), views.size());
    for (double e : res.view_errors) EXPECT_NEAR(e, 0.0, 1e-6);
    EXPECT_NEAR(res.reprojection_error, 0.0, 1e-6);
}
