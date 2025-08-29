#include <gtest/gtest.h>
#include <Eigen/Core>

#include "calibration/scheimpflug.h"

using namespace vitavision;

TEST(ScheimpflugCamera, ZeroTiltMatchesPinhole) {
    Camera cam;
    cam.K.fx = 800; cam.K.fy = 820; cam.K.cx = 320; cam.K.cy = 240;
    cam.distortion.forward = Eigen::VectorXd::Zero(2);
    cam.distortion.inverse = Eigen::VectorXd::Zero(2);

    ScheimpflugCamera sc(cam, 0.0, 0.0);

    Eigen::Vector3d Xc(0.2, -0.1, 1.0);
    Eigen::Vector2d uv_s = sc.project<double>(Xc);
    Eigen::Vector2d uv_p = cam.project(Eigen::Vector2d(Xc.x()/Xc.z(), Xc.y()/Xc.z()));

    EXPECT_NEAR(uv_s.x(), uv_p.x(), 1e-9);
    EXPECT_NEAR(uv_s.y(), uv_p.y(), 1e-9);
}

TEST(ScheimpflugCamera, PrincipalRayStaysAtCenter) {
    Camera cam;
    cam.K.fx = 600; cam.K.fy = 600; cam.K.cx = 400; cam.K.cy = 300;
    cam.distortion.forward = Eigen::VectorXd::Zero(2);
    cam.distortion.inverse = Eigen::VectorXd::Zero(2);

    ScheimpflugCamera sc(cam, 0.1, -0.2);

    Eigen::Vector3d Xc(0.0, 0.0, 1.0);
    Eigen::Vector2d uv = sc.project<double>(Xc);

    EXPECT_NEAR(uv.x(), cam.K.cx, 1e-9);
    EXPECT_NEAR(uv.y(), cam.K.cy, 1e-9);
}

