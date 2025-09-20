#include "calib/scheimpflug.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "calib/pinhole.h"

using namespace calib;

TEST(ScheimpflugCamera, ZeroTiltMatchesPinhole) {
    PinholeCamera<DualDistortion> cam;
    cam.kmtx.fx = 800;
    cam.kmtx.fy = 820;
    cam.kmtx.cx = 320;
    cam.kmtx.cy = 240;
    cam.distortion.forward = Eigen::VectorXd::Zero(2);
    cam.distortion.inverse = Eigen::VectorXd::Zero(2);

    ScheimpflugCamera<PinholeCamera<DualDistortion>> sc(cam, {0.0, 0.0});

    Eigen::Vector3d Xc(0.2, -0.1, 1.0);
    Eigen::Vector2d uv_s = sc.project(Xc);
    Eigen::Vector2d uv_p = cam.project(Xc);

    EXPECT_NEAR(uv_s.x(), uv_p.x(), 1e-9);
    EXPECT_NEAR(uv_s.y(), uv_p.y(), 1e-9);
}

TEST(ScheimpflugCamera, PrincipalRay) {
    PinholeCamera<DualDistortion> cam;
    cam.kmtx.fx = 600;
    cam.kmtx.fy = 600;
    cam.kmtx.cx = 400;
    cam.kmtx.cy = 300;
    cam.distortion.forward = Eigen::VectorXd::Zero(2);
    cam.distortion.inverse = Eigen::VectorXd::Zero(2);

    const double taux = 0.1;
    const double tauy = -0.2;
    ScheimpflugCamera<PinholeCamera<DualDistortion>> sc(cam, {taux, tauy});

    Eigen::Vector3d Xc(0.0, 0.0, 1.0);
    Eigen::Vector2d uv = sc.project(Xc);

    Eigen::Vector2d expected_m0{-std::tan(tauy) / std::cos(taux), std::tan(taux)};
    const auto expected_uv = cam.project(expected_m0);

    EXPECT_NEAR(uv.x(), expected_uv.x(), 1e-9);
    EXPECT_NEAR(uv.y(), expected_uv.y(), 1e-9);
}
