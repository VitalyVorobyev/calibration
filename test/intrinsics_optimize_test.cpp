#include <gtest/gtest.h>

#include "calib/intrinsics.h"
#include "utils.h"

using namespace calib;

TEST(OptimizeIntrinsics, RecoversIntrinsicsNoSkew) {
    RNG rng(7);

    Camera<BrownConradyd> cam_gt;
    cam_gt.K.fx = 1000; cam_gt.K.fy = 1005;
    cam_gt.K.cx = 640;  cam_gt.K.cy = 360;
    cam_gt.K.skew = 0.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    SimulatedHandEye sim{Eigen::Affine3d::Identity(), Eigen::Affine3d::Identity(), cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) views.push_back(ob.view);

    CameraMatrix guess = cam_gt.K;
    guess.fx *= 0.97; guess.fy *= 1.03;
    guess.cx += 5.0; guess.cy -= 4.0;

    IntrinsicsOptions opts;
    opts.num_radial = 3;
    opts.optimize_skew = false;
    auto res = optimize_intrinsics(views, guess, opts);

    const auto& Kf = res.camera.K;
    const auto& K_gt = cam_gt.K;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-6);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-6);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-6);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-6);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-9);
    EXPECT_LT(res.reprojection_error, 1e-6);
}

TEST(OptimizeIntrinsics, RecoversSkew) {
    RNG rng(5);

    Camera<BrownConradyd> cam_gt;
    cam_gt.K.fx = 1000; cam_gt.K.fy = 1005;
    cam_gt.K.cx = 640;  cam_gt.K.cy = 360;
    cam_gt.K.skew = 0.001;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    SimulatedHandEye sim{Eigen::Affine3d::Identity(), Eigen::Affine3d::Identity(), cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) views.push_back(ob.view);

    CameraMatrix guess = cam_gt.K;
    guess.fx *= 0.95; guess.fy *= 1.05;
    guess.cx += 10.0; guess.cy -= 6.0;
    guess.skew = 0.0;

    IntrinsicsOptions opts;
    opts.num_radial = 0;
    opts.optimize_skew = true;
    auto res = optimize_intrinsics(views, guess, opts);

    const auto& Kf = res.camera.K;
    const auto& K_gt = cam_gt.K;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-6);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-6);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-6);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-6);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-9);
}
