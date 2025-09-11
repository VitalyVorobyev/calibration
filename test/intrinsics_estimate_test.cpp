#include <gtest/gtest.h>

#include "calib/intrinsics.h"
#include "utils.h"

using namespace calib;

TEST(EstimateIntrinsics, RecoversIntrinsicsNoSkew) {
    RNG rng(7);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 1000; cam_gt.kmtx.fy = 1005;
    cam_gt.kmtx.cx = 640;  cam_gt.kmtx.cy = 360;
    cam_gt.kmtx.skew = 0.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d b_se3_t =
        Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Isometry3d::Identity();

    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) {
        views.push_back(ob.view);
    }

    IntrinsicsEstimateOptions opts;
    opts.use_skew = false;
    auto res_opt = estimate_intrinsics(views, Eigen::Vector2i(1280, 720), opts);
    ASSERT_TRUE(res_opt.has_value());
    const auto& Kf = res_opt->kmtx;
    const auto& K_gt = cam_gt.kmtx;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-3);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-3);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-3);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-3);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-6);
    EXPECT_EQ(res_opt->c_se3_t.size(), views.size());
}

TEST(EstimateIntrinsics, RecoversSkew) {
    RNG rng(5);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 1000; cam_gt.kmtx.fy = 1005;
    cam_gt.kmtx.cx = 640;  cam_gt.kmtx.cy = 360;
    cam_gt.kmtx.skew = 0.001;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d b_se3_t =
        Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Isometry3d::Identity();

    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) {
        views.push_back(ob.view);
    }

    IntrinsicsEstimateOptions opts;
    opts.use_skew = true;
    auto res_opt = estimate_intrinsics(views, Eigen::Vector2i(1280, 720), opts);
    ASSERT_TRUE(res_opt.has_value());
    const auto& Kf = res_opt->kmtx;
    const auto& K_gt = cam_gt.kmtx;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-3);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-3);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-3);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-3);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-5);
}

