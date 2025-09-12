#include <gtest/gtest.h>

#include "calib/intrinsics.h"
#include "handeyedata.h"

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
    constexpr size_t num_frames = 15;
    constexpr int rows = 8;
    constexpr int cols = 11;
    constexpr double spacing = 0.02;
    sim.make_sequence(num_frames, rng);
    sim.make_target_grid(rows, cols, spacing);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) {
        views.push_back(ob.view);
    }

    IntrinsicsEstimateOptions opts;
    opts.use_skew = false;
    IntrinsicsEstimateResult ires = estimate_intrinsics(views, opts);
    ASSERT_TRUE(ires.success);
    EXPECT_NEAR(ires.kmtx.fx, cam_gt.kmtx.fx, 1e-3);
    EXPECT_NEAR(ires.kmtx.fy, cam_gt.kmtx.fy, 1e-3);
    EXPECT_NEAR(ires.kmtx.cx, cam_gt.kmtx.cx, 1e-3);
    EXPECT_NEAR(ires.kmtx.cy, cam_gt.kmtx.cy, 1e-3);
    EXPECT_NEAR(ires.kmtx.skew, cam_gt.kmtx.skew, 1e-6);
    EXPECT_EQ(ires.views.size(), views.size());
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
    const IntrinsicsEstimateResult ires = estimate_intrinsics(views, opts);
    ASSERT_TRUE(ires.success);
    const auto& Kf = ires.kmtx;
    const auto& K_gt = cam_gt.kmtx;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-3);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-3);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-3);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-3);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-5);
}
