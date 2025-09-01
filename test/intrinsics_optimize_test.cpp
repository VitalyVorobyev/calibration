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

    // Use proper transforms: camera at origin looking at target in front
    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity(); // gripper to camera
    Eigen::Affine3d b_T_t = Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Affine3d::Identity(); // base to target (target 2m away)
    SimulatedHandEye sim{g_T_c, b_T_t, cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) views.push_back(ob.view);

    // Create initial camera guess
    Camera<BrownConradyd> guess_cam = cam_gt;
    guess_cam.K.fx *= 0.97; guess_cam.K.fy *= 1.03;
    guess_cam.K.cx += 5.0; guess_cam.K.cy -= 4.0;
    guess_cam.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Estimate initial poses for each view
    std::vector<Eigen::Affine3d> init_poses;
    init_poses.reserve(views.size());
    for (const auto& view : views) {
        auto pose = estimate_planar_pose_dlt(view, guess_cam.K);
        init_poses.push_back(pose);
    }

    IntrinsicsOptions opts;
    opts.num_radial = 3;
    opts.optimize_skew = false;
    auto res = optimize_intrinsics(views, guess_cam, init_poses, opts);

    const auto& Kf = res.camera.K;
    const auto& K_gt = cam_gt.K;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-6);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-6);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-6);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-6);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-9);
    EXPECT_LT(res.final_cost, 1e-6);
}

TEST(OptimizeIntrinsics, RecoversSkew) {
    RNG rng(5);

    Camera<BrownConradyd> cam_gt;
    cam_gt.K.fx = 1000; cam_gt.K.fy = 1005;
    cam_gt.K.cx = 640;  cam_gt.K.cy = 360;
    cam_gt.K.skew = 0.001;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Use proper transforms: camera at origin looking at target in front
    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity(); // gripper to camera
    Eigen::Affine3d b_T_t = Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Affine3d::Identity(); // base to target (target 2m away)
    SimulatedHandEye sim{g_T_c, b_T_t, cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) views.push_back(ob.view);

    // Create initial camera guess
    Camera<BrownConradyd> guess_cam = cam_gt;
    guess_cam.K.fx *= 0.95; guess_cam.K.fy *= 1.05;
    guess_cam.K.cx += 10.0; guess_cam.K.cy -= 6.0;
    guess_cam.K.skew = 0.0;
    guess_cam.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Estimate initial poses for each view
    std::vector<Eigen::Affine3d> init_poses;
    init_poses.reserve(views.size());
    for (const auto& view : views) {
        auto pose = estimate_planar_pose_dlt(view, guess_cam.K);
        init_poses.push_back(pose);
    }

    IntrinsicsOptions opts;
    opts.num_radial = 0;
    opts.optimize_skew = true;
    auto res = optimize_intrinsics(views, guess_cam, init_poses, opts);

    const auto& Kf = res.camera.K;
    const auto& K_gt = cam_gt.K;
    EXPECT_NEAR(Kf.fx, K_gt.fx, 1e-6);
    EXPECT_NEAR(Kf.fy, K_gt.fy, 1e-6);
    EXPECT_NEAR(Kf.cx, K_gt.cx, 1e-6);
    EXPECT_NEAR(Kf.cy, K_gt.cy, 1e-6);
    EXPECT_NEAR(Kf.skew, K_gt.skew, 1e-9);
}
