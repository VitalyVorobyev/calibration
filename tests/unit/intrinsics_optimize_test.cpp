#include <gtest/gtest.h>

#include "calib/estimation/intrinsics.h"
#include "utils.h"

using namespace calib;

TEST(OptimizeIntrinsics, RecoversIntrinsicsNoSkew) {
    RNG rng(7);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 1000;
    cam_gt.kmtx.fy = 1005;
    cam_gt.kmtx.cx = 640;
    cam_gt.kmtx.cy = 360;
    cam_gt.kmtx.skew = 0.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Use proper transforms: camera at origin looking at target in front
    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();  // gripper to camera
    Eigen::Isometry3d b_se3_t = Eigen::Translation3d(0.0, 0.0, 2.0) *
                                Eigen::Isometry3d::Identity();  // base to target (target 2m away)
    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    views.reserve(sim.observations.size());
    for (const auto& ob : sim.observations) {
        views.push_back(ob.view);
    }

    // Create initial camera guess
    Camera<BrownConradyd> guess_cam = cam_gt;
    guess_cam.kmtx.fx *= 0.97;
    guess_cam.kmtx.fy *= 1.03;
    guess_cam.kmtx.cx += 5.0;
    guess_cam.kmtx.cy -= 4.0;
    guess_cam.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Estimate initial poses for each view
    std::vector<Eigen::Isometry3d> init_poses;
    init_poses.reserve(views.size());
    for (const auto& view : views) {
        auto pose = estimate_planar_pose(view, guess_cam.kmtx);
        init_poses.push_back(pose);
    }

    IntrinsicsOptions opts;
    opts.num_radial = 3;
    opts.optimize_skew = false;
    auto res = optimize_intrinsics(views, guess_cam, init_poses, opts);

    const auto& k_final = res.camera.kmtx;
    const auto& k_gt = cam_gt.kmtx;
    EXPECT_NEAR(k_final.fx, k_gt.fx, 1e-6);
    EXPECT_NEAR(k_final.fy, k_gt.fy, 1e-6);
    EXPECT_NEAR(k_final.cx, k_gt.cx, 1e-6);
    EXPECT_NEAR(k_final.cy, k_gt.cy, 1e-6);
    EXPECT_NEAR(k_final.skew, k_gt.skew, 1e-9);
    EXPECT_LT(res.final_cost, 1e-6);
}

TEST(OptimizeIntrinsics, RecoversSkew) {
    RNG rng(5);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 1000;
    cam_gt.kmtx.fy = 1005;
    cam_gt.kmtx.cx = 640;
    cam_gt.kmtx.cy = 360;
    cam_gt.kmtx.skew = 0.001;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Use proper transforms: camera at origin looking at target in front
    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();  // gripper to camera
    Eigen::Isometry3d b_se3_t = Eigen::Translation3d(0.0, 0.0, 2.0) *
                                Eigen::Isometry3d::Identity();  // base to target (target 2m away)
    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(15, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels();

    std::vector<PlanarView> views;
    views.reserve(sim.observations.size());
    for (const auto& ob : sim.observations) {
        views.push_back(ob.view);
    }

    // Create initial camera guess
    Camera<BrownConradyd> guess_cam = cam_gt;
    guess_cam.kmtx.fx *= 0.95;
    guess_cam.kmtx.fy *= 1.05;
    guess_cam.kmtx.cx += 10.0;
    guess_cam.kmtx.cy -= 6.0;
    guess_cam.kmtx.skew = 0.0;
    guess_cam.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Estimate initial poses for each view
    std::vector<Eigen::Isometry3d> init_poses;
    init_poses.reserve(views.size());
    for (const auto& view : views) {
        auto pose = estimate_planar_pose(view, guess_cam.kmtx);
        init_poses.push_back(pose);
    }

    IntrinsicsOptions opts;
    opts.num_radial = 0;
    opts.optimize_skew = true;
    auto res = optimize_intrinsics(views, guess_cam, init_poses, opts);

    const auto& k_final = res.camera.kmtx;
    const auto& k_gt = cam_gt.kmtx;
    EXPECT_NEAR(k_final.fx, k_gt.fx, 1e-6);
    EXPECT_NEAR(k_final.fy, k_gt.fy, 1e-6);
    EXPECT_NEAR(k_final.cx, k_gt.cx, 1e-6);
    EXPECT_NEAR(k_final.cy, k_gt.cy, 1e-6);
    EXPECT_NEAR(k_final.skew, k_gt.skew, 1e-8);
}
