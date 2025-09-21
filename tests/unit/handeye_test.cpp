// test_hand_eye.cpp
#include <gtest/gtest.h>

// std
#include <iostream>

#include "calib/estimation/handeye.h"
#include "utils.h"

using namespace calib;
// ---------- TESTS ----------

TEST(TsaiLenzAllPairsWeighted, RecoversGroundTruthWithNoise) {
    RNG rng(123);
    // Ground truth X (hand-eye)
    Eigen::Vector3d axX = rng.rand_unit_axis();
    double angX = deg2rad(12.0);
    Eigen::Isometry3d X_gt = make_pose(Eigen::Vector3d(0.03, -0.02, 0.10), axX, angX);

    // Ground truth ^bT_t
    Eigen::Vector3d axT = rng.rand_unit_axis();
    double angT = deg2rad(20.0);
    Eigen::Isometry3d b_se3_t_gt = make_pose(Eigen::Vector3d(0.40, 0.10, 0.60), axT, angT);

    // Intrinsics (not used by linear step, but for later)
    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 900;
    cam_gt.kmtx.fy = 920;
    cam_gt.kmtx.cx = 640;
    cam_gt.kmtx.cy = 360;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Sim data
    SimulatedHandEye sim{X_gt, b_se3_t_gt, cam_gt};
    sim.make_sequence(/*n_frames*/ 20, rng);
    sim.make_target_grid(7, 10, 0.02);  // 7x10 grid, 20 mm spacing
    sim.render_pixels(/*noise_px*/ 0.2, &rng);

    // Build camera_se3_target from sim.c_se3_t
    const auto base_se3_gripper = sim.b_se3_g();
    const auto& camera_se3_target = sim.c_se3_t;

    // Estimate with all-pairs weighted Tsai-Lenz
    Eigen::Isometry3d X_est =
        estimate_handeye_dlt(base_se3_gripper, camera_se3_target, /*min_angle_deg*/ 1.0);

    double rot_err = rad2deg(rotation_angle(X_est.linear().transpose() * X_gt.linear()));
    double trans_err = (X_est.translation() - X_gt.translation()).norm();

    EXPECT_LT(rot_err, 10);       // ~10 deg. TODO: is it too large?
    EXPECT_LT(trans_err, 0.005);  // ~5 mm
}

TEST(TsaiLenzAllPairsWeighted, ThrowsOnDegenerateSmallMotions) {
    // All poses identical -> no valid pairs
    std::vector<Eigen::Isometry3d> b_se3_g(5, Eigen::Isometry3d::Identity());
    std::vector<Eigen::Isometry3d> c_se3_t(5, Eigen::Isometry3d::Identity());
    EXPECT_THROW(
        { estimate_handeye_dlt(b_se3_g, c_se3_t, /*min_angle_deg*/ 2.0); }, std::runtime_error);
}

TEST(TsaiLenzAllPairsWeighted, InvariantToBaseFrameLeftMultiply) {
    RNG rng(77);
    // Make simple ground truth and sequence
    Eigen::Isometry3d X_gt =
        make_pose(Eigen::Vector3d(0.01, 0.02, 0.12), Eigen::Vector3d(0, 0, 1), deg2rad(15));
    Eigen::Isometry3d b_se3_t_gt =
        make_pose(Eigen::Vector3d(0.3, 0.2, 0.7), Eigen::Vector3d(1, 0, 0), deg2rad(10));
    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 1000;
    cam_gt.kmtx.fy = 1000;
    cam_gt.kmtx.cx = 640;
    cam_gt.kmtx.cy = 360;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    SimulatedHandEye sim{X_gt, b_se3_t_gt, cam_gt};
    sim.make_sequence(12, rng);
    sim.make_target_grid(6, 8, 0.03);
    sim.render_pixels(0.0, nullptr);

    const auto base_se3_gripper = sim.b_se3_g();
    const auto& camera_se3_target = sim.c_se3_t;

    // Left-multiply base poses by a fixed transform
    Eigen::Isometry3d B = make_pose(Eigen::Vector3d(0.5, -0.1, 0.2),
                                    Eigen::Vector3d(0.3, 0.7, 0.2).normalized(), deg2rad(25));
    std::vector<Eigen::Isometry3d> base_se3_gripper2 = base_se3_gripper;
    for (auto& T : base_se3_gripper2) T = B * T;

    Eigen::Isometry3d X1 = estimate_handeye_dlt(base_se3_gripper, camera_se3_target, 1.0);
    Eigen::Isometry3d X2 = estimate_handeye_dlt(base_se3_gripper2, camera_se3_target, 1.0);

    double rot_err = rad2deg(rotation_angle(X1.linear().transpose() * X2.linear()));
    double trans_err = (X1.translation() - X2.translation()).norm();

    EXPECT_LT(rot_err, 1e-6);
    EXPECT_LT(trans_err, 1e-9);
}

TEST(CeresAXXBRefine, ImprovesOverInitializer) {
    RNG rng(2024);
    // Ground truth
    Eigen::Isometry3d X_gt =
        make_pose(Eigen::Vector3d(0.02, -0.01, 0.09), rng.rand_unit_axis(), deg2rad(10.0));
    Eigen::Isometry3d b_se3_t_gt =
        make_pose(Eigen::Vector3d(0.25, 0.05, 0.55), rng.rand_unit_axis(), deg2rad(18.0));
    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 950;
    cam_gt.kmtx.fy = 960;
    cam_gt.kmtx.cx = 640;
    cam_gt.kmtx.cy = 360;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    // Data
    SimulatedHandEye sim{X_gt, b_se3_t_gt, cam_gt};
    sim.make_sequence(18, rng);
    sim.make_target_grid(6, 9, 0.025);
    sim.render_pixels(0.0, nullptr);
    const auto base_se3_gripper = sim.b_se3_g();
    const auto& camera_se3_target = sim.c_se3_t;

    // Initializer: perturb X
    Eigen::Isometry3d X0 = X_gt;
    {
        Eigen::Vector3d ax = rng.rand_unit_axis();
        double dang = deg2rad(2.0);
        X0.linear() = axis_angle_to_R(ax, dang) * X0.linear();
        X0.translation() += Eigen::Vector3d(0.01, -0.005, 0.004);  // ~ centimeter
    }

    double err0_rot = rad2deg(rotation_angle(X0.linear().transpose() * X_gt.linear()));
    double err0_tr = (X0.translation() - X_gt.translation()).norm();

    HandeyeOptions ro;
    ro.optimizer = OptimizerType::DENSE_QR;
    ro.max_iterations = 60;
    ro.huber_delta = 1.0;
    ro.verbose = false;

    auto res = optimize_handeye(base_se3_gripper, camera_se3_target, X0, ro);
    Eigen::Isometry3d Xr = res.g_se3_c;

    double err1_rot = rad2deg(rotation_angle(Xr.linear().transpose() * X_gt.linear()));
    double err1_tr = (Xr.translation() - X_gt.translation()).norm();

    // Should improve both
    EXPECT_LT(err1_rot, err0_rot);
    EXPECT_LT(err1_tr, err0_tr);
    EXPECT_LT(err1_rot, 0.05);  // ~0.05 deg
    EXPECT_LT(err1_tr, 0.002);  // ~2 mm
}
