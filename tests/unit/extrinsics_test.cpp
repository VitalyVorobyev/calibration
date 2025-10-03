#include "calib/estimation/optim/extrinsics.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

using namespace calib;

TEST(Extrinsics, RecoverCameraAndTargetPoses) {
    const int kCams = 2;
    CameraMatrix kmtx{100.0, 100.0, 0.0, 0.0};
    Eigen::VectorXd dist(5);
    dist << 0.0, 0.0, 0.0, 0.0, 0.0;  // no distortion
    std::vector<Camera<BrownConradyd>> cameras = {Camera<BrownConradyd>{kmtx, dist},
                                                  Camera<BrownConradyd>{kmtx, dist}};

    // Ground-truth camera poses (reference camera is identity)
    Eigen::Isometry3d cam0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d cam1 = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Isometry3d::Identity();
    std::vector<Eigen::Isometry3d> cam_gt = {cam0, cam1};

    // Initial guesses (perturbed)
    std::vector<Eigen::Isometry3d> cam_init = {
        cam0,
        Eigen::Translation3d(1.2, -0.1, 0.05) * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ())};

    // Target poses for three views
    std::vector<Eigen::Isometry3d> target_gt = {
        Eigen::Translation3d(0.0, 0.0, 5.0) * Eigen::Isometry3d::Identity(),
        Eigen::Translation3d(0.5, -0.2, 4.0) * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()),
        Eigen::Translation3d(-0.3, 0.4, 6.0) * Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX())};

    std::vector<Eigen::Isometry3d> target_init = {
        target_gt[0] * Eigen::Translation3d(0.1, 0.0, 0.0) *
            Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()),
        target_gt[1] * Eigen::Translation3d(-0.05, 0.1, 0.05) *
            Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitY()),
        target_gt[2] * Eigen::Translation3d(0.02, -0.02, -0.1) *
            Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX())};

    // Planar points on target
    std::vector<Eigen::Vector2d> points = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

    std::vector<MulticamPlanarView> views;
    for (size_t v = 0; v < target_gt.size(); ++v) {
        MulticamPlanarView view;
        view.resize(kCams);
        for (int c = 0; c < kCams; ++c) {
            Eigen::Isometry3d T = cam_gt[c] * target_gt[v];  // target -> camera
            for (const auto& xy : points) {
                Eigen::Vector3d P = T * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
                Eigen::Vector2d norm(P.x() / P.z(), P.y() / P.z());
                Eigen::Vector2d pix = denormalize(cameras[c].kmtx, norm);
                view[c].push_back({xy, pix});
            }
        }
        views.push_back(std::move(view));
    }

    ExtrinsicOptions opts;
    opts.optimize_intrinsics = false;
    auto result = optimize_extrinsics(views, cameras, cam_init, target_init, opts);

    EXPECT_LT(result.final_cost, 1e-6);
    ASSERT_EQ(result.c_se3_r.size(), static_cast<size_t>(kCams));
    ASSERT_EQ(result.r_se3_t.size(), target_gt.size());
    EXPECT_TRUE(result.c_se3_r[1].translation().isApprox(cam_gt[1].translation(), 1e-3));
    EXPECT_TRUE(result.c_se3_r[1].linear().isApprox(cam_gt[1].linear(), 1e-3));
    for (size_t v = 0; v < target_gt.size(); ++v) {
        EXPECT_TRUE(result.r_se3_t[v].translation().isApprox(target_gt[v].translation(), 1e-3));
        EXPECT_TRUE(result.r_se3_t[v].linear().isApprox(target_gt[v].linear(), 1e-3));
    }
}

TEST(Extrinsics, RecoverAllParameters) {
    const int kCams = 2;
    CameraMatrix kmtx{100.0, 100.0, 0.0, 0.0};
    Eigen::VectorXd dist(5);
    dist << 0.0, 0.0, 0.0, 0.0, 0.0;  // no distortion
    std::vector<Camera<BrownConradyd>> cameras_gt = {Camera<BrownConradyd>{kmtx, dist},
                                                     Camera<BrownConradyd>{kmtx, dist}};

    Eigen::Isometry3d cam0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d cam1 = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Isometry3d::Identity();
    std::vector<Eigen::Isometry3d> cam_gt = {cam0, cam1};

    std::vector<Eigen::Isometry3d> target_gt = {
        Eigen::Translation3d(0.0, 0.0, 5.0) * Eigen::Isometry3d::Identity(),
        Eigen::Translation3d(0.5, -0.2, 4.0) * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()),
    };

    // need at least 8 points to fit distortions
    std::vector<Eigen::Vector2d> points = {{0.0, 0.0}, {1.0, 0.0},   {1.0, 1.0}, {0.0, 1.0},
                                           {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5}};

    std::vector<MulticamPlanarView> views;
    for (size_t v = 0; v < target_gt.size(); ++v) {
        MulticamPlanarView view;
        view.resize(kCams);
        for (int c = 0; c < kCams; ++c) {
            Eigen::Isometry3d T = cam_gt[c] * target_gt[v];
            for (const auto& xy : points) {
                Eigen::Vector3d P = T * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
                Eigen::Vector2d norm(P.x() / P.z(), P.y() / P.z());
                Eigen::Vector2d pix = denormalize(cameras_gt[c].kmtx, norm);
                view[c].push_back({xy, pix});
            }
        }
        views.push_back(std::move(view));
    }

    // Perturbed intrinsics for initialization
    std::vector<Camera<BrownConradyd>> cam_init = {
        Camera<BrownConradyd>{CameraMatrix{90.0, 95.0, 1.0, -1.0}, Eigen::VectorXd::Zero(5)},
        Camera<BrownConradyd>{CameraMatrix{105.0, 98.0, -0.5, 0.5}, Eigen::VectorXd::Zero(5)}};

    // Change cameras to BrownConradyd for the estimate function
    std::vector<Camera<DualDistortion>> cameras_for_estimate = {
        Camera<DualDistortion>{kmtx, DualDistortion{Eigen::VectorXd::Zero(2)}},
        Camera<DualDistortion>{kmtx, DualDistortion{Eigen::VectorXd::Zero(2)}}};

    auto guess = estimate_extrinsic_dlt(views, cameras_for_estimate);
    ASSERT_TRUE(guess.c_se3_r.front().isApprox(Eigen::Isometry3d::Identity()));

    // Anchor the first target pose to its ground truth to fix the scale.
    guess.r_se3_t[0] = target_gt[0];

    ExtrinsicOptions opts;
    opts.verbose = false;
    auto res = optimize_extrinsics(views, cam_init, guess.c_se3_r, guess.r_se3_t, opts);
    std::cout << res.report << std::endl;

    EXPECT_LT(res.final_cost, 1e-6);
    ASSERT_EQ(res.cameras.size(), static_cast<size_t>(kCams));
    EXPECT_NEAR(res.cameras[0].kmtx.fx, 100.0, 1e-3);
    EXPECT_NEAR(res.cameras[0].kmtx.fy, 100.0, 1e-3);
    EXPECT_TRUE(res.c_se3_r[1].translation().isApprox(cam_gt[1].translation(), 1e-3));
    EXPECT_TRUE(res.r_se3_t[0].translation().isApprox(target_gt[0].translation(), 1e-3));
    EXPECT_GT(res.covariance.trace(), 0.0);
}

TEST(Extrinsics, FirstTargetPoseFixed) {
    const int kCams = 2;
    CameraMatrix kmtx{100.0, 100.0, 0.0, 0.0};
    Eigen::VectorXd dist(5);
    dist << 0.0, 0.0, 0.0, 0.0, 0.0;  // no distortion
    std::vector<Camera<BrownConradyd>> cameras_gt = {Camera<BrownConradyd>{kmtx, dist},
                                                     Camera<BrownConradyd>{kmtx, dist}};

    Eigen::Isometry3d cam0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d cam1 = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Isometry3d::Identity();

    std::vector<Eigen::Isometry3d> target_gt = {
        Eigen::Translation3d(0.0, 0.0, 5.0) * Eigen::Isometry3d::Identity(),
        Eigen::Translation3d(0.5, -0.2, 4.0) * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()),
    };

    std::vector<Eigen::Vector2d> points = {{0.0, 0.0}, {1.0, 0.0},   {1.0, 1.0}, {0.0, 1.0},
                                           {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5}};

    std::vector<MulticamPlanarView> views;
    for (size_t v = 0; v < target_gt.size(); ++v) {
        MulticamPlanarView view;
        view.resize(kCams);
        for (int c = 0; c < kCams; ++c) {
            Eigen::Isometry3d T = (c == 0 ? cam0 : cam1) * target_gt[v];
            for (const auto& xy : points) {
                Eigen::Vector3d P = T * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
                Eigen::Vector2d norm(P.x() / P.z(), P.y() / P.z());
                Eigen::Vector2d pix = denormalize(cameras_gt[c].kmtx, norm);
                view[c].push_back({xy, pix});
            }
        }
        views.push_back(std::move(view));
    }

    std::vector<Camera<BrownConradyd>> cam_init = {
        Camera<BrownConradyd>{CameraMatrix{90.0, 95.0, 1.0, -1.0}, Eigen::VectorXd::Zero(5)},
        Camera<BrownConradyd>{CameraMatrix{105.0, 98.0, -0.5, 0.5}, Eigen::VectorXd::Zero(5)}};

    // Change cameras to DualDistortion for the estimate function
    std::vector<Camera<DualDistortion>> cameras_for_estimate = {
        Camera<DualDistortion>{CameraMatrix{90.0, 95.0, 1.0, -1.0},
                               DualDistortion{Eigen::VectorXd::Zero(2)}},
        Camera<DualDistortion>{CameraMatrix{105.0, 98.0, -0.5, 0.5},
                               DualDistortion{Eigen::VectorXd::Zero(2)}}};

    auto guess = estimate_extrinsic_dlt(views, cameras_for_estimate);
    // Deliberately set an incorrect scale for the first target pose. This pose
    // should remain unchanged by the optimisation.
    guess.r_se3_t[0].translation() = Eigen::Vector3d(0.0, 0.0, 3.0);

    ExtrinsicOptions opts;
    opts.verbose = false;
    auto res = optimize_extrinsics(views, cam_init, guess.c_se3_r, guess.r_se3_t, opts);

    EXPECT_TRUE(res.r_se3_t[0].translation().isApprox(guess.r_se3_t[0].translation(), 1e-12));
    EXPECT_GT(res.final_cost, 0.1);
}
