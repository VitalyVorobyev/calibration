#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "calib/jointintrextr.h"

using namespace calib;

TEST(JointCalibration, RecoverAllParameters) {
    const int kCams = 2;
    CameraMatrix K{100.0, 100.0, 0.0, 0.0};
    Eigen::VectorXd dist(2);
    dist << 0.0, 0.0;
    DualDistortion dd; dd.forward = dist; dd.inverse = dist;
    std::vector<Camera<DualDistortion>> cameras_gt = {Camera<DualDistortion>{K, dd}, Camera<DualDistortion>{K, dd}};

    Eigen::Affine3d cam0 = Eigen::Affine3d::Identity();
    Eigen::Affine3d cam1 = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Affine3d::Identity();
    std::vector<Eigen::Affine3d> cam_gt = {cam0, cam1};

    std::vector<Eigen::Affine3d> target_gt = {
        Eigen::Translation3d(0.0, 0.0, 5.0) * Eigen::Affine3d::Identity(),
        Eigen::Translation3d(0.5, -0.2, 4.0) * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()),
    };

    // need at least 8 points to fit distortions
    std::vector<Eigen::Vector2d> points = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
        {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5}
    };

    std::vector<ExtrinsicPlanarView> views;
    for (size_t v = 0; v < target_gt.size(); ++v) {
        ExtrinsicPlanarView view;
        view.resize(kCams);
        for (int c = 0; c < kCams; ++c) {
            Eigen::Affine3d T = cam_gt[c] * target_gt[v];
            for (const auto& xy : points) {
                Eigen::Vector3d P = T * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
                Eigen::Vector2d norm(P.x()/P.z(), P.y()/P.z());
                Eigen::Vector2d pix = cameras_gt[c].K.denormalize(norm);
                view[c].push_back({xy, pix});
            }
        }
        views.push_back(std::move(view));
    }

    // Perturbed intrinsics for initialization
    std::vector<Camera<DualDistortion>> cam_init = {
        Camera<DualDistortion>{CameraMatrix{90.0, 95.0, 1.0, -1.0}, dd},
        Camera<DualDistortion>{CameraMatrix{105.0, 98.0, -0.5, 0.5}, dd}
    };

    auto guess = estimate_extrinsic_dlt(views, cam_init);
    ASSERT_TRUE(guess.camera_poses.front().isApprox(Eigen::Affine3d::Identity()));

    // Anchor the first target pose to its ground truth to fix the scale.
    guess.target_poses[0] = target_gt[0];

    JointOptions opts; opts.verbose = false;
    auto res = optimize_joint_intrinsics_extrinsics(views, cam_init, guess.camera_poses, guess.target_poses, opts);
    std::cout << res.summary << std::endl;

    EXPECT_LT(res.reprojection_error, 1e-6);
    ASSERT_EQ(res.intrinsics.size(), static_cast<size_t>(kCams));
    EXPECT_NEAR(res.intrinsics[0].fx, 100.0, 1e-3);
    EXPECT_NEAR(res.intrinsics[0].fy, 100.0, 1e-3);
    EXPECT_TRUE(res.camera_poses[1].translation().isApprox(cam_gt[1].translation(), 1e-3));
    EXPECT_TRUE(res.target_poses[0].translation().isApprox(target_gt[0].translation(), 1e-3));
    ASSERT_EQ(res.intrinsic_covariances.size(), static_cast<size_t>(kCams));
    EXPECT_GT(res.intrinsic_covariances[0].trace(), 0.0);
}

TEST(JointCalibration, FirstTargetPoseFixed) {
    const int kCams = 2;
    CameraMatrix K{100.0, 100.0, 0.0, 0.0};
    Eigen::VectorXd dist(2);
    dist << 0.0, 0.0;
    DualDistortion dd; dd.forward = dist; dd.inverse = dist;
    std::vector<Camera<DualDistortion>> cameras_gt = {Camera<DualDistortion>{K, dd}, Camera<DualDistortion>{K, dd}};

    Eigen::Affine3d cam0 = Eigen::Affine3d::Identity();
    Eigen::Affine3d cam1 = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Affine3d::Identity();

    std::vector<Eigen::Affine3d> target_gt = {
        Eigen::Translation3d(0.0, 0.0, 5.0) * Eigen::Affine3d::Identity(),
        Eigen::Translation3d(0.5, -0.2, 4.0) * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()),
    };

    std::vector<Eigen::Vector2d> points = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
        {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5}
    };

    std::vector<ExtrinsicPlanarView> views;
    for (size_t v = 0; v < target_gt.size(); ++v) {
        ExtrinsicPlanarView view;
        view.resize(kCams);
        for (int c = 0; c < kCams; ++c) {
            Eigen::Affine3d T = (c==0?cam0:cam1) * target_gt[v];
            for (const auto& xy : points) {
                Eigen::Vector3d P = T * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
                Eigen::Vector2d norm(P.x()/P.z(), P.y()/P.z());
                Eigen::Vector2d pix = cameras_gt[c].K.denormalize(norm);
                view[c].push_back({xy, pix});
            }
        }
        views.push_back(std::move(view));
    }

    std::vector<Camera<DualDistortion>> cam_init = {
        Camera<DualDistortion>{CameraMatrix{90.0, 95.0, 1.0, -1.0}, dd},
        Camera<DualDistortion>{CameraMatrix{105.0, 98.0, -0.5, 0.5}, dd}
    };

    auto guess = estimate_extrinsic_dlt(views, cam_init);
    // Deliberately set an incorrect scale for the first target pose. This pose
    // should remain unchanged by the optimisation.
    guess.target_poses[0].translation() = Eigen::Vector3d(0.0, 0.0, 3.0);

    JointOptions opts; opts.verbose = false;
    auto res = optimize_joint_intrinsics_extrinsics(views, cam_init, guess.camera_poses, guess.target_poses, opts);

    EXPECT_TRUE(res.target_poses[0].translation().isApprox(guess.target_poses[0].translation(), 1e-12));
    EXPECT_GT(res.reprojection_error, 0.1);
}
