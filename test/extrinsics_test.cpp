#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "calib/extrinsics.h"

using namespace calib;

TEST(Extrinsics, RecoverCameraAndTargetPoses) {
    const int kCams = 2;
    CameraMatrix K{100.0, 100.0, 0.0, 0.0};
    Eigen::VectorXd dist(2);
    dist << 0.0, 0.0; // no distortion
    DualDistortion dd; dd.forward = dist; dd.inverse = dist;
    std::vector<Camera<DualDistortion>> cameras = {Camera<DualDistortion>{K, dd}, Camera<DualDistortion>{K, dd}};

    // Ground-truth camera poses (reference camera is identity)
    Eigen::Affine3d cam0 = Eigen::Affine3d::Identity();
    Eigen::Affine3d cam1 = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Affine3d::Identity();
    std::vector<Eigen::Affine3d> cam_gt = {cam0, cam1};

    // Initial guesses (perturbed)
    std::vector<Eigen::Affine3d> cam_init = {
        cam0,
        Eigen::Translation3d(1.2, -0.1, 0.05) * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ())
    };

    // Target poses for three views
    std::vector<Eigen::Affine3d> target_gt = {
        Eigen::Translation3d(0.0, 0.0, 5.0) * Eigen::Affine3d::Identity(),
        Eigen::Translation3d(0.5, -0.2, 4.0) * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()),
        Eigen::Translation3d(-0.3, 0.4, 6.0) * Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX())
    };

    std::vector<Eigen::Affine3d> target_init = {
        target_gt[0] * Eigen::Translation3d(0.1, 0.0, 0.0) * Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitZ()),
        target_gt[1] * Eigen::Translation3d(-0.05, 0.1, 0.05) * Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitY()),
        target_gt[2] * Eigen::Translation3d(0.02, -0.02, -0.1) * Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX())
    };

    // Planar points on target
    std::vector<Eigen::Vector2d> points = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
    };

    std::vector<ExtrinsicPlanarView> views;
    for (size_t v = 0; v < target_gt.size(); ++v) {
        ExtrinsicPlanarView view;
        view.resize(kCams);
        for (int c = 0; c < kCams; ++c) {
            Eigen::Affine3d T = cam_gt[c] * target_gt[v]; // target -> camera
            for (const auto& xy : points) {
                Eigen::Vector3d P = T * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
                Eigen::Vector2d norm(P.x() / P.z(), P.y() / P.z());
                Eigen::Vector2d pix = cameras[c].K.denormalize(norm);
                view[c].push_back({xy, pix});
            }
        }
        views.push_back(std::move(view));
    }

    auto result = optimize_extrinsic_poses(views, cameras, cam_init, target_init);

    EXPECT_LT(result.reprojection_error, 1e-6);
    ASSERT_EQ(result.camera_poses.size(), static_cast<size_t>(kCams));
    ASSERT_EQ(result.camera_covariances.size(), static_cast<size_t>(kCams));
    ASSERT_EQ(result.target_covariances.size(), target_gt.size());
    EXPECT_GT(result.target_covariances[0].trace(), 0.0);
    EXPECT_TRUE(result.camera_poses[1].translation().isApprox(cam_gt[1].translation(), 1e-3));
    EXPECT_TRUE(result.camera_poses[1].linear().isApprox(cam_gt[1].linear(), 1e-3));
    for (size_t v = 0; v < target_gt.size(); ++v) {
        EXPECT_TRUE(result.target_poses[v].translation().isApprox(target_gt[v].translation(), 1e-3));
        EXPECT_TRUE(result.target_poses[v].linear().isApprox(target_gt[v].linear(), 1e-3));
    }
}

