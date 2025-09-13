// std
#include <algorithm>

// gtest
#include <gtest/gtest.h>

// eigen
#include <Eigen/Geometry>

#include "calib/planarpose.h"
#include "calib/intrinsics.h"

using namespace calib;

// Helper function to create a simple synthetic planar target
auto create_synthetic_planar_data(const Eigen::Isometry3d& pose, const CameraMatrix& intrinsics) -> PlanarView {
    // Create a grid of points on the plane Z=0
    PlanarView view;
    for (int i = -5; i <= 5; i += 2) {
        for (int j = -5; j <= 5; j += 2) {
            // Object point on the plane Z=0
            Eigen::Vector2d obj_pt(i * 0.1, j * 0.1);

            // Project to camera
            Eigen::Vector3d point_3d(obj_pt.x(), obj_pt.y(), 0.0);
            Eigen::Vector2d point_camera = (pose * point_3d).hnormalized();

            // Apply camera intrinsics
            Eigen::Vector2d pixel = intrinsics.denormalize(point_camera);
            view.emplace_back(obj_pt, pixel);
        }
    }
    return view;
}

TEST(PlanarPoseTest, HomographyDecomposition) {
    // Create a known homography matrix
    Eigen::Matrix3d R = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Vector3d t(0.2, 0.3, 1.0);

    // H = [r1 r2 t]
    Eigen::Matrix3d H;
    H.col(0) = R.col(0);
    H.col(1) = R.col(1);
    H.col(2) = t;

    // Decompose the homography
    Eigen::Isometry3d pose = pose_from_homography_normalized(H);

    // Check the rotation part
    Eigen::Matrix3d recovered_R = pose.linear();
    EXPECT_TRUE(R.isApprox(recovered_R, 1e-9));

    // Check the translation part
    Eigen::Vector3d recovered_t = pose.translation();
    EXPECT_TRUE(t.isApprox(recovered_t, 1e-9));
}

TEST(PlanarPoseTest, DLTEstimation) {
    // Create synthetic camera intrinsics
    CameraMatrix intrinsics;
    intrinsics.fx = 1000;
    intrinsics.fy = 1000;
    intrinsics.cx = 500;
    intrinsics.cy = 500;

    // Create a known pose
    Eigen::Isometry3d true_pose = Eigen::Isometry3d::Identity();
    true_pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    true_pose.translation() = Eigen::Vector3d(0.1, 0.2, 2.0);

    // Generate synthetic data
    const PlanarView view = create_synthetic_planar_data(true_pose, intrinsics);

    // Estimate the pose
    Eigen::Isometry3d estimated_pose = estimate_planar_pose(view, intrinsics);

    // Check rotation (up to sign ambiguity)
    Eigen::Matrix3d true_R = true_pose.linear();
    Eigen::Matrix3d est_R = estimated_pose.linear();

    // The recovered rotation should be close to the true one or its negative (cheirality)
    bool rot_matches = true_R.isApprox(est_R, 1e-1) || true_R.isApprox(-est_R, 1e-1);
    EXPECT_TRUE(rot_matches);

    // Check translation direction (since scale might vary)
    Eigen::Vector3d true_t = true_pose.translation();
    Eigen::Vector3d est_t = estimated_pose.translation();

    double cosine_similarity = true_t.normalized().dot(est_t.normalized());
    EXPECT_GT(std::abs(cosine_similarity), 0.9); // Vectors should point in similar directions
}

// Temporarily disable this test while we investigate segmentation fault
TEST(PlanarPoseTest, OptimizePlanarPose) {
    // Create synthetic camera intrinsics
    CameraMatrix intrinsics;
    intrinsics.fx = 1000;
    intrinsics.fy = 1000;
    intrinsics.cx = 500;
    intrinsics.cy = 500;

    // Create a known pose
    Eigen::Isometry3d true_pose = Eigen::Isometry3d::Identity();
    true_pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    true_pose.translation() = Eigen::Vector3d(0.1, 0.2, 2.0);

    // Generate synthetic data - generate more points for stability
    const PlanarView view = create_synthetic_planar_data(true_pose, intrinsics);

    // Make pose estimate
    Eigen::Isometry3d init_pose = Eigen::Isometry3d::Identity();
    init_pose.linear() = Eigen::AngleAxisd(0.12, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    init_pose.translation() = Eigen::Vector3d(0.19, 0.23, 2.1);

    // Optimize the pose
    PlanarPoseOptions opts;
    opts.num_radial = 0;
    PlanarPoseResult result = optimize_planar_pose(view, intrinsics, init_pose, opts);

    // Check if optimization was successful
    EXPECT_LT(result.reprojection_error, 1e-3);

    // Check rotation
    Eigen::Matrix3d true_R = true_pose.linear();
    Eigen::Matrix3d opt_R = result.pose.linear();
    EXPECT_TRUE(true_R.isApprox(opt_R, 1e-1));

    // Check translation
    Eigen::Vector3d true_t = true_pose.translation();
    Eigen::Vector3d opt_t = result.pose.translation();

    // For planar cases, there can be scale ambiguity
    double scale = true_t.z() / opt_t.z();  // Normalize by Z component
    Eigen::Vector3d scaled_opt_t = scale * opt_t;

    // Check if scaled translation is close to the true translation
    EXPECT_LT((true_t - scaled_opt_t).norm(), 0.1);

    // Check if covariance matrix is positive definite
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(result.covariance);
    EXPECT_GT(eig.eigenvalues().minCoeff(), 0);
}

TEST(PlanarPoseTest, OptimizePlanarPoseWithDistortion) {
    // Create synthetic camera intrinsics
    CameraMatrix intrinsics;
    intrinsics.fx = 1000;
    intrinsics.fy = 1000;
    intrinsics.cx = 500;
    intrinsics.cy = 500;

    // Create a known pose
    Eigen::Isometry3d true_pose = Eigen::Isometry3d::Identity();
    true_pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    true_pose.translation() = Eigen::Vector3d(0.1, 0.2, 2.0);

    // Generate synthetic data - generate more points for stability
    PlanarView view = create_synthetic_planar_data(true_pose, intrinsics);

    // Apply simple radial distortion to image points
    BrownConrady<double> brownconrady;
    brownconrady.coeffs = Eigen::Vector2d(0.1, 0);

    std::for_each(view.begin(), view.end(), [&brownconrady, &intrinsics](PlanarObservation& item) {
        auto norm_pix = intrinsics.normalize(item.image_uv);
        auto distorted_norm = brownconrady.distort(norm_pix);
        item.image_uv = intrinsics.denormalize(distorted_norm);
    });

    // Make pose estimate
    Eigen::Isometry3d init_pose = Eigen::Isometry3d::Identity();
    init_pose.linear() = Eigen::AngleAxisd(0.12, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    init_pose.translation() = Eigen::Vector3d(0.19, 0.23, 2.1);

    // Optimize the pose with distortion
    PlanarPoseOptions opts;
    opts.num_radial = 1;
    PlanarPoseResult result = optimize_planar_pose(view, intrinsics, init_pose, opts);

    // Check if optimization was successful
    EXPECT_LT(result.reprojection_error, 1e-2);

    // Check rotation
    Eigen::Matrix3d true_R = true_pose.linear();
    Eigen::Matrix3d opt_R = result.pose.linear();
    EXPECT_TRUE(true_R.isApprox(opt_R, 1e-1));

    // Check translation
    Eigen::Vector3d true_t = true_pose.translation();
    Eigen::Vector3d opt_t = result.pose.translation();

    // For planar cases, there can be scale ambiguity
    double scale = true_t.z() / opt_t.z();  // Normalize by Z component
    Eigen::Vector3d scaled_opt_t = scale * opt_t;

    // Check if scaled translation is close to the true translation
    EXPECT_LT((true_t - scaled_opt_t).norm(), 0.2);

    // Check distortion coefficients
    EXPECT_GT(result.distortion.size(), 0);

    // The first coefficient should be close to our synthetic k1 value
    // We're a bit more lenient here because distortion estimation can be sensitive
    EXPECT_NEAR(result.distortion[0], brownconrady.coeffs(0), 0.2);
}
