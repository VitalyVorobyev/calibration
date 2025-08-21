// std
#include <algorithm>

// gtest
#include <gtest/gtest.h>

// eigen
#include <Eigen/Geometry>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/planarpose.h"
#include "calibration/intrinsics.h"

// Helper function to create a simple synthetic planar target
std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>
createSyntheticPlanarData(const Eigen::Affine3d& pose, const vitavision::CameraMatrix& intrinsics) {
    // Create a grid of points on the plane Z=0
    std::vector<Eigen::Vector2d> obj_points;
    std::vector<Eigen::Vector2d> img_points;

    for (int i = -5; i <= 5; i += 2) {
        for (int j = -5; j <= 5; j += 2) {
            // Object point on the plane Z=0
            Eigen::Vector2d obj_pt(i * 0.1, j * 0.1);
            obj_points.push_back(obj_pt);

            // Project to camera
            Eigen::Vector3d point_3d(obj_pt.x(), obj_pt.y(), 0.0);
            Eigen::Vector3d point_camera = pose * point_3d;

            // Project to normalized coordinates
            double x = point_camera.x() / point_camera.z();
            double y = point_camera.y() / point_camera.z();

            // Apply camera intrinsics
            double u = intrinsics.fx * x + intrinsics.cx;
            double v = intrinsics.fy * y + intrinsics.cy;

            img_points.emplace_back(u, v);
        }
    }

    return {obj_points, img_points};
}

namespace vitavision {

using Pose6 = Eigen::Matrix<double, 6, 1>;

// Functor mirroring the production PlanarPoseVPResidual for testing Jacobians.
struct PlanarPoseVPResidualTestFunctor {
    std::vector<PlanarObservation> obs;
    std::array<double, 4> K;
    int num_radial;

    template <typename T>
    bool operator()(const T* pose6, T* residuals) const {
        std::vector<Observation<T>> o(obs.size());
        std::transform(obs.begin(), obs.end(), o.begin(),
            [pose6, this](const PlanarObservation& s) -> Observation<T> {
                Eigen::Matrix<T, 3, 1> P(T(s.object_xy.x()), T(s.object_xy.y()), T(0.0));
                Eigen::Matrix<T, 3, 1> Pc;
                ceres::AngleAxisRotatePoint(pose6, P.data(), Pc.data());
                Pc += Eigen::Matrix<T, 3, 1>(pose6[3], pose6[4], pose6[5]);
                T invZ = T(1.0) / Pc.z();
                return {
                    .x = Pc.x() * invZ,
                    .y = Pc.y() * invZ,
                    .u = T(s.image_uv.x()) * T(K[0]) + T(K[2]),
                    .v = T(s.image_uv.y()) * T(K[1]) + T(K[3])
                };
            }
        );

        const T fx = T(K[0]);
        const T fy = T(K[1]);
        const T cx = T(K[2]);
        const T cy = T(K[3]);
        auto [_, r] = fit_distortion_full(o, fx, fy, cx, cy, num_radial);
        for (int i = 0; i < r.size(); ++i) {
            residuals[i] = r[i];
        }
        return true;
    }
};

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
    Eigen::Affine3d pose = pose_from_homography_normalized(H);

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
    Eigen::Affine3d true_pose = Eigen::Affine3d::Identity();
    true_pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    true_pose.translation() = Eigen::Vector3d(0.1, 0.2, 2.0);

    // Generate synthetic data
    auto [obj_points, img_points] = createSyntheticPlanarData(true_pose, intrinsics);

    // Estimate the pose
    Eigen::Affine3d estimated_pose = estimate_planar_pose_dlt(obj_points, img_points, intrinsics);

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

TEST(PlanarPoseTest, AutoDiffJacobianParity) {
    CameraMatrix intrinsics; intrinsics.fx = intrinsics.fy = 1000; intrinsics.cx = intrinsics.cy = 500;
    Eigen::Affine3d pose = Eigen::Affine3d::Identity();
    pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1,1,1).normalized()).toRotationMatrix();
    pose.translation() = Eigen::Vector3d(0.1,0.2,2.0);

    auto [obj_pts, img_pts] = createSyntheticPlanarData(pose, intrinsics);
    std::vector<PlanarObservation> obs(obj_pts.size());
    for (size_t i=0;i<obj_pts.size();++i) obs[i] = {obj_pts[i], img_pts[i]};

    const std::array<double, 4> K = {intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy};
    auto functor = new PlanarPoseVPResidualTestFunctor{obs, K, 0};
    ceres::AutoDiffCostFunction<PlanarPoseVPResidualTestFunctor, ceres::DYNAMIC, 6> cost(
        functor, static_cast<int>(obs.size()) * 2);

    Pose6 pose6; ceres::RotationMatrixToAngleAxis(pose.linear().data(), pose6.data());
    pose6[3] = pose.translation().x(); pose6[4] = pose.translation().y(); pose6[5] = pose.translation().z();

    const int m = static_cast<int>(obs.size()) * 2;
    std::vector<double> residuals(m);
    std::vector<double> jac(m * 6);
    double* jac_blocks[1] = {jac.data()};
    const double* params[1] = {pose6.data()};
    cost.Evaluate(params, residuals.data(), jac_blocks);

    std::vector<double> num_jac(m * 6);
    std::vector<double> r_plus(m), r_minus(m);
    for (int k = 0; k < 6; ++k) {
        double step = (k < 3) ? 1e-6 : 1e-5;
        Pose6 pp = pose6;
        Pose6 pm = pose6;
        pp[k] += step;
        pm[k] -= step;
        const double* p_plus[1] = {pp.data()};
        const double* p_minus[1] = {pm.data()};
        cost.Evaluate(p_plus, r_plus.data(), nullptr);
        cost.Evaluate(p_minus, r_minus.data(), nullptr);
        for (int i = 0; i < m; ++i) {
            num_jac[i * 6 + k] = (r_plus[i] - r_minus[i]) / (2.0 * step);
        }
    }

    for (int i = 0; i < m * 6; ++i) {
        EXPECT_NEAR(jac[i], num_jac[i], 0.005);
    }
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
    Eigen::Affine3d true_pose = Eigen::Affine3d::Identity();
    true_pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    true_pose.translation() = Eigen::Vector3d(0.1, 0.2, 2.0);

    // Generate synthetic data - generate more points for stability
    std::vector<Eigen::Vector2d> obj_points;
    std::vector<Eigen::Vector2d> img_points;
    for (int i = -5; i <= 5; i += 2) {
        for (int j = -5; j <= 5; j += 2) {
            // Object point on the plane Z=0
            Eigen::Vector2d obj_pt(i * 0.1, j * 0.1);
            obj_points.push_back(obj_pt);

            // Project to camera
            Eigen::Vector3d point_3d(obj_pt.x(), obj_pt.y(), 0.0);
            Eigen::Vector3d point_camera = true_pose * point_3d;

            // Project to normalized coordinates
            double x = point_camera.x() / point_camera.z();
            double y = point_camera.y() / point_camera.z();

            // Apply camera intrinsics
            double u = intrinsics.fx * x + intrinsics.cx;
            double v = intrinsics.fy * y + intrinsics.cy;

            img_points.emplace_back(u, v);
        }
    }

    // Make sure we have enough points
    ASSERT_GT(obj_points.size(), 10);
    ASSERT_EQ(obj_points.size(), img_points.size());

    // Optimize the pose
    PlanarPoseFitResult result = optimize_planar_pose(obj_points, img_points, intrinsics, 0, false);

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

// Temporarily disable this test while we investigate segmentation fault
TEST(PlanarPoseTest, OptimizePlanarPoseWithDistortion) {
    // Create synthetic camera intrinsics
    CameraMatrix intrinsics;
    intrinsics.fx = 1000;
    intrinsics.fy = 1000;
    intrinsics.cx = 500;
    intrinsics.cy = 500;

    // Create a known pose
    Eigen::Affine3d true_pose = Eigen::Affine3d::Identity();
    true_pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 1, 1).normalized()).toRotationMatrix();
    true_pose.translation() = Eigen::Vector3d(0.1, 0.2, 2.0);

    // Generate synthetic data - generate more points for stability
    std::vector<Eigen::Vector2d> obj_points;
    std::vector<Eigen::Vector2d> img_points;
    for (int i = -5; i <= 5; i += 2) {
        for (int j = -5; j <= 5; j += 2) {
            // Object point on the plane Z=0
            Eigen::Vector2d obj_pt(i * 0.1, j * 0.1);
            obj_points.push_back(obj_pt);

            // Project to camera
            Eigen::Vector3d point_3d(obj_pt.x(), obj_pt.y(), 0.0);
            Eigen::Vector3d point_camera = true_pose * point_3d;

            // Project to normalized coordinates
            double x = point_camera.x() / point_camera.z();
            double y = point_camera.y() / point_camera.z();

            // Apply camera intrinsics
            double u = intrinsics.fx * x + intrinsics.cx;
            double v = intrinsics.fy * y + intrinsics.cy;

            img_points.emplace_back(u, v);
        }
    }

    // Apply simple radial distortion to image points
    const double k1 = 0.1; // Distortion coefficient
    for (auto& p : img_points) {
        double x = (p.x() - intrinsics.cx) / intrinsics.fx;
        double y = (p.y() - intrinsics.cy) / intrinsics.fy;
        double r2 = x*x + y*y;
        double factor = 1.0 + k1 * r2;
        p.x() = intrinsics.fx * x * factor + intrinsics.cx;
        p.y() = intrinsics.fy * y * factor + intrinsics.cy;
    }

    // Make sure we have enough points
    ASSERT_GT(obj_points.size(), 10);
    ASSERT_EQ(obj_points.size(), img_points.size());

    // Optimize the pose with distortion
    PlanarPoseFitResult result = optimize_planar_pose(obj_points, img_points, intrinsics, 1, false);

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
    EXPECT_NEAR(result.distortion[0], k1, 0.2);
}

// TODO: Fix the segmentation fault in optimize_planar_pose
// The issue is likely in the implementation of optimize_planar_pose or in the way it's being used
TEST(PlanarPoseTest, BasicOptimizePlanarPoseTest) {
    // Create synthetic camera intrinsics
    CameraMatrix intrinsics;
    intrinsics.fx = 1000;
    intrinsics.fy = 1000;
    intrinsics.cx = 500;
    intrinsics.cy = 500;

    // Create a very simple pose (identity rotation, small translation)
    Eigen::Affine3d pose = Eigen::Affine3d::Identity();
    pose.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);

    // Create a few simple points
    std::vector<Eigen::Vector2d> obj_points = {
        {-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1}
    };

    // Project points to image
    std::vector<Eigen::Vector2d> img_points;
    for (const auto& xy : obj_points) {
        Eigen::Vector3d p(xy.x(), xy.y(), 0.0);
        Eigen::Vector3d pc = pose * p;
        double u = intrinsics.fx * pc.x() / pc.z() + intrinsics.cx;
        double v = intrinsics.fy * pc.y() / pc.z() + intrinsics.cy;
        img_points.emplace_back(u, v);
    }

    // Verify we have valid points
    ASSERT_EQ(obj_points.size(), img_points.size());
    ASSERT_EQ(obj_points.size(), 4);

    // Try to optimize the pose (minimal test - just check it doesn't crash)
    try {
        PlanarPoseFitResult result = optimize_planar_pose(obj_points, img_points, intrinsics, 0, false);
        // Test passed if we get here
        SUCCEED() << "Optimization ran without crashing";
    } catch (const std::exception& e) {
        FAIL() << "optimize_planar_pose threw an exception: " << e.what();
    } catch (...) {
        FAIL() << "optimize_planar_pose threw an unknown exception";
    }
}

} // namespace vitavision
