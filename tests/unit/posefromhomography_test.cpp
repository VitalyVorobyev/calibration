#include "calib/estimation/linear/posefromhomography.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

using namespace calib;

TEST(PoseFromHomography, RecoversPose) {
    CameraMatrix K;
    K.fx = 800;
    K.fy = 820;
    K.cx = 320;
    K.cy = 240;
    K.skew = 0.0;

    Eigen::Matrix3d R = (Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()) *
                         Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitZ()))
                            .toRotationMatrix();
    Eigen::Vector3d t(0.1, -0.2, 3.0);

    Eigen::Matrix3d H;
    H.col(0) = R.col(0);
    H.col(1) = R.col(1);
    H.col(2) = t;
    H = K.matrix() * H;

    auto res = pose_from_homography(K, H);
    ASSERT_TRUE(res.success);
    EXPECT_TRUE(R.isApprox(res.c_se3_t.linear(), 1e-9));
    EXPECT_TRUE(t.isApprox(res.c_se3_t.translation(), 1e-9));
    EXPECT_NEAR(res.scale, 1.0, 1e-12);
    EXPECT_NEAR(res.cond_check, 1.0, 1e-12);
}

TEST(PoseFromHomography, NegativeZFlipsPose) {
    CameraMatrix K;
    K.fx = 500;
    K.fy = 510;
    K.cx = 320;
    K.cy = 240;
    K.skew = 0.0;

    Eigen::Matrix3d R = (Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()) *
                         Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()))
                            .toRotationMatrix();
    Eigen::Vector3d t(0.2, 0.1, -2.0);  // negative z

    Eigen::Matrix3d H;
    H.col(0) = R.col(0);
    H.col(1) = R.col(1);
    H.col(2) = t;
    H = K.matrix() * H;

    auto res = pose_from_homography(K, H);
    ASSERT_TRUE(res.success);
    EXPECT_GT(res.c_se3_t.translation().z(), 0.0);
    EXPECT_TRUE((-R).isApprox(res.c_se3_t.linear(), 1e-9));
    EXPECT_TRUE((-t).isApprox(res.c_se3_t.translation(), 1e-9));
}

TEST(PoseFromHomography, DegenerateHomographyFails) {
    CameraMatrix K;
    K.fx = 400;
    K.fy = 400;
    K.cx = 320;
    K.cy = 240;
    K.skew = 0.0;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    auto res = pose_from_homography(K, H);
    EXPECT_FALSE(res.success);
}
