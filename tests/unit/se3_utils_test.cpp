#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/estimation/common/se3_utils.h"

using namespace calib;

TEST(Se3Utils, ProjectToSO3ReturnsRotation) {
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d perturbed = R;
    perturbed(0, 1) += 0.05;
    perturbed(1, 0) -= 0.02;

    Eigen::Matrix3d projected = project_to_so3(perturbed);
    EXPECT_NEAR(projected.determinant(), 1.0, 1e-12);
    EXPECT_NEAR((projected * projected.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0,
                1e-10);
}

TEST(Se3Utils, LogExpRoundTrip) {
    Eigen::Vector3d w(0.1, -0.2, 0.3);
    Eigen::Matrix3d R = exp_so3(w);
    Eigen::Vector3d w_back = log_so3(R);
    EXPECT_NEAR((w - w_back).norm(), 0.0, 1e-10);
}

TEST(Se3Utils, AverageAffinesComputesMean) {
    std::vector<Eigen::Isometry3d> poses;
    for (int i = 0; i < 5; ++i) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(static_cast<double>(i), 0.0, 0.0);
        poses.push_back(T);
    }
    Eigen::Isometry3d avg = average_affines(poses);
    EXPECT_NEAR(avg.translation().x(), 2.0, 1e-12);
    EXPECT_TRUE(avg.linear().isApprox(Eigen::Matrix3d::Identity()));
}

TEST(Se3Utils, RidgeAndSolveLLSQ) {
    Eigen::MatrixXd A(3, 2);
    A << 1, 0, 0, 1, 1, 1;
    Eigen::VectorXd b(3);
    b << 1, 2, 3;

    Eigen::VectorXd x = solve_llsq(A, b);
    Eigen::VectorXd x_ridge = ridge_llsq(A, b, 1e-6);

    EXPECT_NEAR((A * x - b).norm(), 0.0, 1e-10);
    EXPECT_NEAR((A * x_ridge - b).norm(), 0.0, 2e-6);
}
