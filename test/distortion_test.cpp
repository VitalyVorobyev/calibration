
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// std
#include <random>
#include <cmath>
#include <algorithm>

#include "calib/distortion.h"
#include "calib/cameramatrix.h"

using namespace calib;
using Vec2 = Eigen::Vector2d;

// Generate synthetic observations with known distortion
std::vector<Observation<double>> generate_synthetic_data(
    const std::vector<double>& k_radial,
    double p1, double p2, const CameraMatrix& camera,
    int n_points = 100,
    double noise_level = 0.0
) {

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.6, 0.6); // Normalized coords
    std::normal_distribution<double> noise(0.0, noise_level);

    const int numk = static_cast<int>(k_radial.size());
    Eigen::VectorXd coeffs(numk + 2);
    for (size_t i = 0; i < numk; ++i) {
        coeffs[i] = k_radial[i];
    }
    coeffs[k_radial.size()] = p1;
    coeffs[k_radial.size() + 1] = p2;

    std::vector<Observation<double>> observations(static_cast<size_t>(n_points));
    std::generate(observations.begin(), observations.end(),
        [&]() -> Observation<double> {
            Eigen::Vector2d xy{dist(rng), dist(rng)};

            auto distorted = apply_distortion(xy, coeffs);
            auto uv = camera.denormalize(distorted);
            uv.x() += noise(rng);
            uv.y() += noise(rng);

            return {xy.x(), xy.y(), uv.x(), uv.y()};
        });

    return observations;
}

// Custom matcher for vector comparison
MATCHER_P2(IsVectorNear, expected, tolerance, "") {
    bool result = arg.size() == expected.size();
    if (!result) return false;

    for (size_t i = 0; i < arg.size() && result; ++i) {
        result = std::abs(arg[i] - expected[i]) < tolerance;
    }
    return result;
}

TEST(DistortionTest, ExactFit) {
    // Camera intrinsics
    CameraMatrix camera{800.0, 800.0, 400.0, 300.0, 0.0};

    // True distortion coefficients
    std::vector<double> k_true = {-0.2, 0.05};  // k1, k2
    double p1_true = 0.001, p2_true = -0.0005;

    // Generate perfect synthetic data
    auto observations = generate_synthetic_data(
        k_true, p1_true, p2_true, camera, 500, 0.0);

    // Fit distortion parameters
    auto distortion_opt = fit_distortion(observations, camera, 2);
    ASSERT_TRUE(distortion_opt.has_value());
    Eigen::VectorXd distortion = distortion_opt->distortion;

    // Check results
    EXPECT_NEAR(distortion[0], k_true[0], 1e-10);
    EXPECT_NEAR(distortion[1], k_true[1], 1e-10);
    EXPECT_NEAR(distortion[2], p1_true, 1e-10);
    EXPECT_NEAR(distortion[3], p2_true, 1e-10);
}

TEST(DistortionTest, NoisyFit) {
    // Camera intrinsics
    CameraMatrix camera{800.0, 820.0, 400.0, 300.0, 0.0};
    // double fx = 800.0, fy = 820.0, cx = 400.0, cy = 300.0;

    // True distortion coefficients
    std::vector<double> k_true = {-0.2, 0.05};  // k1, k2
    double p1_true = 0.001, p2_true = -0.0005;

    // Generate synthetic data with noise
    auto observations = generate_synthetic_data(
        k_true, p1_true, p2_true, camera, 1000, 0.5);

    // Fit distortion parameters
    auto distortion_opt = fit_distortion(observations, camera, 2);
    ASSERT_TRUE(distortion_opt.has_value());
    Eigen::VectorXd distortion = distortion_opt->distortion;

    // Results should be close but not exact due to noise
    EXPECT_NEAR(distortion[0], k_true[0], 0.01);
    EXPECT_NEAR(distortion[1], k_true[1], 0.01);
    EXPECT_NEAR(distortion[2], p1_true, 0.001);
    EXPECT_NEAR(distortion[3], p2_true, 0.001);
}

TEST(DistortionTest, DualModel) {
    const CameraMatrix camera{800.0, 800.0, 400.0, 300.0, 0.0};
    std::vector<double> k_true = {-0.2, 0.05};
    double p1_true = 0.001, p2_true = -0.0005;

    auto observations = generate_synthetic_data(
        k_true, p1_true, p2_true, camera, 200, 0.0);

    auto dual_opt = fit_distortion_dual(observations, camera, 2);
    ASSERT_TRUE(dual_opt.has_value());
    const auto& model = dual_opt->distortion;

    Vec2 pt(0.1, -0.2);
    Vec2 distorted = model.distort(pt);
    Vec2 recovered = model.undistort(distorted);

    EXPECT_NEAR(pt.x(), recovered.x(), 1e-4);
    EXPECT_NEAR(pt.y(), recovered.y(), 1e-4);
}
