#include "calibration/distortion.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <cmath>

using namespace vitavision;
using Vec2 = Eigen::Vector2d;

namespace {

// Helper to apply distortion to a point
void distort_point(double x, double y, 
                   const std::vector<double>& k_radial,
                   double p1, double p2,
                   double& x_distorted, double& y_distorted) {
    const double r2 = x*x + y*y;
    
    // Apply radial distortion
    double radial = 1.0;
    double rpow = r2;
    for (double k : k_radial) {
        radial += k * rpow;
        rpow *= r2;
    }
    
    // Apply tangential distortion
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
}

// Generate synthetic observations with known distortion
std::vector<Observation<double>> generate_synthetic_data(
    const std::vector<double>& k_radial,
    double p1, double p2,
    double fx, double fy, double cx, double cy,
    int n_points = 100,
    double noise_level = 0.0) {
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.6, 0.6); // Normalized coords
    std::normal_distribution<double> noise(0.0, noise_level);

    std::vector<Observation<double>> observations;
    observations.reserve(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        
        // Apply distortion
        double x_distorted, y_distorted;
        distort_point(x, y, k_radial, p1, p2, x_distorted, y_distorted);
        
        // Project to pixel coords
        double u = fx * x_distorted + cx;
        double v = fy * y_distorted + cy;
        
        // Add noise
        u += noise(rng);
        v += noise(rng);
        
        observations.push_back({x, y, u, v});
    }
    
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

} // anonymous namespace

TEST(DistortionTest, ExactFit) {
    // Camera intrinsics
    double fx = 800.0, fy = 800.0, cx = 400.0, cy = 300.0;
    
    // True distortion coefficients
    std::vector<double> k_true = {-0.2, 0.05};  // k1, k2
    double p1_true = 0.001, p2_true = -0.0005;
    
    // Generate perfect synthetic data
    auto observations = generate_synthetic_data(
        k_true, p1_true, p2_true, fx, fy, cx, cy, 500, 0.0);
    
    // Fit distortion parameters
    auto distortion_opt = fit_distortion(observations, fx, fy, cx, cy, 2);
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
    double fx = 800.0, fy = 820.0, cx = 400.0, cy = 300.0;
    
    // True distortion coefficients
    std::vector<double> k_true = {-0.2, 0.05};  // k1, k2
    double p1_true = 0.001, p2_true = -0.0005;
    
    // Generate synthetic data with noise
    auto observations = generate_synthetic_data(
        k_true, p1_true, p2_true, fx, fy, cx, cy, 1000, 0.5);
    
    // Fit distortion parameters
    auto distortion_opt = fit_distortion(observations, fx, fy, cx, cy, 2);
    ASSERT_TRUE(distortion_opt.has_value());
    Eigen::VectorXd distortion = distortion_opt->distortion;
    
    // Results should be close but not exact due to noise
    EXPECT_NEAR(distortion[0], k_true[0], 0.01);
    EXPECT_NEAR(distortion[1], k_true[1], 0.01);
    EXPECT_NEAR(distortion[2], p1_true, 0.001);
    EXPECT_NEAR(distortion[3], p2_true, 0.001);
}

TEST(DistortionTest, DualModel) {
    double fx = 800.0, fy = 800.0, cx = 400.0, cy = 300.0;
    std::vector<double> k_true = {-0.2, 0.05};
    double p1_true = 0.001, p2_true = -0.0005;

    auto observations = generate_synthetic_data(
        k_true, p1_true, p2_true, fx, fy, cx, cy, 200, 0.0);

    auto dual_opt = fit_distortion_dual(observations, fx, fy, cx, cy, 2);
    ASSERT_TRUE(dual_opt.has_value());
    const auto& model = dual_opt->distortion;

    Vec2 pt(0.1, -0.2);
    Vec2 distorted = model.distort(pt);
    Vec2 recovered = model.undistort(distorted);

    EXPECT_NEAR(pt.x(), recovered.x(), 1e-10);
    EXPECT_NEAR(pt.y(), recovered.y(), 1e-10);
}

#if 0
TEST(DistortionTest, LSDesignMatrix) {
    // Test building the design matrix
    double fx = 800.0, fy = 800.0, cx = 400.0, cy = 300.0;
    
    std::vector<Observation<double>> obs = {
        {0.1, 0.2, 490.0, 470.0},  // x, y, u, v
        {-0.3, 0.4, 320.0, 630.0}
    };
    
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    const int num_radial = 2;
    
    LSDesign::build(obs, num_radial, fx, fy, cx, cy, A, b);
    
    // Check dimensions
    EXPECT_EQ(A.rows(), 4);  // 2 observations * 2 coordinates
    EXPECT_EQ(A.cols(), 4);  // 2 radial + 2 tangential
    EXPECT_EQ(b.size(), 4);
    
    // Check right-hand side (observed - undistorted)
    EXPECT_NEAR(b(0), 490.0 - (fx * 0.1 + cx), 1e-10);
    EXPECT_NEAR(b(1), 470.0 - (fy * 0.2 + cy), 1e-10);
    EXPECT_NEAR(b(2), 320.0 - (fx * -0.3 + cx), 1e-10);
    EXPECT_NEAR(b(3), 630.0 - (fy * 0.4 + cy), 1e-10);
}

TEST(DistortionTest, NormalEquations) {
    Eigen::MatrixXd A(4, 2);
    A << 1.0, 2.0,
    3.0, 4.0,
    5.0, 6.0,
    7.0, 8.0;
    
    Eigen::VectorXd b(4);
    b << 10.0, 20.0, 30.0, 40.0;
    
    // Solve normal equations
    Eigen::VectorXd x = LSDesign::solveNormal(A, b);
    
    // Verify solution is correct for this simple case
    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::VectorXd Atb = A.transpose() * b;
    Eigen::VectorXd expected = AtA.ldlt().solve(Atb);
    
    EXPECT_NEAR(x(0), expected(0), 1e-10);
    EXPECT_NEAR(x(1), expected(1), 1e-10);
}
#endif
