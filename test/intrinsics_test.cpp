#include "calibration/intrinsics.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <cmath>

using namespace vitavision;
using Vec2 = Eigen::Vector2d;

namespace {

// Helper function to distort a point with given intrinsics and distortion
void distort_and_project(double x, double y,
                         const Intrinsic& intr,
                         const std::vector<double>& k_radial,
                         double p1, double p2,
                         double& u, double& v) {
    const double r2 = x*x + y*y;
    
    // Apply radial distortion
    double radial = 1.0;
    double rpow = r2;
    for (double k : k_radial) {
        radial += k * rpow;
        rpow *= r2;
    }
    
    // Apply tangential distortion
    double x_t = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double y_t = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    
    // Project to pixel coords
    u = intr.fx * x_t + intr.cx;
    v = intr.fy * y_t + intr.cy;
}

// Generate synthetic observations with known intrinsics and distortion
std::vector<Observation> generate_synthetic_data(
    const Intrinsic& intr_true,
    const std::vector<double>& k_radial,
    double p1, double p2,
    int n_points = 300,
    double noise_level = 0.2) {
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.6, 0.6); // Normalized coords
    std::normal_distribution<double> noise(0.0, noise_level);
    
    std::vector<Observation> observations;
    observations.reserve(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        
        // Apply distortion and projection
        double u, v;
        distort_and_project(x, y, intr_true, k_radial, p1, p2, u, v);
        
        // Add noise
        u += noise(rng);
        v += noise(rng);
        
        observations.push_back({x, y, u, v});
    }
    
    return observations;
}

} // anonymous namespace

TEST(IntrinsicsTest, OptimizeExact) {
    // True intrinsics
    Intrinsic intr_true{800.0, 820.0, 640.0, 360.0};
    
    // True distortion
    std::vector<double> k_radial = {-0.20, 0.03};
    double p1 = 0.001, p2 = -0.0005;
    
    // Generate perfect synthetic data
    auto observations = generate_synthetic_data(
        intr_true, k_radial, p1, p2, 300, 0.0);
    
    // Initial guess (slightly off)
    Intrinsic initial_guess{780.0, 800.0, 630.0, 350.0};
    
    // Optimize
    auto result = optimize_intrinsics(observations, 2, initial_guess, false);
    
    // Check results
    EXPECT_NEAR(result.intrinsics.fx, intr_true.fx, 1e-6);
    EXPECT_NEAR(result.intrinsics.fy, intr_true.fy, 1e-6);
    EXPECT_NEAR(result.intrinsics.cx, intr_true.cx, 1e-6);
    EXPECT_NEAR(result.intrinsics.cy, intr_true.cy, 1e-6);
    
    EXPECT_NEAR(result.distortion[0], k_radial[0], 1e-6);
    EXPECT_NEAR(result.distortion[1], k_radial[1], 1e-6);
    EXPECT_NEAR(result.distortion[2], p1, 1e-6);
    EXPECT_NEAR(result.distortion[3], p2, 1e-6);
}

TEST(IntrinsicsTest, OptimizeNoisy) {
    // True intrinsics
    Intrinsic intr_true{800.0, 820.0, 640.0, 360.0};
    
    // True distortion
    std::vector<double> k_radial = {-0.20, 0.03};
    double p1 = 0.001, p2 = -0.0005;
    
    // Generate synthetic data with noise
    auto observations = generate_synthetic_data(
        intr_true, k_radial, p1, p2, 500, 0.5);
    
    // Initial guess (slightly off)
    Intrinsic initial_guess{780.0, 800.0, 630.0, 350.0};
    
    // Optimize
    auto result = optimize_intrinsics(observations, 2, initial_guess, false);
    
    // Check results (with tolerance for noise)
    EXPECT_NEAR(result.intrinsics.fx, intr_true.fx, 10.0);
    EXPECT_NEAR(result.intrinsics.fy, intr_true.fy, 10.0);
    EXPECT_NEAR(result.intrinsics.cx, intr_true.cx, 5.0);
    EXPECT_NEAR(result.intrinsics.cy, intr_true.cy, 5.0);
    
    EXPECT_NEAR(result.distortion[0], k_radial[0], 0.05);
    EXPECT_NEAR(result.distortion[1], k_radial[1], 0.05);
    EXPECT_NEAR(result.distortion[2], p1, 0.001);
    EXPECT_NEAR(result.distortion[3], p2, 0.001);
    
    // Check covariance matrix is valid (positive definite)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(result.covariance);
    EXPECT_GT(eigensolver.eigenvalues().minCoeff(), 0.0);
}

TEST(IntrinsicsTest, DifferentRadialCoeffs) {
    // Test with different numbers of radial coefficients
    Intrinsic intr_true{800.0, 820.0, 640.0, 360.0};
    
    // True distortion with 3 radial terms
    std::vector<double> k_radial = {-0.20, 0.03, 0.01};
    double p1 = 0.001, p2 = -0.0005;
    
    // Generate synthetic data
    auto observations = generate_synthetic_data(
        intr_true, k_radial, p1, p2, 300, 0.1);
    
    // Initial guess
    Intrinsic initial_guess{780.0, 800.0, 630.0, 350.0};
    
    // Optimize with 1, 2, and 3 radial coefficients
    auto result1 = optimize_intrinsics(observations, 1, initial_guess, false);
    auto result2 = optimize_intrinsics(observations, 2, initial_guess, false);
    auto result3 = optimize_intrinsics(observations, 3, initial_guess, false);
    
    // Check that results improve with more coefficients
    EXPECT_EQ(result1.distortion.size(), 3);  // 1 radial + 2 tangential
    EXPECT_EQ(result2.distortion.size(), 4);  // 2 radial + 2 tangential
    EXPECT_EQ(result3.distortion.size(), 5);  // 3 radial + 2 tangential
    
    // The third model should be closest to the true values
    EXPECT_NEAR(result3.distortion[0], k_radial[0], 0.05);
    EXPECT_NEAR(result3.distortion[1], k_radial[1], 0.05);
    EXPECT_NEAR(result3.distortion[2], k_radial[2], 0.05);
}

TEST(IntrinsicsTest, OptimizationSummary) {
    // True intrinsics
    Intrinsic intr_true{800.0, 820.0, 640.0, 360.0};
    
    // True distortion
    std::vector<double> k_radial = {-0.20, 0.03};
    double p1 = 0.001, p2 = -0.0005;
    
    // Generate synthetic data
    auto observations = generate_synthetic_data(
        intr_true, k_radial, p1, p2, 100, 0.1);
    
    // Initial guess
    Intrinsic initial_guess{780.0, 800.0, 630.0, 350.0};
    
    // Optimize with verbose output
    auto result = optimize_intrinsics(observations, 2, initial_guess, true);
    
    // Check that the summary is not empty
    EXPECT_FALSE(result.summary.empty());

    // Should contain success or convergence information
    EXPECT_TRUE(result.summary.find("Iterations") != std::string::npos && 
                result.summary.find("Final cost") != std::string::npos &&
                result.summary.find("CONVERGENCE") != std::string::npos);
}
