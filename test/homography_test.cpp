#include "calibration/homography.h"

// std
#include <random>
#include <cmath>

// gtest
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace vitavision;
using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;

namespace {

// Custom matcher for floating point matrix comparisons
MATCHER_P(IsMatrixNear, expected, "") {
    bool result = arg.rows() == expected.rows() && arg.cols() == expected.cols();
    if (!result) return false;
    
    for (int i = 0; i < arg.rows() && result; ++i) {
        for (int j = 0; j < arg.cols() && result; ++j) {
            result = std::abs(arg(i, j) - expected(i, j)) < 1e-6;
        }
    }
    return result;
}

// Helper to apply a homography to a point
Vec2 apply_homography(const Mat3& H, const Vec2& p) {
    Eigen::Vector3d ph(p.x(), p.y(), 1.0);
    Eigen::Vector3d q = H * ph;
    return Vec2(q.x() / q.z(), q.y() / q.z());
}

// Generate synthetic data with a known homography
void generate_synthetic_data(std::vector<Vec2>& src, 
                             std::vector<Vec2>& dst, 
                             Mat3& true_H,
                             int n_points = 50,
                             double noise_level = 0.5) {
    // Create a non-trivial homography (rotation + translation + perspective)
    const double angle = 0.1;  // radians
    const double c = std::cos(angle), s = std::sin(angle);
    const double tx = 10.0, ty = -5.0;
    true_H << c, -s, tx,
              s,  c, ty,
              0.001, -0.002, 1.0;
    
    // Normalize the scale
    true_H /= true_H(2, 2);
    
    // Random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::normal_distribution<double> noise(0.0, noise_level);
    
    src.resize(n_points);
    dst.resize(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        // Random source point
        src[i] = Vec2(dist(rng), dist(rng));
        
        // Apply true homography
        Vec2 exact_dst = apply_homography(true_H, src[i]);
        
        // Add noise to destination points
        dst[i] = Vec2(exact_dst.x() + noise(rng), exact_dst.y() + noise(rng));
    }
}

} // anonymous namespace

TEST(HomographyTest, ExactHomography) {
    // Create a simple homography (pure translation)
    Mat3 H_true = Mat3::Identity();
    H_true(0, 2) = 10.0;  // x translation
    H_true(1, 2) = -5.0;  // y translation
    
    // Create source and destination points with exact transformation
    std::vector<Vec2> src = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    
    std::vector<Vec2> dst;
    dst.reserve(src.size());
    for (const auto& p : src) {
        dst.push_back(apply_homography(H_true, p));
    }
    
    // Estimate homography
    Mat3 H_est = fit_homography(src, dst);
    
    // The estimated homography should be very close to the true one
    EXPECT_THAT(H_est, IsMatrixNear(H_true));
}

TEST(HomographyTest, NoisyHomography) {
    std::vector<Vec2> src, dst;
    Mat3 H_true;
    
    // Generate data with low noise
    generate_synthetic_data(src, dst, H_true, 50, 0.1);
    
    // Estimate homography
    Mat3 H_est = fit_homography(src, dst);
    
    // Verify that the estimated homography is close to the ground truth
    // with some tolerance for noise
    double tolerance = 1e-2;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(H_est(i, j), H_true(i, j), tolerance) 
                << "Matrix differs at (" << i << "," << j << ")";
        }
    }
    
    // Check the reprojection error
    double avg_error = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        Vec2 projected = apply_homography(H_est, src[i]);
        avg_error += (projected - dst[i]).norm();
    }
    avg_error /= src.size();
    
    // Average reprojection error should be small
    EXPECT_LT(avg_error, 0.15); // Should be close to the noise level
}

TEST(HomographyTest, InsufficientPoints) {
    // Less than 4 points should throw an exception
    std::vector<Vec2> src = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    std::vector<Vec2> dst = {{10.0, 0.0}, {11.0, 0.0}, {10.0, 1.0}};
    
    EXPECT_THROW(fit_homography(src, dst), std::invalid_argument);
}

TEST(HomographyTest, MismatchedSizes) {
    // Different number of source and destination points should throw
    std::vector<Vec2> src = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    std::vector<Vec2> dst = {{10.0, 0.0}, {11.0, 0.0}, {10.0, 1.0}};
    
    EXPECT_THROW(fit_homography(src, dst), std::invalid_argument);
}
