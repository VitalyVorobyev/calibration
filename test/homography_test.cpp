#include "calib/homography.h"

// std
#include <random>
#include <cmath>

// gtest
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace calib;
using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;

// Helper to apply a homography to a point
static Vec2 apply_homography(const Mat3& H, const Vec2& p) {
    return (H * p.homogeneous()).hnormalized();
}

// Generate synthetic data with a known homography
void generate_synthetic_data(PlanarView& view,
                             Mat3& true_H,
                             int n_points = 50,
                             double noise_level = -1) {
    // Create a non-trivial homography (rotation + translation + perspective)
    const double angle = 0.1;  // radians
    const double c = std::cos(angle), s = std::sin(angle);
    const double tx = 10.0, ty = -5.0;
    true_H << c, -s, tx,
              s,  c, ty,
              0.001, -0.002, 1.0;

    // Random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::normal_distribution<double> noise(0.0, noise_level);

    view.resize(n_points);
    std::generate(view.begin(), view.end(), [&]() -> PlanarObservation {
        const Vec2 point(dist(rng), dist(rng));

        // Apply true homography
        Vec2 pixel = apply_homography(true_H, point);

        // Add noise to destination points
        if (noise_level > 0) {
            pixel += Vec2(noise(rng), noise(rng));
        }

        return { point, pixel };
    });
}

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
    PlanarView view(src.size());
    std::transform(src.begin(), src.end(), view.begin(),
        [&H_true](const Vec2& point) -> PlanarObservation {
            return {point, apply_homography(H_true, point)};
        });

    // Estimate homography
    const auto hres = estimate_homography(view);
    ASSERT_TRUE(hres.success);
    auto result = optimize_homography(view, hres.hmtx);
    ASSERT_TRUE(result.success);

    // The estimated homography should be very close to the true one
    ASSERT_TRUE(result.homography.isApprox(H_true, 1e-6));
}

TEST(HomographyTest, NoisyHomography) {
    PlanarView view;
    Mat3 H_true;

    // Generate data with low noise
    generate_synthetic_data(view, H_true, 50, 0.1);

    // Estimate homography
    const auto hres = estimate_homography(view);
    ASSERT_TRUE(hres.success);
    // Average reprojection error should be small
    EXPECT_LT(hres.symmetric_rms_px, 0.25); // Should be close to the noise level

    auto result = optimize_homography(view, hres.hmtx);
    ASSERT_TRUE(result.success);

    // Verify that the estimated homography is close to the ground truth
    // with some tolerance for noise
    constexpr double tolerance = 1e-2;
    ASSERT_TRUE(result.homography.isApprox(H_true, tolerance));
}

TEST(HomographyTest, InsufficientPoints) {
    // Less than 4 points should throw an exception
    PlanarView view {
        {{0.0, 0.0}, {10.0, 0.0}},
        {{1.0, 0.0}, {11.0, 0.0}},
        {{0.0, 1.0}, {10.0, 1.0}}
    };
    EXPECT_THROW(optimize_homography(view, Mat3::Identity()), std::invalid_argument);
}
