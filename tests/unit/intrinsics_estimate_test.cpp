#include <gtest/gtest.h>

#include <limits>
#include <optional>

#include "calib/estimation/intrinsics.h"
#include "calib/utils/detail/intrinsics_utils.h"
#include "utils.h"

using namespace calib;

TEST(EstimateIntrinsics, RecoversCameraMatrix) {
    RNG rng(10);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 900;
    cam_gt.kmtx.fy = 920;
    cam_gt.kmtx.cx = 640;
    cam_gt.kmtx.cy = 360;
    cam_gt.kmtx.skew = 0.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d b_se3_t = Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Isometry3d::Identity();

    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(8, rng);
    sim.make_target_grid(6, 9, 0.03);
    sim.render_pixels();

    std::vector<PlanarView> views;
    views.reserve(sim.observations.size());
    for (const auto& ob : sim.observations) views.push_back(ob.view);

    auto res = estimate_intrinsics(views);
    ASSERT_TRUE(res.success);

    const auto& K = res.kmtx;
    const auto& Kg = cam_gt.kmtx;
    EXPECT_NEAR(K.fx, Kg.fx, 1e-6);
    EXPECT_NEAR(K.fy, Kg.fy, 1e-6);
    EXPECT_NEAR(K.cx, Kg.cx, 1e-6);
    EXPECT_NEAR(K.cy, Kg.cy, 1e-6);
    EXPECT_NEAR(K.skew, Kg.skew, 1e-9);

    ASSERT_EQ(res.views.size(), views.size());
    for (size_t i = 0; i < res.views.size(); ++i) {
        const auto& est = res.views[i].c_se3_t;
        const auto& gt = sim.c_se3_t[i];
        bool rot_match =
            gt.linear().isApprox(est.linear(), 1e-6) || gt.linear().isApprox(-est.linear(), 1e-6);
        EXPECT_TRUE(rot_match);
        double cosang = gt.translation().normalized().dot(est.translation().normalized());
        EXPECT_GT(std::abs(cosang), 0.999);
    }
}

TEST(EstimateIntrinsics, FailsWithTooFewViews) {
    RNG rng(5);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 800;
    cam_gt.kmtx.fy = 805;
    cam_gt.kmtx.cx = 320;
    cam_gt.kmtx.cy = 240;
    cam_gt.kmtx.skew = 0.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d b_se3_t = Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Isometry3d::Identity();

    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(3, rng);  // fewer than required views
    sim.make_target_grid(5, 7, 0.04);
    sim.render_pixels();

    std::vector<PlanarView> views;
    for (const auto& ob : sim.observations) views.push_back(ob.view);

    auto res = estimate_intrinsics(views);
    EXPECT_FALSE(res.success);
}

TEST(SanitizeIntrinsics, ClampsValuesWithinBounds) {
    CameraMatrix original{-50.0, std::numeric_limits<double>::infinity(), -100.0, 2000.0,
                          std::numeric_limits<double>::quiet_NaN()};

    CalibrationBounds bounds;
    bounds.fx_min = 200.0;
    bounds.fy_min = 150.0;
    bounds.cx_min = 100.0;
    bounds.cx_max = 200.0;
    bounds.cy_min = 50.0;
    bounds.cy_max = 75.0;
    bounds.skew_min = -1.0;
    bounds.skew_max = 1.0;

    auto [adjusted, modified] =
        calib::detail::sanitize_intrinsics(original, std::optional<CalibrationBounds>(bounds));

    EXPECT_TRUE(modified);
    EXPECT_DOUBLE_EQ(adjusted.fx, bounds.fx_min);
    EXPECT_DOUBLE_EQ(adjusted.fy, bounds.fy_min);
    EXPECT_DOUBLE_EQ(adjusted.cx, 150.0);
    EXPECT_DOUBLE_EQ(adjusted.cy, 62.5);
    EXPECT_DOUBLE_EQ(adjusted.skew, 0.0);
}
