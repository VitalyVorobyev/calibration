#include "calib/io/json.h"

#include <gtest/gtest.h>

#include <stdexcept>

#include <nlohmann/json.hpp>

#include "calib/estimation/common/ransac.h"
#include "calib/estimation/linear/planarpose.h"
#include "calib/estimation/optim/intrinsics.h"
#include "calib/estimation/optim/optimize.h"

using namespace calib;

TEST(JsonReflection, CameraMatrixRoundTrip) {
    CameraMatrix cam{100.0, 110.0, 10.0, 20.0, 1.5};

    nlohmann::json j = cam;
    CameraMatrix parsed = j.get<CameraMatrix>();

    EXPECT_DOUBLE_EQ(parsed.fx, cam.fx);
    EXPECT_DOUBLE_EQ(parsed.fy, cam.fy);
    EXPECT_DOUBLE_EQ(parsed.cx, cam.cx);
    EXPECT_DOUBLE_EQ(parsed.cy, cam.cy);
    EXPECT_DOUBLE_EQ(parsed.skew, cam.skew);
}

TEST(JsonSerialization, ObservationRoundTrip) {
    Observation<double> o{1.0, 2.0, 3.0, 4.0};
    nlohmann::json j = o;
    auto o2 = j.get<Observation<double>>();
    EXPECT_DOUBLE_EQ(o2.x, o.x);
    EXPECT_DOUBLE_EQ(o2.y, o.y);
    EXPECT_DOUBLE_EQ(o2.u, o.u);
    EXPECT_DOUBLE_EQ(o2.v, o.v);
}

TEST(JsonSerialization, IntrinsicsResultRoundTrip) {
    IntrinsicsOptimizationResult<Camera<BrownConradyd>> res;
    res.camera.kmtx = CameraMatrix{100, 100, 0, 0, 0};
    res.camera.distortion.coeffs = Eigen::VectorXd::Zero(5);
    res.covariance = Eigen::MatrixXd::Identity(5, 5);
    res.view_errors = {0.1, 0.2};
    res.report = "ok";
    res.c_se3_t = {Eigen::Isometry3d::Identity()};
    nlohmann::json j = res;
    auto r2 = j.get<IntrinsicsOptimizationResult<Camera<BrownConradyd>>>();
    EXPECT_NEAR(r2.camera.kmtx.fx, 100, 1e-9);
    EXPECT_EQ(r2.report, "ok");
    EXPECT_EQ(r2.view_errors.size(), 2U);
}

TEST(JsonSerialization, PlanarObservationVariations) {
    nlohmann::json object_format = {
        {"object", {1.0, 2.0}},
        {"image", {3.0, 4.0}},
    };
    PlanarObservation obs1 = object_format.get<PlanarObservation>();
    EXPECT_DOUBLE_EQ(obs1.object_xy.x(), 1.0);
    EXPECT_DOUBLE_EQ(obs1.image_uv.y(), 4.0);

    nlohmann::json tuple_format = {5.0, 6.0, 7.0, 8.0};
    PlanarObservation obs2 = tuple_format.get<PlanarObservation>();
    EXPECT_DOUBLE_EQ(obs2.object_xy.x(), 5.0);
    EXPECT_DOUBLE_EQ(obs2.object_xy.y(), 6.0);
    EXPECT_DOUBLE_EQ(obs2.image_uv.x(), 7.0);
    EXPECT_DOUBLE_EQ(obs2.image_uv.y(), 8.0);
}

TEST(JsonSerialization, PlanarViewRequiresMinimumObservations) {
    nlohmann::json valid = nlohmann::json::array({
        nlohmann::json{{"object", {0.0, 0.0}}, {"image", {1.0, 1.0}}},
        nlohmann::json{{"object", {0.0, 1.0}}, {"image", {1.0, 2.0}}},
        nlohmann::json{{"object", {1.0, 0.0}}, {"image", {2.0, 1.0}}},
        nlohmann::json{{"object", {1.0, 1.0}}, {"image", {2.0, 2.0}}},
    });
    auto view = valid.get<PlanarView>();
    EXPECT_EQ(view.size(), 4U);

    nlohmann::json invalid = nlohmann::json::array({
        nlohmann::json{{"object", {0.0, 0.0}}, {"image", {1.0, 1.0}}},
        nlohmann::json{{"object", {0.0, 1.0}}, {"image", {1.0, 2.0}}},
        nlohmann::json{{"object", {1.0, 0.0}}, {"image", {2.0, 1.0}}},
    });
    EXPECT_THROW(invalid.get<PlanarView>(), std::runtime_error);
}

TEST(JsonSerialization, RansacOptionsSupportsLegacyKeys) {
    nlohmann::json config = {
        {"max_iters", 200},
        {"thresh", 4.5},
        {"min_inliers", 10},
        {"confidence", 0.97},
        {"seed", 42},
        {"refit", false},
    };
    auto opts = config.get<RansacOptions>();
    EXPECT_EQ(opts.max_iters, 200);
    EXPECT_DOUBLE_EQ(opts.thresh, 4.5);
    EXPECT_FALSE(opts.refit_on_inliers);
}

TEST(JsonSerialization, OptimOptionsParsesStringsAndIntegers) {
    nlohmann::json node = {
        {"optimizer", "DENSE_SCHUR"},
        {"huber_delta", 0.5},
        {"epsilon", 1e-8},
        {"max_iterations", 50},
        {"compute_covariance", false},
        {"verbose", true},
    };
    auto opts = node.get<OptimOptions>();
    EXPECT_EQ(opts.optimizer, OptimizerType::DENSE_SCHUR);
    EXPECT_DOUBLE_EQ(opts.huber_delta, 0.5);
    EXPECT_TRUE(opts.verbose);

    nlohmann::json legacy = {
        {"optimizer", 3},
    };
    auto legacy_opts = legacy.get<OptimOptions>();
    EXPECT_EQ(legacy_opts.optimizer, OptimizerType::DENSE_QR);
}
