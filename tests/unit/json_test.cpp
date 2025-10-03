#include "calib/io/json.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "calib/estimation/common/ransac.h"
#include "calib/estimation/linear/planarpose.h"
#include "calib/estimation/optim/intrinsics.h"
#include "calib/estimation/optim/optimize.h"
#include "calib/pipeline/handeye.h"

using namespace calib;

using calib::pipeline::BundlePipelineConfig;
using calib::pipeline::BundleRigConfig;
using calib::pipeline::HandEyeObservationConfig;
using calib::pipeline::HandEyePipelineConfig;
using calib::pipeline::HandEyeRigConfig;

namespace {

Eigen::Isometry3d makePose(const Eigen::Vector3d& translation, const Eigen::Vector3d& axis,
                           double angle) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    if (axis.norm() > 0.0) {
        T.linear() = Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
    }
    T.translation() = translation;
    return T;
}

}  // namespace

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
    const nlohmann::json object_format = {
        {"object_xy", {1.0, 2.0}},
        {"image_uv", {3.0, 4.0}},
    };
    const PlanarObservation obs1 = object_format;
    EXPECT_DOUBLE_EQ(obs1.object_xy.x(), 1.0);
    EXPECT_DOUBLE_EQ(obs1.image_uv.y(), 4.0);
}

TEST(JsonSerialization, PlanarViewRequiresMinimumObservations) {
    const nlohmann::json valid = nlohmann::json::array({
        nlohmann::json{{"object_xy", {0.0, 0.0}}, {"image_uv", {1.0, 1.0}}},
        nlohmann::json{{"object_xy", {0.0, 1.0}}, {"image_uv", {1.0, 2.0}}},
        nlohmann::json{{"object_xy", {1.0, 0.0}}, {"image_uv", {2.0, 1.0}}},
        nlohmann::json{{"object_xy", {1.0, 1.0}}, {"image_uv", {2.0, 2.0}}},
    });
    const PlanarView view = valid;
    EXPECT_EQ(view.size(), 4U);
}

TEST(JsonSerialization, RansacOptionsSupportsLegacyKeys) {
    const nlohmann::json config = {
        {"max_iters", 200},   {"thresh", 4.5}, {"min_inliers", 10},
        {"confidence", 0.97}, {"seed", 42},    {"refit_on_inliers", false},
    };
    const RansacOptions opts = config;
    EXPECT_EQ(opts.max_iters, 200);
    EXPECT_DOUBLE_EQ(opts.thresh, 4.5);
    EXPECT_FALSE(opts.refit_on_inliers);
}

TEST(JsonSerialization, OptimOptionsParsesStringsAndIntegers) {
    const nlohmann::json node = {
        {"optimizer", "dense_schur"}, {"huber_delta", 0.5},          {"epsilon", 1e-8},
        {"max_iterations", 50},       {"compute_covariance", false}, {"verbose", true},
    };
    const OptimOptions opts = node;
    EXPECT_EQ(opts.optimizer, OptimizerType::DENSE_SCHUR);
    EXPECT_DOUBLE_EQ(opts.huber_delta, 0.5);
    EXPECT_TRUE(opts.verbose);
}

TEST(HandEyeJsonIo, ObservationRoundTrip) {
    HandEyeObservationConfig cfg;
    cfg.view_id = "view-A";
    cfg.base_se3_gripper = makePose({0.1, -0.2, 0.3}, {1.0, 0.0, 0.5}, 0.25);
    cfg.images = {{"cam_left", "left.png"}, {"cam_right", "right.png"}};

    const nlohmann::json node = cfg;
    ASSERT_TRUE(node.contains("id"));
    ASSERT_TRUE(node.contains("base_se3_gripper"));
    ASSERT_TRUE(node.contains("images"));

    const HandEyeObservationConfig parsed = node.get<HandEyeObservationConfig>();
    EXPECT_EQ(parsed.view_id, cfg.view_id);
    EXPECT_TRUE(parsed.base_se3_gripper.matrix().isApprox(cfg.base_se3_gripper.matrix(), 1e-12));
    EXPECT_EQ(parsed.images, cfg.images);
}

TEST(HandEyeJsonIo, ObservationLegacyKeyFallback) {
    const Eigen::Isometry3d legacy_pose = makePose({-0.4, 0.1, 0.2}, {0.0, 1.0, 0.0}, 0.5);
    const nlohmann::json node = {
        {"id", "legacy"},
        {"b_T_g", legacy_pose},
        {"images", nlohmann::json::object()},
    };

    const HandEyeObservationConfig parsed = node.get<HandEyeObservationConfig>();
    EXPECT_EQ(parsed.view_id, "legacy");
    EXPECT_TRUE(parsed.base_se3_gripper.matrix().isApprox(legacy_pose.matrix(), 1e-12));
    EXPECT_TRUE(parsed.images.empty());
}

TEST(HandEyeJsonIo, RigRoundTripAndDefaults) {
    HandEyeObservationConfig obs;
    obs.view_id = "view-1";
    obs.base_se3_gripper = makePose({0.0, 0.0, 0.25}, {0.0, 0.0, 1.0}, 0.15);
    obs.images = {{"cam0", "frame0001.png"}};

    HandEyeRigConfig rig;
    rig.rig_id = "rig-main";
    rig.sensors = {"cam0", "cam1"};
    rig.observations = {obs};
    rig.min_angle_deg = 2.5;
    rig.options.optimizer = OptimizerType::DENSE_QR;
    rig.options.max_iterations = 55;
    rig.options.verbose = true;

    const nlohmann::json node = rig;
    EXPECT_TRUE(node.contains("options"));

    const HandEyeRigConfig parsed = node.get<HandEyeRigConfig>();
    ASSERT_EQ(parsed.sensors.size(), 2U);
    EXPECT_EQ(parsed.rig_id, rig.rig_id);
    EXPECT_DOUBLE_EQ(parsed.min_angle_deg, rig.min_angle_deg);
    EXPECT_EQ(parsed.options.optimizer, rig.options.optimizer);
    EXPECT_EQ(parsed.options.max_iterations, rig.options.max_iterations);
    EXPECT_TRUE(parsed.options.verbose);
    ASSERT_EQ(parsed.observations.size(), 1U);
    EXPECT_EQ(parsed.observations.front().view_id, obs.view_id);
    EXPECT_TRUE(parsed.observations.front().base_se3_gripper.matrix().isApprox(
        obs.base_se3_gripper.matrix(), 1e-12));

    nlohmann::json defaults_node = {
        {"sensors", nlohmann::json::array({"camX", "camY"})},
    };
    const HandEyeRigConfig defaults = defaults_node.get<HandEyeRigConfig>();
    EXPECT_EQ(defaults.rig_id, "camX");
    EXPECT_DOUBLE_EQ(defaults.min_angle_deg, 1.0);
    EXPECT_EQ(defaults.options.optimizer, OptimizerType::DEFAULT);
    EXPECT_TRUE(defaults.observations.empty());
}

TEST(HandEyeJsonIo, PipelineRoundTrip) {
    HandEyeRigConfig rig_a;
    rig_a.rig_id = "rig-A";
    rig_a.sensors = {"cam0"};
    rig_a.observations = {
        HandEyeObservationConfig{
            "view-A0", makePose({0.1, 0.2, 0.3}, {1.0, 0.0, 0.0}, 0.05), {{"cam0", "img.png"}}},
    };

    HandEyeRigConfig rig_b;
    rig_b.sensors = {"cam1"};
    rig_b.observations = {};

    HandEyePipelineConfig pipeline_cfg;
    pipeline_cfg.rigs = {rig_a, rig_b};

    const nlohmann::json node = pipeline_cfg;
    ASSERT_TRUE(node.contains("rigs"));

    const HandEyePipelineConfig parsed = node.get<HandEyePipelineConfig>();
    ASSERT_EQ(parsed.rigs.size(), 2U);
    EXPECT_EQ(parsed.rigs.front().rig_id, "rig-A");
    EXPECT_EQ(parsed.rigs.back().rig_id, "cam1");  // defaults to sensor name
}

TEST(BundleJsonIo, BundleRigRoundTripAndOptionalTarget) {
    BundleRigConfig rig;
    rig.rig_id = "bundle";
    rig.sensors = {"cam0", "cam1"};
    rig.min_angle_deg = 3.2;
    rig.options.optimize_intrinsics = true;
    rig.options.optimize_skew = true;
    rig.options.optimize_target_pose = false;
    rig.options.optimize_hand_eye = false;
    rig.options.verbose = true;
    rig.initial_target = makePose({0.4, -0.1, 0.6}, {0.0, 0.0, 1.0}, 0.2);
    rig.observations = {
        HandEyeObservationConfig{"bundle-view",
                                 makePose({0.0, 0.0, 0.1}, {0.0, 1.0, 0.0}, 0.3),
                                 {{"cam0", "bundle.png"}}},
    };

    const nlohmann::json node = rig;
    ASSERT_TRUE(node.contains("initial_target"));

    const BundleRigConfig parsed = node.get<BundleRigConfig>();
    EXPECT_EQ(parsed.rig_id, rig.rig_id);
    EXPECT_TRUE(parsed.initial_target.has_value());
    EXPECT_TRUE(parsed.initial_target->matrix().isApprox(rig.initial_target->matrix(), 1e-12));
    EXPECT_TRUE(parsed.options.optimize_intrinsics);
    EXPECT_TRUE(parsed.options.optimize_skew);
    EXPECT_FALSE(parsed.options.optimize_target_pose);
    EXPECT_FALSE(parsed.options.optimize_hand_eye);
    EXPECT_TRUE(parsed.options.verbose);

    nlohmann::json legacy = node;
    legacy.erase("initial_target");
    const BundleRigConfig no_target = legacy.get<BundleRigConfig>();
    EXPECT_FALSE(no_target.initial_target.has_value());
}

TEST(BundleJsonIo, PipelineRoundTrip) {
    BundleRigConfig rig;
    rig.sensors = {"cam-only"};
    rig.observations = {};

    BundlePipelineConfig pipeline_cfg;
    pipeline_cfg.rigs = {rig};

    const nlohmann::json node = pipeline_cfg;
    const BundlePipelineConfig parsed = node.get<BundlePipelineConfig>();
    ASSERT_EQ(parsed.rigs.size(), 1U);
    EXPECT_EQ(parsed.rigs.front().rig_id, "cam-only");
    EXPECT_FALSE(parsed.rigs.front().initial_target.has_value());
}
