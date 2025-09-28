#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "calib/estimation/optim/bundle.h"
#include "calib/pipeline/pipeline.h"
#include "calib/pipeline/stages.h"
#include "utils.h"

namespace calib::pipeline {
namespace {

[[nodiscard]] auto image_name_for(const std::string& sensor_id, std::size_t idx) -> std::string {
    return sensor_id + "_view" + std::to_string(idx) + ".json";
}

[[nodiscard]] auto make_planar_detections(const std::string& sensor_id,
                                          const std::vector<BundleObservation>& observations)
    -> planar::PlanarDetections {
    planar::PlanarDetections detections;
    detections.sensor_id = sensor_id;
    detections.feature_type = "synthetic";
    detections.algo_version = "test";

    for (std::size_t idx = 0; idx < observations.size(); ++idx) {
        const auto& obs = observations[idx];
        planar::PlanarImageDetections image;
        image.file = image_name_for(sensor_id, idx);
        int point_id = 0;
        for (const auto& point : obs.view) {
            planar::PlanarTargetPoint target;
            target.id = point_id++;
            target.local_x = point.object_xy.x();
            target.local_y = point.object_xy.y();
            target.local_z = 0.0;
            target.x = point.image_uv.x();
            target.y = point.image_uv.y();
            image.points.push_back(std::move(target));
        }
        detections.images.push_back(std::move(image));
    }
    return detections;
}

struct SyntheticHandEyeData final {
    PinholeCamera<BrownConradyd> camera;
    Eigen::Isometry3d g_se3_c;
    Eigen::Isometry3d b_se3_t;
    std::vector<BundleObservation> observations;
};

[[nodiscard]] auto make_synthetic_handeye_data() -> SyntheticHandEyeData {
    RNG rng(17);
    PinholeCamera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 750.0;
    cam_gt.kmtx.fy = 760.0;
    cam_gt.kmtx.cx = 640.0;
    cam_gt.kmtx.cy = 360.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    Eigen::Isometry3d g_se3_c =
        make_pose(Eigen::Vector3d(0.05, -0.02, 0.1), Eigen::Vector3d(0.0, 1.0, 0.0), deg2rad(5.0));
    Eigen::Isometry3d b_se3_t =
        make_pose(Eigen::Vector3d(0.4, 0.1, 0.8), Eigen::Vector3d(0.0, 0.0, 1.0), deg2rad(-8.0));

    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(12, rng);
    sim.make_target_grid(6, 8, 0.03);
    sim.render_pixels(0.0, nullptr);

    std::vector<BundleObservation> observations;
    observations.reserve(sim.observations.size());
    for (const auto& obs : sim.observations) {
        if (obs.view.size() >= 16U) {
            observations.push_back(obs);
        }
    }

    SyntheticHandEyeData data;
    data.camera = cam_gt;
    data.g_se3_c = g_se3_c;
    data.b_se3_t = b_se3_t;
    data.observations = std::move(observations);
    return data;
}

TEST(IntrinsicStageTest, NameMatchesIdentifier) {
    IntrinsicStage stage;
    EXPECT_EQ(stage.name(), "intrinsics");
}

TEST(IntrinsicStageTest, FailsWhenConfigMissing) {
    PipelineContext context;
    IntrinsicStage stage;

    const auto result = stage.run(context);

    EXPECT_EQ(result.name, "intrinsics");
    EXPECT_FALSE(result.success);
    ASSERT_TRUE(result.summary.contains("error"));
    EXPECT_EQ(result.summary.at("error").get<std::string>(),
              "No intrinsics configuration supplied.");
    EXPECT_TRUE(context.intrinsic_results.empty());
}

TEST(IntrinsicStageTest, FailsWhenDatasetMissing) {
    PipelineContext context;

    planar::PlanarCalibrationConfig cfg;
    planar::CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam0";
    cfg.cameras.push_back(cam_cfg);
    context.set_intrinsics_config(cfg);

    IntrinsicStage stage;
    const auto result = stage.run(context);

    EXPECT_FALSE(result.success);
    ASSERT_TRUE(result.summary.contains("error"));
    EXPECT_EQ(result.summary.at("error").get<std::string>(),
              "Dataset does not contain planar camera captures.");
    EXPECT_TRUE(context.intrinsic_results.empty());
}

TEST(IntrinsicStageTest, ReportsMissingCameraConfigAndTagSummary) {
    PipelineContext context;

    planar::PlanarCalibrationConfig cfg;
    // Configure a different camera id so the lookup fails for cam0.
    planar::CameraConfig other_cam_cfg;
    other_cam_cfg.camera_id = "cam1";
    cfg.cameras.push_back(other_cam_cfg);
    context.set_intrinsics_config(cfg);

    planar::PlanarDetections detections;
    detections.tags.insert("synthetic");
    detections.tags.insert("recorded");
    context.dataset.planar_cameras.push_back(detections);

    IntrinsicStage stage;
    const auto result = stage.run(context);

    EXPECT_FALSE(result.success);
    ASSERT_TRUE(result.summary.contains("cameras"));
    const auto& cameras = result.summary.at("cameras");
    ASSERT_EQ(cameras.size(), 1);
    EXPECT_EQ(cameras.at(0).at("sensor_id").get<std::string>(), "cam0");
    EXPECT_EQ(cameras.at(0).at("status").get<std::string>(), "missing_camera_config");

    const auto& gating = result.summary.at("gating");
    EXPECT_TRUE(gating.at("synthetic").get<bool>());
    EXPECT_TRUE(gating.at("recorded").get<bool>());
    EXPECT_TRUE(context.intrinsic_results.empty());
}

TEST(IntrinsicStageTest, CapturesCalibrationFailure) {
    PipelineContext context;

    planar::PlanarCalibrationConfig cfg;
    cfg.options.min_corners_per_view = 1;
    planar::CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam0";
    cfg.cameras.push_back(cam_cfg);
    context.set_intrinsics_config(cfg);

    planar::PlanarDetections detections;
    detections.sensor_id = "cam0";
    planar::PlanarImageDetections image;
    image.file = "view0.png";
    planar::PlanarTargetPoint point;
    point.x = 10.0;
    point.y = 12.0;
    point.local_x = 1.0;
    point.local_y = 2.0;
    image.points.push_back(point);
    detections.images.push_back(image);
    context.dataset.planar_cameras.push_back(detections);

    IntrinsicStage stage;
    const auto result = stage.run(context);

    EXPECT_FALSE(result.success);
    ASSERT_TRUE(result.summary.contains("cameras"));
    const auto& cameras = result.summary.at("cameras");
    ASSERT_EQ(cameras.size(), 1);
    EXPECT_EQ(cameras.at(0).at("sensor_id").get<std::string>(), "cam0");
    EXPECT_EQ(cameras.at(0).at("status").get<std::string>(), "calibration_failed");
    EXPECT_THAT(cameras.at(0).at("error").get<std::string>(),
                ::testing::HasSubstr("Need at least 4 views"));
    EXPECT_TRUE(context.intrinsic_results.empty());
}

TEST(StereoCalibrationStageTest, MissingConfigFails) {
    PipelineContext context;
    context.intrinsic_results.emplace("cam0", planar::CalibrationRunResult{});

    StereoCalibrationStage stage;
    const auto result = stage.run(context);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.summary.at("status"), "missing_config");
}

TEST(StereoCalibrationStageTest, WaitsForMultipleIntrinsicResults) {
    PipelineContext context;
    StereoCalibrationConfig cfg;
    StereoPairConfig pair;
    pair.reference_sensor = "cam0";
    pair.target_sensor = "cam1";
    pair.views.push_back({"ref.json", "tgt.json"});
    cfg.pairs.push_back(pair);
    context.set_stereo_config(cfg);
    context.intrinsic_results.emplace("cam0", planar::CalibrationRunResult{});

    StereoCalibrationStage stage;
    const auto result = stage.run(context);

    EXPECT_EQ(result.name, "stereo");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.summary.at("input_cameras").get<std::size_t>(), 1U);
    EXPECT_EQ(result.summary.at("status").get<std::string>(),
              "waiting_for_multiple_intrinsic_results");
}

TEST(HandEyeCalibrationStageTest, WaitsForIntrinsicResults) {
    PipelineContext context;

    HandEyeCalibrationStage stage;
    const auto result = stage.run(context);

    EXPECT_EQ(result.name, "hand_eye");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.summary.at("status").get<std::string>(), "waiting_for_intrinsic_stage");
}

TEST(HandEyeCalibrationStageTest, RequiresConfigurationWhenIntrinsicsReady) {
    PipelineContext context;
    context.intrinsic_results.emplace("cam0", planar::CalibrationRunResult{});

    HandEyeCalibrationStage stage;
    const auto result = stage.run(context);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.summary.at("status").get<std::string>(), "missing_config");
}

TEST(BundleAdjustmentStageTest, WaitsForIntrinsicResults) {
    PipelineContext context;

    BundleAdjustmentStage stage;
    const auto result = stage.run(context);

    EXPECT_EQ(result.name, "bundle");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.summary.at("status").get<std::string>(), "waiting_for_intrinsic_stage");
}

TEST(BundleAdjustmentStageTest, RequiresConfigurationWhenIntrinsicsReady) {
    PipelineContext context;
    context.intrinsic_results.emplace("cam0", planar::CalibrationRunResult{});

    BundleAdjustmentStage stage;
    const auto result = stage.run(context);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.summary.at("status").get<std::string>(), "missing_config");
}

TEST(HandEyeCalibrationStageTest, CalibratesSyntheticHandEye) {
    const auto data = make_synthetic_handeye_data();
    ASSERT_GE(data.observations.size(), 4U);

    const std::string sensor_id = "cam0";

    PipelineContext context;
    context.dataset.planar_cameras.push_back(make_planar_detections(sensor_id, data.observations));

    planar::CalibrationRunResult intrinsics_result;
    intrinsics_result.outputs.point_scale = 1.0;
    intrinsics_result.outputs.point_center = {0.0, 0.0};
    intrinsics_result.outputs.refine_result.success = true;
    intrinsics_result.outputs.refine_result.camera = data.camera;
    context.intrinsic_results.emplace(sensor_id, intrinsics_result);

    HandEyeRigConfig rig;
    rig.rig_id = "arm";
    rig.sensors = {sensor_id};
    rig.min_angle_deg = 1.0;
    rig.options.max_iterations = 50;

    for (std::size_t idx = 0; idx < data.observations.size(); ++idx) {
        HandEyeObservationConfig view_cfg;
        view_cfg.view_id = "pose" + std::to_string(idx);
        view_cfg.base_se3_gripper = data.observations[idx].b_se3_g;
        view_cfg.images[sensor_id] = image_name_for(sensor_id, idx);
        rig.observations.push_back(std::move(view_cfg));
    }

    HandEyePipelineConfig he_cfg;
    he_cfg.rigs.push_back(rig);
    context.set_handeye_config(std::move(he_cfg));

    HandEyeCalibrationStage stage;
    const auto report = stage.run(context);

    EXPECT_TRUE(report.success);
    EXPECT_EQ(report.summary.at("status").get<std::string>(), "ok");
    ASSERT_TRUE(context.handeye_results.contains("arm"));
    const auto& sensor_results = context.handeye_results.at("arm");
    ASSERT_TRUE(sensor_results.contains(sensor_id));
    const auto& he_result = sensor_results.at(sensor_id);
    EXPECT_TRUE(he_result.success);
    EXPECT_LT((he_result.g_se3_c.translation() - data.g_se3_c.translation()).norm(), 5e-3);
    EXPECT_LT((he_result.g_se3_c.linear() - data.g_se3_c.linear()).norm(), 5e-2);
}

TEST(BundleAdjustmentStageTest, CalibratesSyntheticBundle) {
    auto data = make_synthetic_handeye_data();
    ASSERT_GE(data.observations.size(), 4U);

    const std::string sensor_id = "cam0";
    PipelineContext context;
    context.dataset.planar_cameras.push_back(make_planar_detections(sensor_id, data.observations));

    planar::CalibrationRunResult intrinsics_result;
    intrinsics_result.outputs.point_scale = 1.0;
    intrinsics_result.outputs.point_center = {0.0, 0.0};
    intrinsics_result.outputs.refine_result.success = true;
    intrinsics_result.outputs.refine_result.camera = data.camera;
    context.intrinsic_results.emplace(sensor_id, intrinsics_result);

    HandEyeRigConfig he_rig;
    he_rig.rig_id = "arm";
    he_rig.sensors = {sensor_id};
    he_rig.min_angle_deg = 1.0;
    he_rig.options.max_iterations = 50;
    for (std::size_t idx = 0; idx < data.observations.size(); ++idx) {
        HandEyeObservationConfig view_cfg;
        view_cfg.view_id = "pose" + std::to_string(idx);
        view_cfg.base_se3_gripper = data.observations[idx].b_se3_g;
        view_cfg.images[sensor_id] = image_name_for(sensor_id, idx);
        he_rig.observations.push_back(std::move(view_cfg));
    }

    HandEyePipelineConfig he_cfg;
    he_cfg.rigs.push_back(he_rig);
    context.set_handeye_config(he_cfg);

    HandEyeCalibrationStage handeye_stage;
    const auto he_report = handeye_stage.run(context);
    ASSERT_TRUE(he_report.success);

    BundleRigConfig bundle_rig;
    bundle_rig.rig_id = "arm";
    bundle_rig.sensors = {sensor_id};
    bundle_rig.min_angle_deg = 1.0;
    bundle_rig.options.optimize_intrinsics = false;
    bundle_rig.options.optimize_skew = false;
    bundle_rig.options.optimize_target_pose = true;
    bundle_rig.options.optimize_hand_eye = true;
    bundle_rig.options.max_iterations = 60;

    BundlePipelineConfig bundle_cfg;
    bundle_cfg.rigs.push_back(bundle_rig);
    context.set_bundle_config(bundle_cfg);

    BundleAdjustmentStage bundle_stage;
    const auto bundle_report = bundle_stage.run(context);

    EXPECT_TRUE(bundle_report.success);
    EXPECT_EQ(bundle_report.summary.at("status").get<std::string>(), "ok");
    ASSERT_TRUE(context.bundle_results.contains("arm"));
    const auto& bundle_result = context.bundle_results.at("arm");
    EXPECT_TRUE(bundle_result.success);
    EXPECT_LT((bundle_result.b_se3_t.translation() - data.b_se3_t.translation()).norm(), 1e-2);
    ASSERT_EQ(bundle_result.g_se3_c.size(), 1U);
    EXPECT_LT((bundle_result.g_se3_c.front().translation() - data.g_se3_c.translation()).norm(),
              5e-3);
}

}  // namespace
}  // namespace calib::pipeline
