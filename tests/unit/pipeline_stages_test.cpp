#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "calib/pipeline/pipeline.h"
#include "calib/pipeline/stages.h"

namespace calib::pipeline {
namespace {

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

TEST(HandEyeCalibrationStageTest, ReturnsPlaceholderWhenIntrinsicsReady) {
    PipelineContext context;
    context.intrinsic_results.emplace("cam0", planar::CalibrationRunResult{});

    HandEyeCalibrationStage stage;
    const auto result = stage.run(context);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.summary.at("status").get<std::string>(), "not_implemented");
}

}  // namespace
}  // namespace calib::pipeline
