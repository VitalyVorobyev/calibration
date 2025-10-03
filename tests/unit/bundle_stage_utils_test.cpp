#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <string>
#include <unordered_map>
#include <vector>

#include "calib/pipeline/dataset.h"
#include "calib/pipeline/facades/intrinsics.h"
#include "calib/pipeline/detail/bundle_utils.h"

namespace calib::pipeline::detail {
namespace {

auto make_camera() -> PinholeCamera<BrownConradyd> {
    PinholeCamera<BrownConradyd> camera;
    camera.kmtx.fx = 500.0;
    camera.kmtx.fy = 500.0;
    camera.kmtx.cx = 320.0;
    camera.kmtx.cy = 240.0;
    camera.distortion.coeffs = Eigen::VectorXd::Zero(5);
    return camera;
}

IntrinsicCalibrationOutputs make_intrinsics_result() {
    IntrinsicCalibrationOutputs intrinsics;
    intrinsics.refine_result.success = true;
    intrinsics.refine_result.camera = make_camera();
    return intrinsics;
}

PlanarDetections make_detections(const std::string& sensor_id,
                                         const std::string& image_file) {
    PlanarDetections detections;
    detections.sensor_id = sensor_id;
    PlanarImageDetections image;
    image.file = image_file;
    for (int idx = 0; idx < 4; ++idx) {
        PlanarTargetPoint pt;
        pt.x = static_cast<double>(idx);
        pt.y = static_cast<double>(idx);
        pt.local_x = static_cast<double>(idx);
        pt.local_y = static_cast<double>(idx);
        image.points.push_back(pt);
    }
    detections.images.push_back(std::move(image));
    return detections;
}

TEST(BundleStageUtilsTest, SelectsRigObservationsBeforeHandeyeFallback) {
    BundleRigConfig rig;
    rig.rig_id = "rig";
    HandEyeObservationConfig obs;
    obs.view_id = "view";
    rig.observations.push_back(obs);

    HandEyePipelineConfig handeye_cfg;

    const auto* selected = select_bundle_observations(rig, &handeye_cfg);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected, &rig.observations);
}

TEST(BundleStageUtilsTest, SelectsHandeyeObservationsWhenRigEmpty) {
    BundleRigConfig rig;
    rig.rig_id = "rig";

    HandEyePipelineConfig handeye_cfg;
    HandEyeRigConfig he_rig;
    he_rig.rig_id = "rig";
    HandEyeObservationConfig obs;
    obs.view_id = "from_handeye";
    he_rig.observations.push_back(obs);
    handeye_cfg.rigs.push_back(he_rig);

    const auto* selected = select_bundle_observations(rig, &handeye_cfg);
    ASSERT_NE(selected, nullptr);
    ASSERT_EQ(selected->size(), 1U);
    EXPECT_EQ(selected->front().view_id, "from_handeye");
}

TEST(BundleStageUtilsTest, CollectSensorSetupReportsMissingSensors) {
    BundleRigConfig rig;
    rig.sensors = {"cam0", "cam1"};

    std::unordered_map<std::string, IntrinsicCalibrationOutputs> intrinsics;
    intrinsics.emplace("cam0", make_intrinsics_result());

    const auto setup = collect_bundle_sensor_setup(rig, intrinsics);
    EXPECT_EQ(setup.cameras.size(), 1U);
    ASSERT_EQ(setup.missing_sensors.size(), 1U);
    EXPECT_EQ(setup.missing_sensors.front(), "cam1");
    EXPECT_EQ(setup.sensor_to_index.at("cam0"), 0U);
}

TEST(BundleStageUtilsTest, CollectBundleObservationsBuildsViewSummaries) {
    const std::string sensor_id = "cam0";
    const std::string image_file = "frame0";

    auto detections = make_detections(sensor_id, image_file);
    detail::SensorDetectionsIndex index_entry;
    index_entry.detections = &detections;
    index_entry.image_lookup.emplace(image_file, &detections.images.front());
    std::unordered_map<std::string, detail::SensorDetectionsIndex> sensor_index;
    sensor_index.emplace(sensor_id, index_entry);

    std::unordered_map<std::string, IntrinsicCalibrationOutputs> intrinsics;
    intrinsics.emplace(sensor_id, make_intrinsics_result());

    HandEyeObservationConfig observation;
    observation.base_se3_gripper = Eigen::Isometry3d::Identity();
    observation.images.emplace(sensor_id, image_file);
    std::vector<HandEyeObservationConfig> observation_config{observation};

    std::vector<std::string> sensors{sensor_id};
    std::unordered_map<std::string, std::size_t> sensor_to_index{{sensor_id, 0U}};

    const auto result = collect_bundle_observations(observation_config, sensors, sensor_to_index,
                                                    sensor_index, intrinsics);

    EXPECT_EQ(result.observations.size(), 1U);
    ASSERT_EQ(result.accumulators.size(), 1U);
    EXPECT_EQ(result.accumulators.front().base.size(), 1U);
    EXPECT_EQ(result.used_view_count, 1U);
    ASSERT_EQ(result.views.size(), 1U);
    const auto& view_json = result.views.at(0);
    ASSERT_TRUE(view_json.contains("sensors"));
    ASSERT_EQ(view_json.at("sensors").size(), 1U);
    EXPECT_EQ(view_json.at("sensors").at(0).at("status"), "ok");
    EXPECT_TRUE(view_json.at("used").get<bool>());
}

TEST(BundleStageUtilsTest, HandeyeInitializationPrefersExistingResults) {
    BundleRigConfig rig;
    rig.rig_id = "rig";
    rig.sensors = {"cam0"};

    HandeyeResult handeye;
    handeye.success = true;
    handeye.g_se3_c = Eigen::Translation3d(1.0, 0.0, 0.0) * Eigen::Isometry3d::Identity();

    std::unordered_map<std::string, std::unordered_map<std::string, HandeyeResult>> handeye_results;
    handeye_results[rig.rig_id].emplace("cam0", handeye);

    std::vector<SensorAccumulator> accumulators(rig.sensors.size());

    const auto init = compute_handeye_initialization(rig, handeye_results, accumulators);
    EXPECT_FALSE(init.failed);
    ASSERT_EQ(init.report.size(), 1U);
    EXPECT_EQ(init.report.at(0).at("source"), "handeye");
    ASSERT_EQ(init.transforms.size(), 1U);
    EXPECT_NEAR(init.transforms.front().translation().x(), 1.0, 1e-9);
}

TEST(BundleStageUtilsTest, ChooseInitialTargetUsesConfigurationWhenProvided) {
    BundleRigConfig rig;
    rig.sensors = {"cam0"};
    rig.initial_target = Eigen::Translation3d(0.0, 1.0, 0.0) * Eigen::Isometry3d::Identity();

    std::vector<SensorAccumulator> accumulators(rig.sensors.size());
    std::vector<Eigen::Isometry3d> init_g_se3_c(rig.sensors.size(), Eigen::Isometry3d::Identity());

    const auto target = choose_initial_target(rig, accumulators, init_g_se3_c);
    EXPECT_EQ(target.source, "config");
    EXPECT_NEAR(target.pose.translation().y(), 1.0, 1e-9);
}

TEST(BundleStageUtilsTest, ChooseInitialTargetEstimatesFromAccumulatedPoses) {
    BundleRigConfig rig;
    rig.sensors = {"cam0"};

    SensorAccumulator accumulator;
    accumulator.base.push_back(Eigen::Isometry3d::Identity());
    Eigen::Isometry3d cam_pose = Eigen::Isometry3d::Identity();
    cam_pose.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);
    accumulator.cam.push_back(cam_pose);

    std::vector<SensorAccumulator> accumulators{accumulator};
    std::vector<Eigen::Isometry3d> init_g_se3_c{Eigen::Isometry3d::Identity()};

    const auto target = choose_initial_target(rig, accumulators, init_g_se3_c);
    EXPECT_EQ(target.source, "estimated");
    EXPECT_NEAR(target.pose.translation().z(), 1.0, 1e-9);
}

TEST(BundleStageUtilsTest, HandeyeInitializationSignalsFailureWithoutData) {
    BundleRigConfig rig;
    rig.rig_id = "rig";
    rig.sensors = {"cam0"};
    rig.min_angle_deg = 1.0;

    std::unordered_map<std::string, std::unordered_map<std::string, HandeyeResult>> handeye_results;
    std::vector<SensorAccumulator> accumulators(rig.sensors.size());

    const auto init = compute_handeye_initialization(rig, handeye_results, accumulators);
    EXPECT_TRUE(init.failed);
    ASSERT_EQ(init.report.size(), 1U);
    EXPECT_EQ(init.report.at(0).at("success"), false);
}

}  // namespace
}  // namespace calib::pipeline::detail
