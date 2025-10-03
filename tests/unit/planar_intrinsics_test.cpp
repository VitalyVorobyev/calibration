#include "calib/pipeline/facades/intrinsics.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "calib/pipeline/dataset.h"
#include "calib/pipeline/reports/intrinsics.h"
#include "utils.h"

using namespace calib;
using namespace calib::pipeline;

TEST(PlanarIntrinsicsUtils, CollectPlanarViewsRespectsThreshold) {
    PlanarDetections detections;
    PlanarImageDetections accepted;
    accepted.file = "view0.png";
    accepted.points = {PlanarTargetPoint{.x = 100.0, .y = 200.0, .local_x = 1.0, .local_y = 1.0},
                       PlanarTargetPoint{.x = 150.0, .y = 220.0, .local_x = 2.0, .local_y = 1.0},
                       PlanarTargetPoint{.x = 180.0, .y = 240.0, .local_x = 1.0, .local_y = 3.0}};
    PlanarImageDetections rejected;
    rejected.file = "view1.png";
    rejected.points = {PlanarTargetPoint{.x = 50.0, .y = 60.0, .local_x = 0.0, .local_y = 0.0},
                       PlanarTargetPoint{.x = 55.0, .y = 70.0, .local_x = 0.5, .local_y = 0.5}};
    detections.images = {accepted, rejected};

    IntrinsicCalibrationOptions opts;
    opts.min_corners_per_view = 3;

    std::vector<ActiveView> views;
    auto planar_views = collect_planar_views(detections, opts, views);

    ASSERT_EQ(planar_views.size(), 1);
    ASSERT_EQ(views.size(), 1);
    EXPECT_EQ(views[0].source_image, "view0.png");
    EXPECT_EQ(views[0].corner_count, 3U);

    const auto& view = planar_views[0];
    ASSERT_EQ(view.size(), 3U);
    EXPECT_DOUBLE_EQ(view[0].object_xy.x(), 0.0);
    EXPECT_DOUBLE_EQ(view[0].object_xy.y(), 0.0);
    EXPECT_DOUBLE_EQ(view[1].object_xy.x(), 2.0);
    EXPECT_DOUBLE_EQ(view[1].object_xy.y(), 0.0);
    EXPECT_DOUBLE_EQ(view[2].object_xy.x(), 0.0);
    EXPECT_DOUBLE_EQ(view[2].object_xy.y(), 4.0);
}

TEST(PlanarIntrinsicsUtils, LoadPlanarObservationsParsesDetectorMetadata) {
    const auto temp_dir = std::filesystem::path(testing::TempDir());
    const auto json_path = temp_dir / "planar_observations.json";

    std::ofstream file(json_path);
    ASSERT_TRUE(file.is_open());
    file << R"({
        "image_directory": "imgs",
        "feature_type": "planar",
        "algo_version": "v2",
        "params_hash": "hash123",
        "sensor_id": "cam0",
        "tags": ["recorded"],
        "images": [
            {
                "file": "view0.png",
                "points": [
                    {"x": 10.0, "y": 20.0, "id": 5, "local_x": 1.0, "local_y": 2.0, "local_z": 0.1},
                    {"x": 30.0, "y": 40.0, "id": 6, "local_x": 3.0, "local_y": 4.0, "local_z": 0.2}
                ]
            }
        ]
    })";
    file.close();

    std::ifstream input(json_path);
    ASSERT_TRUE(input.is_open());
    nlohmann::json json_data;
    input >> json_data;
    auto detections = json_data.get<PlanarDetections>();
    EXPECT_EQ(detections.image_directory, "imgs");
    EXPECT_EQ(detections.feature_type, "planar");
    EXPECT_EQ(detections.algo_version, "v2");
    EXPECT_EQ(detections.params_hash, "hash123");
    ASSERT_EQ(detections.images.size(), 1U);

    const auto& image = detections.images.front();
    EXPECT_EQ(image.file, "view0.png");
    ASSERT_EQ(image.points.size(), 2U);

    const auto& pt0 = image.points[0];
    EXPECT_DOUBLE_EQ(pt0.x, 10.0);
    EXPECT_DOUBLE_EQ(pt0.y, 20.0);
    EXPECT_EQ(pt0.id, 5);
    EXPECT_DOUBLE_EQ(pt0.local_x, 1.0);
    EXPECT_DOUBLE_EQ(pt0.local_y, 2.0);
    EXPECT_DOUBLE_EQ(pt0.local_z, 0.1);

    const auto& pt1 = image.points[1];
    EXPECT_DOUBLE_EQ(pt1.x, 30.0);
    EXPECT_DOUBLE_EQ(pt1.y, 40.0);
    EXPECT_EQ(pt1.id, 6);
    EXPECT_DOUBLE_EQ(pt1.local_x, 3.0);
    EXPECT_DOUBLE_EQ(pt1.local_y, 4.0);
    EXPECT_DOUBLE_EQ(pt1.local_z, 0.2);
}

TEST(PlanarIntrinsicsUtils, LoadPlanarObservationsRequiresFeatureType) {
    const auto temp_dir = std::filesystem::path(testing::TempDir());
    const auto json_path = temp_dir / "planar_observations_missing_feature.json";

    std::ofstream file(json_path);
    ASSERT_TRUE(file.is_open());
    file << R"({
        "image_directory": "imgs",
        "algo_version": "v2",
        "params_hash": "hash123",
        "images": []
    })";
    file.close();

    std::ifstream input(json_path);
    ASSERT_TRUE(input.is_open());
    nlohmann::json json_data;
    input >> json_data;

    EXPECT_THROW({ (void)json_data.get<PlanarDetections>(); }, nlohmann::json::exception);
}

TEST(PlanarIntrinsicsUtils, PrintCalibrationSummaryIncludesKeyData) {
    CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam_test";

    IntrinsicCalibrationOutputs outputs;
    outputs.invalid_k_warnings = 1;
    outputs.pose_warnings = 2;
    outputs.linear_kmtx = CameraMatrix{900.0, 910.0, 320.0, 240.0, 0.05};
    outputs.total_input_views = 4;
    outputs.accepted_views = 3;
    outputs.refine_result.camera = PinholeCamera<BrownConradyd>(
        CameraMatrix{905.0, 915.0, 321.0, 241.0, 0.02}, Eigen::VectorXd::LinSpaced(5, 0.0, 0.4));
    outputs.refine_result.view_errors = {0.1, 0.2, 0.3};

    std::ostringstream oss;
    print_calibration_summary(oss, cam_cfg, outputs);
    const auto summary = oss.str();
    EXPECT_NE(summary.find("== Camera cam_test =="), std::string::npos);
    EXPECT_NE(summary.find("Point scale applied to board coordinates: 2"), std::string::npos);
    EXPECT_NE(summary.find("Point center removed before scaling: [1.5, -0.5]"), std::string::npos);
    EXPECT_NE(summary.find("Linear stage warnings: 1 invalid camera matrices, 2 homography"),
              std::string::npos);
    EXPECT_NE(summary.find("Initial fx/fy/cx/cy: 900"), std::string::npos);
    EXPECT_NE(summary.find("Refined fx/fy/cx/cy: 905"), std::string::npos);
    EXPECT_NE(summary.find("Distortion coeffs:"), std::string::npos);
    EXPECT_NE(summary.find("0 0.1 0.2 0.3 0.4"), std::string::npos);
    EXPECT_NE(summary.find("Views considered: 4, after threshold: 3"), std::string::npos);
    EXPECT_NE(summary.find("Per-view RMS (px): 0.1 0.2 0.3"), std::string::npos);
}

TEST(PlanarIntrinsicsUtils, BuildOutputJsonIncludesFixedDistortionMetadata) {
    IntrinsicCalibrationConfig cfg;
    cfg.algorithm = "planar";
    cfg.options.optim_options.fixed_distortion_indices = {0, 2};
    cfg.options.optim_options.fixed_distortion_values = {0.1, 0.05};
    cfg.options.optim_options.num_radial = 3;
    cfg.options.min_corners_per_view = 20;

    CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam0";
    cam_cfg.model = "pinhole_brown_conrady";
    cam_cfg.image_size = std::array<int, 2>{640, 480};

    PlanarDetections detections;
    detections.image_directory = "imgs";
    detections.algo_version = "v1";
    detections.params_hash = "hash";

    IntrinsicCalibrationOutputs outputs;
    outputs.total_input_views = 5;
    outputs.accepted_views = 3;
    outputs.min_corner_threshold = 20;
    outputs.total_points_used = 60;
    outputs.invalid_k_warnings = 1;
    outputs.pose_warnings = 2;
    outputs.linear_kmtx = CameraMatrix{800.0, 810.0, 320.0, 240.0, 0.5};
    outputs.linear_view_indices = {0, 2};
    outputs.active_views = {ActiveView{"view0.png", 25}, ActiveView{"view1.png", 35}};

    outputs.refine_result.camera = PinholeCamera<BrownConradyd>(
        CameraMatrix{805.0, 815.0, 322.0, 241.0, 0.4}, Eigen::VectorXd::Constant(5, 0.01));
    outputs.refine_result.view_errors = {0.5, 0.7};

    const auto report = build_planar_intrinsics_report(cfg, cam_cfg, detections, outputs);
    const nlohmann::json json = report;

    ASSERT_TRUE(json.contains("calibrations"));
    const auto& calibrations = json.at("calibrations");
    ASSERT_EQ(calibrations.size(), 1);
    const auto& calibration = calibrations[0];
    const auto& options_json = calibration.at("options");
    ASSERT_TRUE(options_json.contains("fixed_distortion_indices"));
    EXPECT_EQ(options_json.at("fixed_distortion_indices").size(), 2);
    EXPECT_EQ(options_json.at("fixed_distortion_indices")[0], 0);
    EXPECT_EQ(options_json.at("fixed_distortion_indices")[1], 2);
    ASSERT_TRUE(options_json.contains("fixed_distortion_values"));
    EXPECT_DOUBLE_EQ(options_json.at("fixed_distortion_values")[0], 0.1);
    EXPECT_DOUBLE_EQ(options_json.at("fixed_distortion_values")[1], 0.05);

    const auto& cameras = calibration.at("cameras");
    ASSERT_EQ(cameras.size(), 1);
    const auto& camera_json = cameras[0];
    ASSERT_TRUE(camera_json.contains("initial_guess"));
    const auto& warnings = camera_json.at("initial_guess").at("warning_counts");
    EXPECT_EQ(warnings.at("invalid_camera_matrix"), 1);
    EXPECT_EQ(warnings.at("homography_decomposition_failures"), 2);

    const auto& result = camera_json.at("result");
    ASSERT_TRUE(result.contains("distortion"));
    EXPECT_EQ(result.at("distortion").at("coefficients").size(), 5);
}

TEST(PlanarIntrinsicsReport, JsonRoundTripPreservesData) {
    CalibrationReport report;
    IntrinsicCalibrationOptions options;
    options.min_corners_per_view = 10;
    options.refine = true;
    options.optimize_skew = true;
    options.num_radial = 3;
    options.huber_delta = 1.5;
    options.max_iterations = 50;
    options.epsilon = 1e-6;
    options.point_scale = 2.0;
    options.auto_center_points = false;
    options.point_center = std::array<double, 2>{0.1, -0.2};
    options.fixed_distortion_indices = {0, 2};
    options.fixed_distortion_values = {0.01, -0.02};
    options.homography_ransac = RansacConfig{
        .max_iters = 500, .thresh = 0.5, .min_inliers = 20, .confidence = 0.85};

    InitialGuessReport guess;
    guess.intrinsics.fx = 800.0;
    guess.intrinsics.fy = 790.0;
    guess.intrinsics.cx = 320.0;
    guess.intrinsics.cy = 240.0;
    guess.intrinsics.skew = 0.1;
    guess.used_view_indices = {0, 2};
    guess.warning_counts = InitialGuessWarningCounts{.invalid_camera_matrix = 1,
                                                     .homography_decomposition_failures = 2};

    IntrinsicsResultReport result;
    result.intrinsics.fx = 810.0;
    result.intrinsics.fy = 805.0;
    result.intrinsics.cx = 321.0;
    result.intrinsics.cy = 239.0;
    result.distortion_model = "pinhole_brown_conrady";
    result.distortion_coefficients = {0.01, -0.02, 0.003};
    result.reprojection_rms_px = 0.25;
    result.per_view.push_back(PlanarViewReport{.source_image = "view0.png",
                                               .corner_count = 50,
                                               .rms_px = 0.2,
                                               .used_in_linear_stage = true});

    CameraReport camera;
    camera.camera_id = "cam0";
    camera.model = "pinhole_brown_conrady";
    camera.image_size = std::array<int, 2>{640, 480};
    camera.initial_guess = guess;
    camera.result = result;

    CalibrationReport calibration;
    calibration.type = "intrinsics";
    calibration.algorithm = "planar";
    calibration.options = options;
    calibration.detector = nlohmann::json{{"type", "synthetic"}};
    calibration.cameras = {camera};

    report.calibrations = {calibration};

    const nlohmann::json json_report = report;
    const auto restored = json_report.get<PlanarIntrinsicsReport>();

    EXPECT_EQ(restored.session.id, report.session.id);
    ASSERT_EQ(restored.calibrations.size(), 1U);
    const auto& restored_cal = restored.calibrations.front();
    EXPECT_EQ(restored_cal.algorithm, "planar");
    ASSERT_EQ(restored_cal.cameras.size(), 1U);
    const auto& restored_cam = restored_cal.cameras.front();
    EXPECT_EQ(restored_cam.camera_id, "cam0");
    EXPECT_TRUE(restored_cam.image_size.has_value());
    EXPECT_EQ((*restored_cam.image_size)[0], 640);
    EXPECT_EQ((*restored_cam.image_size)[1], 480);
    EXPECT_NEAR(restored_cam.result.reprojection_rms_px, 0.25, 1e-9);
    ASSERT_EQ(restored_cam.result.per_view.size(), 1U);
    EXPECT_TRUE(restored_cam.result.per_view.front().used_in_linear_stage);
}

TEST(PlanarIntrinsicCalibrationFacadeTest, CalibratesSyntheticData) {
    RNG rng(7);

    Camera<BrownConradyd> cam_gt;
    cam_gt.kmtx.fx = 900.0;
    cam_gt.kmtx.fy = 880.0;
    cam_gt.kmtx.cx = 640.0;
    cam_gt.kmtx.cy = 360.0;
    cam_gt.kmtx.skew = 0.0;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d b_se3_t = Eigen::Translation3d(0.0, 0.0, 2.0) * Eigen::Isometry3d::Identity();

    SimulatedHandEye sim{g_se3_c, b_se3_t, cam_gt};
    sim.make_sequence(6, rng);
    sim.make_target_grid(6, 9, 0.03);
    sim.render_pixels();

    PlanarDetections detections;
    detections.feature_type = "planar";
    detections.image_directory = "synthetic";
    detections.algo_version = "v1";
    detections.params_hash = "hash";
    detections.images.reserve(sim.observations.size());

    for (std::size_t idx = 0; idx < sim.observations.size(); ++idx) {
        const auto& observation = sim.observations[idx];
        PlanarImageDetections image;
        image.file = "view" + std::to_string(idx) + ".png";
        image.points.reserve(observation.view.size());
        for (std::size_t j = 0; j < observation.view.size(); ++j) {
            const auto& ob = observation.view[j];
            PlanarTargetPoint pt;
            pt.x = ob.image_uv.x();
            pt.y = ob.image_uv.y();
            pt.id = static_cast<int>(j);
            pt.local_x = ob.object_xy.x();
            pt.local_y = ob.object_xy.y();
            pt.local_z = 0.0;
            image.points.push_back(pt);
        }
        detections.images.push_back(std::move(image));
    }

    IntrinsicCalibrationConfig cfg;
    cfg.algorithm = "planar";
    cfg.options.min_corners_per_view = 20;
    cfg.options.refine = true;
    cfg.options.point_scale = 1.0;
    cfg.options.auto_center = true;

    CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam0";
    cam_cfg.model = "pinhole_brown_conrady";
    cam_cfg.image_size = std::array<int, 2>{1280, 720};
    cfg.cameras = {cam_cfg};

    PlanarIntrinsicCalibrationFacade facade;
    auto result =
        facade.calibrate(cfg, cam_cfg, detections, std::filesystem::path{"synthetic.json"});

    EXPECT_TRUE(result.outputs.refine_result.success);
    const auto& estimated = result.outputs.refine_result.camera.kmtx;
    EXPECT_NEAR(estimated.fx, cam_gt.kmtx.fx, 5.0);
    EXPECT_NEAR(estimated.fy, cam_gt.kmtx.fy, 5.0);
    EXPECT_NEAR(estimated.cx, cam_gt.kmtx.cx, 5.0);
    EXPECT_NEAR(estimated.cy, cam_gt.kmtx.cy, 5.0);

    const nlohmann::json report_json = result.report;
    ASSERT_TRUE(report_json.contains("calibrations"));
    const auto& cameras_json =
        report_json.at("calibrations")[0].at("cameras")[0].at("initial_guess");
    EXPECT_EQ(cameras_json.at("used_view_indices").size(),
              result.outputs.linear_view_indices.size());
}

TEST(PlanarIntrinsicConfig, LoadConfigParsesOptions) {
    const auto tests_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto config_path =
        tests_dir.parent_path() / "apps" / "examples" / "planar_intrinsics_config.json";
    auto cfg = load_calibration_config(config_path);

    EXPECT_FALSE(cfg.cameras.empty());
    EXPECT_EQ(cfg.cameras[0].camera_id, "cam0");
    EXPECT_TRUE(cfg.options.homography_ransac.has_value());
    if (cfg.options.homography_ransac.has_value()) {
        EXPECT_EQ(cfg.options.homography_ransac->min_inliers, 50);
    }
}
