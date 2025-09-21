#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "calib/pipeline/planar_intrinsics.h"
#include "calib/datasets/planar.h"
#include "calib/reports/planar_intrinsics.h"
#include "utils.h"

using namespace calib;
using namespace calib::planar;

TEST(PlanarIntrinsicsUtils, DeterminePointCenterOverrideWins) {
    PlanarDetections detections;
    IntrinsicCalibrationOptions opts;
    opts.auto_center = false;
    opts.point_center_override = std::array<double, 2>{10.0, -5.0};

    auto center = determine_point_center(detections, opts);
    EXPECT_DOUBLE_EQ(center[0], 10.0);
    EXPECT_DOUBLE_EQ(center[1], -5.0);
}

TEST(PlanarIntrinsicsUtils, DeterminePointCenterAutoFromPoints) {
    PlanarDetections detections;
    PlanarImageDetections img;
    img.points.push_back(PlanarTargetPoint{.local_x = -2.0, .local_y = 1.0});
    img.points.push_back(PlanarTargetPoint{.local_x = 4.0, .local_y = -3.0});
    detections.images.push_back(img);

    IntrinsicCalibrationOptions opts;
    opts.auto_center = true;
    opts.point_scale = 1.0;

    auto center = determine_point_center(detections, opts);
    EXPECT_DOUBLE_EQ(center[0], 1.0);
    EXPECT_DOUBLE_EQ(center[1], -1.0);
}

TEST(PlanarIntrinsicsUtils, CollectPlanarViewsRespectsThresholdAndScaling) {
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
    opts.point_scale = 2.0;

    std::vector<ActiveView> views;
    const std::array<double, 2> center{1.0, 1.0};
    auto planar_views = collect_planar_views(detections, opts, center, views);

    ASSERT_EQ(planar_views.size(), 1);
    ASSERT_EQ(views.size(), 1);
    EXPECT_EQ(views[0].source_image, "view0.png");
    EXPECT_EQ(views[0].corner_count, 3u);

    const auto& view = planar_views[0];
    ASSERT_EQ(view.size(), 3u);
    EXPECT_DOUBLE_EQ(view[0].object_xy.x(), 0.0);
    EXPECT_DOUBLE_EQ(view[0].object_xy.y(), 0.0);
    EXPECT_DOUBLE_EQ(view[1].object_xy.x(), 2.0);
    EXPECT_DOUBLE_EQ(view[1].object_xy.y(), 0.0);
    EXPECT_DOUBLE_EQ(view[2].object_xy.x(), 0.0);
    EXPECT_DOUBLE_EQ(view[2].object_xy.y(), 4.0);
}

TEST(PlanarIntrinsicsUtils, BuildRansacOptionsMatchesConfig) {
    HomographyRansacConfig cfg;
    cfg.max_iters = 50;
    cfg.thresh = 0.25;
    cfg.min_inliers = 12;
    cfg.confidence = 0.85;

    const auto opts = build_ransac_options(cfg);
    EXPECT_EQ(opts.max_iters, cfg.max_iters);
    EXPECT_DOUBLE_EQ(opts.thresh, cfg.thresh);
    EXPECT_EQ(opts.min_inliers, cfg.min_inliers);
    EXPECT_DOUBLE_EQ(opts.confidence, cfg.confidence);
    EXPECT_EQ(opts.seed, RansacOptions{}.seed);
    EXPECT_TRUE(opts.refit_on_inliers);
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
        "images": [
            {
                "count": 2,
                "file": "view0.png",
                "points": [
                    {"x": 10.0, "y": 20.0, "id": 5, "local_x": 1.0, "local_y": 2.0, "local_z": 0.1},
                    {"x": 30.0, "y": 40.0, "id": 6, "local_x": 3.0, "local_y": 4.0, "local_z": 0.2}
                ]
            }
        ]
    })";
    file.close();

    const auto detections = load_planar_dataset(json_path);
    EXPECT_EQ(detections.image_directory, "imgs");
    EXPECT_EQ(detections.feature_type, "planar");
    EXPECT_EQ(detections.algo_version, "v2");
    EXPECT_EQ(detections.params_hash, "hash123");
    ASSERT_EQ(detections.images.size(), 1u);

    const auto& image = detections.images.front();
    EXPECT_EQ(image.file, "view0.png");
    EXPECT_EQ(image.count, 2);
    ASSERT_EQ(image.points.size(), 2u);

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

    EXPECT_THROW({ (void)load_planar_dataset(json_path); }, std::runtime_error);
}

TEST(PlanarIntrinsicsUtils, PrintCalibrationSummaryIncludesKeyData) {
    CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam_test";

    CalibrationOutputs outputs;
    outputs.point_scale = 2.0;
    outputs.point_center = {1.5, -0.5};
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
    PlanarCalibrationConfig cfg;
    cfg.session.id = "session";
    cfg.session.description = "test";
    cfg.algorithm = "planar";
    cfg.options.fixed_distortion_indices = {0, 2};
    cfg.options.fixed_distortion_values = {0.1, 0.05};
    cfg.options.num_radial = 3;
    cfg.options.min_corners_per_view = 20;

    CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam0";
    cam_cfg.model = "pinhole_brown_conrady";
    cam_cfg.image_size = std::array<int, 2>{640, 480};

    PlanarDetections detections;
    detections.image_directory = "imgs";
    detections.algo_version = "v1";
    detections.params_hash = "hash";

    CalibrationOutputs outputs;
    outputs.total_input_views = 5;
    outputs.accepted_views = 3;
    outputs.min_corner_threshold = 20;
    outputs.point_scale = 1.5;
    outputs.point_center = {0.2, -0.3};
    outputs.total_points_used = 60;
    outputs.invalid_k_warnings = 1;
    outputs.pose_warnings = 2;
    outputs.linear_kmtx = CameraMatrix{800.0, 810.0, 320.0, 240.0, 0.5};
    outputs.linear_view_indices = {0, 2};
    outputs.active_views = {ActiveView{"view0.png", 25}, ActiveView{"view1.png", 35}};

    outputs.refine_result.camera = PinholeCamera<BrownConradyd>(
        CameraMatrix{805.0, 815.0, 322.0, 241.0, 0.4}, Eigen::VectorXd::Constant(5, 0.01));
    outputs.refine_result.view_errors = {0.5, 0.7};

    const auto json =
        build_planar_intrinsics_report(cfg, cam_cfg, detections, outputs, std::filesystem::path{"feat.json"});

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
        image.count = static_cast<int>(observation.view.size());
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

    PlanarCalibrationConfig cfg;
    cfg.session.id = "session";
    cfg.session.description = "synthetic";
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

    ASSERT_TRUE(result.report.contains("calibrations"));
    const auto& cameras_json =
        result.report.at("calibrations")[0].at("cameras")[0].at("initial_guess");
    EXPECT_EQ(cameras_json.at("used_view_indices").size(),
              result.outputs.linear_view_indices.size());
}

TEST(PlanarIntrinsicConfig, LoadConfigParsesOptions) {
    const auto tests_dir = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto config_path =
        tests_dir.parent_path() / "apps" / "examples" / "planar_intrinsics_config.json";
    auto cfg = load_calibration_config(config_path);

    EXPECT_EQ(cfg.session.id, "default_planar_session");
    EXPECT_FALSE(cfg.cameras.empty());
    EXPECT_EQ(cfg.cameras[0].camera_id, "cam0");
    EXPECT_TRUE(cfg.options.homography_ransac.has_value());
    EXPECT_EQ(cfg.options.homography_ransac->min_inliers, 50);
}
