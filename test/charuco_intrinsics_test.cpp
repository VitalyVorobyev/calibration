#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include "calib/charuco_intrinsics_utils.h"

using namespace calib;
using namespace calib::charuco;

TEST(CharucoIntrinsicsUtils, DeterminePointCenterOverrideWins) {
    CharucoDetections detections;
    IntrinsicsExampleOptions opts;
    opts.auto_center = false;
    opts.point_center_override = std::array<double, 2>{10.0, -5.0};

    auto center = determine_point_center(detections, opts);
    EXPECT_DOUBLE_EQ(center[0], 10.0);
    EXPECT_DOUBLE_EQ(center[1], -5.0);
}

TEST(CharucoIntrinsicsUtils, DeterminePointCenterAutoFromPoints) {
    CharucoDetections detections;
    CharucoImageDetections img;
    img.points.push_back(CharucoPoint{.local_x = -2.0, .local_y = 1.0});
    img.points.push_back(CharucoPoint{.local_x = 4.0, .local_y = -3.0});
    detections.images.push_back(img);

    IntrinsicsExampleOptions opts;
    opts.auto_center = true;
    opts.point_scale = 1.0;

    auto center = determine_point_center(detections, opts);
    EXPECT_DOUBLE_EQ(center[0], 1.0);
    EXPECT_DOUBLE_EQ(center[1], -1.0);
}

TEST(CharucoIntrinsicsUtils, CollectPlanarViewsRespectsThresholdAndScaling) {
    CharucoDetections detections;
    CharucoImageDetections accepted;
    accepted.file = "view0.png";
    accepted.points = {CharucoPoint{.x = 100.0, .y = 200.0, .local_x = 1.0, .local_y = 1.0},
                       CharucoPoint{.x = 150.0, .y = 220.0, .local_x = 2.0, .local_y = 1.0},
                       CharucoPoint{.x = 180.0, .y = 240.0, .local_x = 1.0, .local_y = 3.0}};
    CharucoImageDetections rejected;
    rejected.file = "view1.png";
    rejected.points = {CharucoPoint{.x = 50.0, .y = 60.0, .local_x = 0.0, .local_y = 0.0},
                       CharucoPoint{.x = 55.0, .y = 70.0, .local_x = 0.5, .local_y = 0.5}};
    detections.images = {accepted, rejected};

    IntrinsicsExampleOptions opts;
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

TEST(CharucoIntrinsicsUtils, BuildOutputJsonIncludesFixedDistortionMetadata) {
    ExampleConfig cfg;
    cfg.session.id = "session";
    cfg.session.description = "test";
    cfg.algorithm = "charuco_planar";
    cfg.options.fixed_distortion_indices = {0, 2};
    cfg.options.fixed_distortion_values = {0.1, 0.05};
    cfg.options.num_radial = 3;
    cfg.options.min_corners_per_view = 20;

    CameraConfig cam_cfg;
    cam_cfg.camera_id = "cam0";
    cam_cfg.model = "pinhole_brown_conrady";
    cam_cfg.image_size = std::array<int, 2>{640, 480};

    CharucoDetections detections;
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
        build_output_json(cfg, cam_cfg, detections, outputs, std::filesystem::path{"feat.json"});

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
