#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "calib/pipeline/facades/extrinsics.h"
#include "calib/pipeline/pipeline.h"
#include "calib/pipeline/stages.h"

namespace calib::pipeline {
namespace {

struct SyntheticStereoData {
    PlanarDetections reference_detections;
    PlanarDetections target_detections;
    IntrinsicCalibrationOutputs reference_intrinsics;
    IntrinsicCalibrationOutputs target_intrinsics;
    StereoPairConfig pair_config;
    std::vector<Eigen::Isometry3d> camera_poses;
    std::vector<Eigen::Isometry3d> target_poses;
};

[[nodiscard]] auto create_camera() -> PinholeCamera<BrownConradyd> {
    CameraMatrix kmtx{400.0, 400.0, 0.0, 0.0};
    Eigen::VectorXd dist = Eigen::VectorXd::Zero(5);
    return PinholeCamera<BrownConradyd>(kmtx, dist);
}

void append_view(PlanarDetections& detections, const std::string& file_name,
                 const std::vector<Eigen::Vector2d>& object_points,
                 const Eigen::Isometry3d& cam_pose, const PinholeCamera<BrownConradyd>& camera) {
    PlanarImageDetections image;
    image.file = file_name;
    int point_id = 0;
    for (const auto& xy : object_points) {
        const Eigen::Vector3d P = cam_pose * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
        const Eigen::Vector2d norm_xy(P.x() / P.z(), P.y() / P.z());
        const Eigen::Vector2d pixels = denormalize(camera.kmtx, norm_xy);

        PlanarTargetPoint pt;
        pt.id = point_id++;
        pt.local_x = xy.x();
        pt.local_y = xy.y();
        pt.local_z = 0.0;
        pt.x = pixels.x();
        pt.y = pixels.y();
        image.points.push_back(std::move(pt));
    }
    detections.images.push_back(std::move(image));
}

[[nodiscard]] auto make_synthetic_stereo_data() -> SyntheticStereoData {
    SyntheticStereoData data;

    const auto camera_model = create_camera();
    data.reference_detections.sensor_id = "cam0";
    data.target_detections.sensor_id = "cam1";

    data.camera_poses.push_back(Eigen::Isometry3d::Identity());
    data.camera_poses.push_back(Eigen::Translation3d(0.5, 0.0, 0.0) *
                                Eigen::Isometry3d::Identity());

    data.target_poses.push_back(Eigen::Translation3d(0.0, 0.0, 4.0) *
                                Eigen::Isometry3d::Identity());
    data.target_poses.push_back(Eigen::Translation3d(0.2, -0.1, 3.5) *
                                Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()));
    data.target_poses.push_back(Eigen::Translation3d(-0.1, 0.2, 4.5) *
                                Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX()));

    const std::vector<Eigen::Vector2d> object_points = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0},
                                                        {0.0, 1.0}, {0.5, 0.5}, {1.5, 0.5}};

    for (std::size_t view_idx = 0; view_idx < data.target_poses.size(); ++view_idx) {
        const auto reference_pose = data.camera_poses[0] * data.target_poses[view_idx];
        const auto target_pose = data.camera_poses[1] * data.target_poses[view_idx];

        const std::string ref_file = "ref_view" + std::to_string(view_idx) + ".json";
        const std::string tgt_file = "tgt_view" + std::to_string(view_idx) + ".json";

        append_view(data.reference_detections, ref_file, object_points, reference_pose,
                    camera_model);
        append_view(data.target_detections, tgt_file, object_points, target_pose, camera_model);

        StereoViewSelection view_cfg;
        view_cfg.reference_image = ref_file;
        view_cfg.target_image = tgt_file;
        data.pair_config.views.push_back(std::move(view_cfg));
    }

    data.pair_config.reference_sensor = data.reference_detections.sensor_id;
    data.pair_config.target_sensor = data.target_detections.sensor_id;
    data.pair_config.pair_id = "rig";
    data.pair_config.options.optimize_intrinsics = false;

    data.reference_intrinsics.refine_result.core.success = true;
    data.reference_intrinsics.refine_result.camera = camera_model;

    data.target_intrinsics.refine_result.core.success = true;
    data.target_intrinsics.refine_result.camera = camera_model;

    return data;
}

TEST(StereoCalibrationFacadeTest, CalibratesSyntheticData) {
    auto data = make_synthetic_stereo_data();

    StereoCalibrationFacade facade;
    auto run_result =
        facade.calibrate(data.pair_config, data.reference_detections, data.target_detections,
                         data.reference_intrinsics, data.target_intrinsics);

    EXPECT_TRUE(run_result.success);
    EXPECT_EQ(run_result.used_views, data.pair_config.views.size());
    ASSERT_EQ(run_result.optimization.c_se3_r.size(), 2);
    ASSERT_EQ(run_result.optimization.r_se3_t.size(), data.target_poses.size());

    const Eigen::Vector3d expected_translation = data.camera_poses[1].translation();
    EXPECT_TRUE(
        run_result.optimization.c_se3_r[1].translation().isApprox(expected_translation, 1e-2));

    for (std::size_t idx = 0; idx < data.target_poses.size(); ++idx) {
        EXPECT_TRUE(run_result.optimization.r_se3_t[idx].translation().isApprox(
            data.target_poses[idx].translation(), 1e-2));
    }
}

TEST(StereoCalibrationStageTest, ProducesArtifactsAndResults) {
    auto data = make_synthetic_stereo_data();

    PipelineContext context;
    StereoCalibrationConfig cfg;
    cfg.pairs.push_back(data.pair_config);
    context.set_stereo_config(cfg);

    context.dataset.planar_cameras.push_back(data.reference_detections);
    context.dataset.planar_cameras.push_back(data.target_detections);

    context.intrinsic_results[data.reference_detections.sensor_id] = data.reference_intrinsics;
    context.intrinsic_results[data.target_detections.sensor_id] = data.target_intrinsics;

    StereoCalibrationStage stage;
    const auto report = stage.run(context);

    EXPECT_TRUE(report.success);
    EXPECT_EQ(report.summary.at("status"), "ok");
    ASSERT_TRUE(context.stereo_results.contains(data.pair_config.pair_id));

    const auto& opt_result = context.stereo_results.at(data.pair_config.pair_id);
    const Eigen::Vector3d expected_translation = data.camera_poses[1].translation();
    EXPECT_TRUE(opt_result.c_se3_r[1].translation().isApprox(expected_translation, 1e-2));

    ASSERT_TRUE(context.artifacts.contains("stereo"));
    const auto& stereo_art = context.artifacts.at("stereo");
    ASSERT_TRUE(stereo_art.contains("pairs"));
    const auto& pair_entry = stereo_art.at("pairs").at(data.pair_config.pair_id);
    EXPECT_TRUE(pair_entry.contains("optimization"));
    EXPECT_TRUE(pair_entry.contains("initial_guess"));
}

}  // namespace
}  // namespace calib::pipeline
