#include <gtest/gtest.h>

#include "calibration/handeye.h"
#include "calibration/camera.h"

using namespace vitavision;

static PlanarView make_view(const std::vector<Eigen::Vector2d>& obj,
                            const std::vector<Eigen::Vector2d>& img) {
    return PlanarView{obj, img};
}

TEST(HandEye, SingleCameraOptimization) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};
    Eigen::VectorXd dist(2); dist << 0.0, 0.0;

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    X.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Affine3d base_T_target = Eigen::Affine3d::Identity();
    base_T_target.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    // need at least 8 points to fit distortions
    std::vector<Eigen::Vector2d> obj{
        {-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1},
        {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5}
    };

    std::vector<HandEyeObservation> observations;
    std::vector<Eigen::Affine3d> base_T_gripper_list;
    std::vector<Eigen::Affine3d> target_T_camera_list;

    for (int i = 0; i < 4; ++i) {
        Eigen::Affine3d base_T_gripper = Eigen::Affine3d::Identity();
        base_T_gripper.translation() = Eigen::Vector3d(0.0, 0.0, 0.3 + 0.1 * i);
        base_T_gripper_list.push_back(base_T_gripper);

        Eigen::Affine3d base_T_camera = base_T_gripper * X;
        Eigen::Affine3d target_T_camera = base_T_target.inverse() * base_T_camera;
        target_T_camera_list.push_back(target_T_camera);

        Camera cam{K, dist, base_T_camera};
        std::vector<Eigen::Vector2d> img;
        img.reserve(obj.size());
        for (const auto& xy : obj) {
            Eigen::Vector3d Pw = base_T_target * Eigen::Vector3d(xy.x(), xy.y(), 0.0);
            img.emplace_back(cam.project(Pw));
        }
        HandEyeObservation ho{make_view(obj, img), base_T_gripper, 0};
        observations.push_back(ho);
    }

    Eigen::Affine3d initX = estimate_hand_eye_initial(base_T_gripper_list, target_T_camera_list);
    HandEyeOptions opts; opts.optimize_intrinsics = false; opts.optimize_target_pose = true;
    HandEyeResult res = calibrate_hand_eye(observations, {K}, initX, {}, Eigen::Affine3d::Identity(), opts);

    EXPECT_NEAR(0.0, (res.hand_eye[0].translation() - X.translation()).norm(), 0.001);
    EXPECT_NEAR(0.0, (res.base_T_target.translation() - base_T_target.translation()).norm(), 0.001);
}
