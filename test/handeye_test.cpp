#include <gtest/gtest.h>

#include "calibration/handeye.h"

using namespace vitavision;

static PlanarView make_view(const std::vector<Eigen::Vector2d>& obj,
                            const std::vector<Eigen::Vector2d>& img) {
    return PlanarView{obj, img};
}

TEST(HandEye, SingleCameraOptimization) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    X.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Affine3d base_T_target = Eigen::Affine3d::Identity();
    base_T_target.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    std::vector<Eigen::Vector2d> obj{
        {-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1}
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

        std::vector<Eigen::Vector2d> img;
        img.reserve(obj.size());
        for (const auto& xy : obj) {
            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            Eigen::Vector3d Pc = target_T_camera.linear() * P + target_T_camera.translation();
            double u = K.fx * (Pc.x() / Pc.z()) + K.cx;
            double v = K.fy * (Pc.y() / Pc.z()) + K.cy;
            img.emplace_back(u, v);
        }
        HandEyeObservation ho{make_view(obj, img), base_T_gripper, 0};
        observations.push_back(ho);
    }

    Eigen::Affine3d initX = estimate_hand_eye_initial(base_T_gripper_list, target_T_camera_list);
    HandEyeOptions opts; opts.optimize_intrinsics = false; opts.optimize_target_pose = true;
    HandEyeResult res = calibrate_hand_eye(observations, {K}, {initX}, Eigen::Affine3d::Identity(), opts);

    EXPECT_NEAR(0.0, (res.hand_eye[0].translation() - X.translation()).norm(), 1e-3);
    EXPECT_NEAR(0.0, (res.base_T_target.translation() - base_T_target.translation()).norm(), 1e-3);
}

