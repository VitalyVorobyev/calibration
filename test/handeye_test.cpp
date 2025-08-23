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

    // need at least 8 points to fit distortions
    std::vector<Eigen::Vector2d> obj{
        {-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1},
        {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5},
        {-2.0, 1.0}, {0.5, -2.5}, {-2.5, -0.5}
    };

    std::vector<HandEyeObservation> observations;
    std::vector<Eigen::Affine3d> base_T_gripper_list;
    std::vector<Eigen::Affine3d> target_T_camera_list;

    // Create diverse motion patterns - rotations and translations in different directions
    for (int i = 0; i < 8; ++i) {  // Increased number of poses
        Eigen::Affine3d base_T_gripper = Eigen::Affine3d::Identity();

        // Add translation in all 3 dimensions
        double angle = i * M_PI / 4.0;  // Rotate around in a circle
        base_T_gripper.translation() = Eigen::Vector3d(
            0.1 * std::cos(angle),      // X varies in a circle
            0.1 * std::sin(angle),      // Y varies in a circle
            0.3 + 0.05 * i              // Z still increases
        );

        // Add rotation around different axes
        Eigen::Matrix3d rot = Eigen::AngleAxisd(0.1 * i, Eigen::Vector3d(
            std::cos(angle),
            std::sin(angle),
            0.5).normalized()).toRotationMatrix();

        base_T_gripper.linear() = rot;
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

    // Configure options - consider fixing some parameters if needed
    HandEyeOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = true;

    // Add small perturbation to initial estimate to test convergence
    initX.translation() += Eigen::Vector3d(0.01, 0.01, 0.01);
    initX.linear() = initX.linear() * Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitX()).toRotationMatrix();

    HandEyeResult res = calibrate_hand_eye(observations, {K}, initX, {}, Eigen::Affine3d::Identity(), opts);

    EXPECT_NEAR(0.0, (res.hand_eye[0].translation() - X.translation()).norm(), 0.001);
    EXPECT_NEAR(0.0, (res.base_T_target.translation() - base_T_target.translation()).norm(), 0.001);

    // Also test rotation accuracy
    Eigen::AngleAxisd rot_diff(res.hand_eye[0].linear() * X.linear().transpose());
    EXPECT_NEAR(0.0, rot_diff.angle(), 0.01);
}
