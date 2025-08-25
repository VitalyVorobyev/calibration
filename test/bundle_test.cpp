#include <gtest/gtest.h>

#include "calibration/handeyedlt.h"
#include "calibration/handeye.h"

using namespace vitavision;

static PlanarView make_view(const std::vector<Eigen::Vector2d>& obj,
                            const std::vector<Eigen::Vector2d>& img) {
    PlanarView view(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) {
        view[i].object_xy = obj[i];
        view[i].image_uv = img[i];
    }
    return view;
}

TEST(HandEye, TsaiLenz) {
    std::vector<Eigen::Affine3d> base_T_gripper;
    std::vector<Eigen::Affine3d> camera_T_target;

    // r_T_t = r_T_g * g_T_b * b_T_t

    CameraMatrix K{100.0, 100.0, 64.0, 48.0};

    // reference -> gripper (refcam_T_gripper)
    Eigen::Affine3d g_T_r = Eigen::Affine3d::Identity();
    g_T_r.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_r.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();  // target -> base
    b_T_t.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    // Create diverse motion patterns - rotations and translations in different directions
    for (int i = 0; i < 8; ++i) {  // Increased number of poses
        Eigen::Affine3d b_T_g = Eigen::Affine3d::Identity();

        // Add translation in all 3 dimensions
        double angle = i * M_PI / 4.0;  // Rotate around in a circle
        b_T_g.translation() = Eigen::Vector3d(
            0.1 * std::cos(angle),      // X varies in a circle
            0.1 * std::sin(angle),      // Y varies in a circle
            0.3 + 0.05 * i              // Z still increases
        );

        // Add rotation around different axes
        Eigen::Matrix3d rot = Eigen::AngleAxisd(0.1 * i, Eigen::Vector3d(
            std::cos(angle),
            std::sin(angle),
            0.5).normalized()).toRotationMatrix();

        b_T_g.linear() = rot;
        base_T_gripper.push_back(b_T_g);

        Eigen::Affine3d base_T_camera = base_T_gripper * X.inverse();
        Eigen::Affine3d target_T_camera = base_T_target.inverse() * base_T_camera;
        Eigen::Affine3d camera_T_target = target_T_camera.inverse();
        camera_T_target_list.push_back(camera_T_target);

        std::vector<Eigen::Vector2d> img;
        img.reserve(obj.size());
        for (const auto& xy : obj) {
            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            P = camera_T_target * P;
            double u = K.fx * (P.x() / P.z()) + K.cx;
            double v = K.fy * (P.y() / P.z()) + K.cy;
            img.emplace_back(u, v);
        }
        HandEyeObservation ho{make_view(obj, img), base_T_gripper, 0};
        observations.push_back(ho);
    }

    Eigen::Affine3d initX = estimate_hand_eye_tsai_lenz(base_T_gripper_list, camera_T_target_list);
    std::cout << "Initial hand-eye estimate:\n" << initX.matrix() << "\n";
    std::cout << "Ground truth hand-eye:\n" << X.matrix() << "\n";

    EXPECT_NEAR(0.0, (initX.translation() - X.translation()).norm(), 0.001);
    EXPECT_NEAR(0.0, (initX.rotation() * X.rotation().inverse()).norm(), 0.01);

    Eigen::Affine3d result = estimate_hand_eye_tsai_lenz(base_T_gripper, camera_T_target);

    // Validate the result
    // ...
}

TEST(HandEye, SingleCamera_HandEyeWithFixedTarget) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};

    // reference -> gripper (refcam_T_gripper)
    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    X.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Affine3d base_T_target = Eigen::Affine3d::Identity();  // target -> base
    base_T_target.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    // need at least 8 points to fit distortions
    std::vector<Eigen::Vector2d> obj{
        {-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1},
        {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5},
        {-2.0, 1.0}, {0.5, -2.5}, {-2.5, -0.5}
    };

    std::vector<HandEyeObservation> observations;
    std::vector<Eigen::Affine3d> base_T_gripper_list;
    std::vector<Eigen::Affine3d> camera_T_target_list;

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

        Eigen::Affine3d base_T_camera = base_T_gripper * X.inverse();
        Eigen::Affine3d target_T_camera = base_T_target.inverse() * base_T_camera;
        Eigen::Affine3d camera_T_target = target_T_camera.inverse();
        camera_T_target_list.push_back(camera_T_target);

        std::vector<Eigen::Vector2d> img;
        img.reserve(obj.size());
        for (const auto& xy : obj) {
            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            P = camera_T_target * P;
            double u = K.fx * (P.x() / P.z()) + K.cx;
            double v = K.fy * (P.y() / P.z()) + K.cy;
            img.emplace_back(u, v);
        }
        HandEyeObservation ho{make_view(obj, img), base_T_gripper, 0};
        observations.push_back(ho);
    }

    Eigen::Affine3d initX = estimate_hand_eye_tsai_lenz(base_T_gripper_list, camera_T_target_list);
    std::cout << "Initial hand-eye estimate:\n" << initX.matrix() << "\n";
    std::cout << "Ground truth hand-eye:\n" << X.matrix() << "\n";

    EXPECT_NEAR(0.0, (initX.translation() - X.translation()).norm(), 0.001);
    EXPECT_NEAR(0.0, (initX.rotation() * X.rotation().inverse()).norm(), 0.01);

    // Configure options - consider fixing some parameters if needed
    HandEyeOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = false;

    HandEyeResult res = calibrate_hand_eye(
        observations, {K}, initX, {Eigen::Affine3d::Identity()}, base_T_target, opts);
    std::cout << res.summary << std::endl;

    EXPECT_NEAR(0.0, (res.hand_eye.translation() - X.translation()).norm(), 0.001);
    EXPECT_NEAR(0.0, (res.base_T_target.translation() - base_T_target.translation()).norm(), 0.001);

    // Also test rotation accuracy
    Eigen::AngleAxisd rot_diff(res.hand_eye.linear() * X.linear().transpose());
    EXPECT_NEAR(0.0, rot_diff.angle(), 0.01);
}

TEST(HandEye, SingleCamera_TargetPoseWithKnownHandEye) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};

    Eigen::Affine3d X = Eigen::Affine3d::Identity();  // gripper_T_camera
    X.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    X.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Affine3d base_T_target = Eigen::Affine3d::Identity();
    base_T_target.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    std::vector<Eigen::Vector2d> obj{
        {-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1},
        {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5},
        {-2.0, 1.0}, {0.5, -2.5}, {-2.5, -0.5}
    };

    std::vector<HandEyeObservation> observations;

    for (int i = 0; i < 8; ++i) {
        double angle = i * M_PI / 4.0;
        Eigen::Affine3d base_T_gripper = Eigen::Affine3d::Identity();
        base_T_gripper.translation() = Eigen::Vector3d(
            0.1 * std::cos(angle),
            0.1 * std::sin(angle),
            0.3 + 0.05 * i);
        Eigen::Matrix3d rot = Eigen::AngleAxisd(0.1 * i, Eigen::Vector3d(
            std::cos(angle),
            std::sin(angle),
            0.5).normalized()).toRotationMatrix();
        base_T_gripper.linear() = rot;

        Eigen::Affine3d base_T_camera = base_T_gripper * X.inverse();
        Eigen::Affine3d target_T_camera = base_T_target.inverse() * base_T_camera;
        Eigen::Affine3d camera_T_target = target_T_camera.inverse();

        std::vector<Eigen::Vector2d> img;
        img.reserve(obj.size());
        for (const auto& xy : obj) {
            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            P = camera_T_target * P;
            double u = K.fx * (P.x() / P.z()) + K.cx;
            double v = K.fy * (P.y() / P.z()) + K.cy;
            img.emplace_back(u, v);
        }
        observations.push_back({make_view(obj, img), base_T_gripper, 0});
    }

    // Initial base->target is perturbed
    Eigen::Affine3d init_base_T_target = base_T_target;
    init_base_T_target.translation() += Eigen::Vector3d(0.01, -0.01, 0.02);

    HandEyeOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = true;
    opts.optimize_hand_eye = false; // hand-eye is assumed known

    HandEyeResult res = calibrate_hand_eye(
        observations, {K}, X, {Eigen::Affine3d::Identity()}, init_base_T_target, opts);
    std::cout << res.summary << std::endl;

    EXPECT_NEAR(0.0, (res.base_T_target.translation() - base_T_target.translation()).norm(), 1e-3);
}
