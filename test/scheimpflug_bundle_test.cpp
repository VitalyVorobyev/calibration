#include <gtest/gtest.h>

// std
#include <numbers>
#include <vector>

#include "calib/bundle.h"
#include "calib/scheimpflug.h"
#include "utils.h"

using namespace calib;

TEST(ScheimpflugBundle, IntrinsicsWithFixedHandeye) {
    CameraMatrix K{ 100.0, 100.0, 64.0, 48.0 };
    Camera<BrownConradyd> cam(K, Eigen::VectorXd::Zero(5));
    const double taux = 0.02;
    const double tauy = -0.015;
    ScheimpflugCamera sc(cam, taux, tauy);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    g_se3_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_se3_c.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Isometry3d b_se3_t = Eigen::Isometry3d::Identity();
    b_se3_t.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1},
                                     {0.05, 0.0}, {-0.05, 0.0}, {0.0, 0.05}, {0.0, -0.05}};
    auto poses = make_circle_poses(8, 0.1, 0.3, 0.05, 0.1, 0.5);
    auto obs = make_scheimpflug_observations<BrownConradyd>({sc}, {g_se3_c}, b_se3_t, obj, poses);

    sc.tau_x += 0.01;
    sc.tau_y -= 0.01;

    Eigen::Isometry3d init_g_se3_c = g_se3_c;
    // init_g_se3_c.translation() += Eigen::Vector3d(0.01,-0.01,0.02);

    BundleOptions opts;
    opts.optimize_intrinsics = true;
    opts.optimize_target_pose = false;
    opts.optimize_hand_eye = false;
    opts.optimizer = OptimizerType::DENSE_QR;
    opts.verbose = false;

    AnyCamera anycam {std::move(sc)};
    auto res = optimize_bundle(obs, {anycam}, {init_g_se3_c}, b_se3_t, opts);
    std::cout << res.report << std::endl;

    EXPECT_LT((res.g_se3_c[0].translation() - g_se3_c.translation()).norm(), 1e-6);
    Eigen::AngleAxisd diff(res.g_se3_c[0].linear()*g_se3_c.linear().transpose());
    EXPECT_LT(diff.angle(), 1e-6);

    auto camptr = res.cameras[0].as<ScheimpflugCamera<BrownConradyd>>();
    EXPECT_NEAR(camptr->tau_x, taux, 1e-6);
    EXPECT_NEAR(camptr->tau_y, tauy, 1e-6);
}

TEST(ScheimpflugBundle, HandeyeWithFixedIntrinsics) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};
    Camera<BrownConradyd> cam(K, Eigen::VectorXd::Zero(5));
    const double taux = 0.02;
    const double tauy = -0.015;
    ScheimpflugCamera sc(cam, taux, tauy);

    Eigen::Isometry3d g_se3_c = Eigen::Isometry3d::Identity();
    g_se3_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_se3_c.translation() = Eigen::Vector3d(0.1,0.0,0.05);

    Eigen::Isometry3d b_se3_t = Eigen::Isometry3d::Identity();
    b_se3_t.translation() = Eigen::Vector3d(0.2,0.0,0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1,-0.1},{0.1,-0.1},{0.1,0.1},{-0.1,0.1},
                                     {0.05,0.0},{-0.05,0.0},{0.0,0.05},{0.0,-0.05}};
    auto poses = make_circle_poses(8, 0.1, 0.3, 0.05, 0.1, 0.5);
    auto observations = make_scheimpflug_observations<BrownConradyd>({sc}, {g_se3_c}, b_se3_t, obj, poses);
    Eigen::Isometry3d init_g_se3_c = g_se3_c;
    init_g_se3_c.translation() += Eigen::Vector3d(0.01,-0.01,0.02);

    BundleOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = false;
    opts.optimize_hand_eye = true;
    opts.verbose = false;

    AnyCamera anycam{std::move(sc)};
    auto res = optimize_bundle(observations, {anycam}, {init_g_se3_c}, b_se3_t, opts);
    EXPECT_LT((res.g_se3_c[0].translation() - g_se3_c.translation()).norm(),1e-6);
    Eigen::AngleAxisd diff(res.g_se3_c[0].linear()*g_se3_c.linear().transpose());
    EXPECT_LT(diff.angle(),1e-6);

    auto camptr = res.cameras[0].as<ScheimpflugCamera<BrownConradyd>>();
    EXPECT_NEAR(camptr->tau_x, taux, 1e-6);
    EXPECT_NEAR(camptr->tau_y, tauy, 1e-6);
}
