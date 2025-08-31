#include <gtest/gtest.h>

#include "calib/bundle.h"

#include "utils.h"

using namespace calib;

TEST(OptimizeBundle, RecoversXAndIntrinsics_NoDistortion) {
    for (double skew : {0.0, 5.0}) {
        RNG rng(7);
        // GT
        Eigen::Affine3d g_T_c_gt = make_pose(
            Eigen::Vector3d(0.03, 0.00, 0.12),
            Eigen::Vector3d(0,1,0), deg2rad(8.0)
        );
        Eigen::Affine3d b_T_t_gt = make_pose(
            Eigen::Vector3d(0.5, -0.1, 0.8),
            Eigen::Vector3d(1,0,0), deg2rad(14.0)
        );

        Camera<BrownConradyd> cam_gt;
        cam_gt.K.fx = 1000;
        cam_gt.K.fy = 1005;
        cam_gt.K.cx = 640;
        cam_gt.K.cy = 360;
        cam_gt.K.skew = skew;
        cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);

        // Data
        SimulatedHandEye sim{g_T_c_gt, b_T_t_gt, cam_gt};
        sim.make_sequence(25, rng);
        sim.make_target_grid(8, 11, 0.02);
        sim.render_pixels(0.3, &rng); // 0.3 px noise
        sim.render_pixels();

        // Bad initial intrinsics and X
        Camera<BrownConradyd> cam0;
        cam0.K.fx = cam_gt.K.fx * 0.97;
        cam0.K.fy = cam_gt.K.fy * 1.03;
        cam0.K.cx = cam_gt.K.cx + 5.0;
        cam0.K.cy = cam_gt.K.cy - 4.0;
        cam0.K.skew = cam_gt.K.skew + 0.5;
        cam0.distortion.coeffs = Eigen::VectorXd::Zero(5);

        Eigen::Affine3d g_T_c0 = g_T_c_gt;
        g_T_c0.translation() += Eigen::Vector3d(-0.01, 0.006, -0.004);
        g_T_c0.linear() = axis_angle_to_R(Eigen::Vector3d(0.3,0.7,-0.2).normalized(), deg2rad(2.0)) * g_T_c0.linear();

        // Options: refine intrinsics (no distortion)
        BundleOptions opts;
        opts.optimize_intrinsics = true;
        opts.optimize_skew = (skew != 0.0);
        opts.verbose = false;

        auto result = optimize_bundle<Camera<BrownConradyd>>(sim.observations, {cam0}, {g_T_c0}, b_T_t_gt, opts);
        const auto& X = result.g_T_c[0];
        const auto& Kf = result.cameras[0].K;
        const auto& b_T_t_est = result.b_T_t;

        const auto& K_gt = cam_gt.K;
        const auto& X_gt = g_T_c_gt;
        // X close to GT
        double rot_err = rad2deg(rotation_angle(X.linear().transpose() * X_gt.linear()));
        double tr_err  = (X.translation() - X_gt.translation()).norm();
        EXPECT_LT(rot_err, 0.08);
        EXPECT_LT(tr_err,  0.003);

        // Intrinsics recovered
        EXPECT_NEAR(Kf.fx, K_gt.fx, 2.0);
        EXPECT_NEAR(Kf.fy, K_gt.fy, 2.0);
        EXPECT_NEAR(Kf.cx, K_gt.cx, 2.0);
        EXPECT_NEAR(Kf.cy, K_gt.cy, 2.0);
        EXPECT_NEAR(Kf.skew, K_gt.skew, 2.0);

        // Recovered base->target should be close
        double bt_rot = rad2deg(rotation_angle(b_T_t_est.linear().transpose() * b_T_t_gt.linear()));
        double bt_tr  = (b_T_t_est.translation() - b_T_t_gt.translation()).norm();
        EXPECT_LT(bt_rot, 0.10);
        EXPECT_LT(bt_tr,  0.004);
    }
}

TEST(ReprojectionRefine, DistortionRecoveryOptional) {
    RNG rng(137);
    // GT with distortion
    Camera<BrownConradyd> cam_gt;
    cam_gt.K.fx = 900;
    cam_gt.K.fy = 905;
    cam_gt.K.cx = 640;
    cam_gt.K.cy = 360;
    cam_gt.distortion.coeffs = Eigen::VectorXd::Zero(5);
    cam_gt.distortion.coeffs << -0.12, 0.02, 0.0005, -0.0007, 0.001;

    Eigen::Affine3d g_T_c_gt = make_pose(
        Eigen::Vector3d(0.03, 0.00, 0.12),
        Eigen::Vector3d(0,1,0), deg2rad(8.0)
    );
    Eigen::Affine3d b_T_t_gt = make_pose(
        Eigen::Vector3d(0.5, -0.1, 80),
        Eigen::Vector3d(1,0,0), deg2rad(14.0)
    );

    SimulatedHandEye sim{g_T_c_gt, b_T_t_gt, cam_gt};
    sim.make_sequence(22, rng);
    sim.make_target_grid(7, 10, 0.022);
    sim.render_pixels();

    // Start from wrong K (no distortion) and perturbed X
    Camera<BrownConradyd> cam0 = cam_gt;
    cam0.distortion.coeffs = Eigen::VectorXd::Zero(5); // start at zero

    Eigen::Affine3d X0 = g_T_c_gt;
    X0.translation() += Eigen::Vector3d(0.01, 0.006, -0.003);
    X0.linear() = axis_angle_to_R(Eigen::Vector3d(0.1,0.8,0.1).normalized(), deg2rad(2.0)) * X0.linear();

    BundleOptions opts;
    opts.optimize_intrinsics = true;
    opts.optimize_hand_eye = true;
    opts.optimize_target_pose = true;
    opts.optimizer = OptimizerType::DENSE_QR;
    opts.verbose = false;

    auto result = optimize_bundle<Camera<BrownConradyd>>(sim.observations, {cam0}, {X0}, b_T_t_gt, opts);
    const auto& X = result.g_T_c[0];
    const auto& dist = result.cameras[0].distortion.coeffs;

    // Check X quality
    double rot_err = rad2deg(rotation_angle(X.linear().transpose() * g_T_c_gt.linear()));
    double tr_err = (X.translation() - g_T_c_gt.translation()).norm();
    EXPECT_LT(rot_err, 0.1);
    EXPECT_LT(tr_err,  0.02);

    const auto& dist_gt = cam_gt.distortion.coeffs;
    EXPECT_NEAR(dist[0], dist_gt[0], 1e-5);  // k1
    EXPECT_NEAR(dist[1], dist_gt[1], 1e-5);  // k2
    EXPECT_NEAR(dist[2], dist_gt[2], 1e-5);  // p1
    EXPECT_NEAR(dist[3], dist_gt[3], 1e-5);  // p2
    EXPECT_NEAR(dist[4], dist_gt[4], 1e-5);  // k3
}

TEST(OptimizeBundle, InputValidation) {
    // Mismatched sizes should throw
    std::vector<BundleObservation> observations(2);
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};
    Camera<BrownConradyd> cam(K, Eigen::VectorXd::Zero(5));
    Eigen::Affine3d X0 = Eigen::Affine3d::Identity();
    Eigen::Affine3d init_b_T_t = Eigen::Affine3d::Identity();
    BundleOptions opts;

    EXPECT_THROW({
        optimize_bundle<Camera<BrownConradyd>>(observations, {cam, cam}, {X0}, init_b_T_t, opts);
    }, std::invalid_argument);
}

TEST(OptimizeBundle, SingleCameraHandEye) {
    CameraMatrix K{100.0,100.0,64.0,48.0};
    Camera<BrownConradyd> cam(K, Eigen::VectorXd::Zero(5));

    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity();
    g_T_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_c.translation() = Eigen::Vector3d(0.1,0.0,0.05);

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.translation() = Eigen::Vector3d(0.2,0.0,0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1,-0.1},{0.1,-0.1},{0.1,0.1},{-0.1,0.1},
                                     {0.5,0.5},{-1.0,-1.0},{2.0,2.0},{2.5,0.5}, {9, 0}};

    std::vector<Camera<BrownConradyd>> cams{cam};
    auto poses = make_circle_poses(8, 0.1, 0.3, 0.05, 0.1, 0.5);
    auto observations = make_bundle_observations(cams, {g_T_c}, b_T_t, obj, poses);
    Eigen::Affine3d init_g_T_c = g_T_c;
    init_g_T_c.translation() += Eigen::Vector3d(0.01, -0.01, 0.02);

    BundleOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = false;
    opts.optimize_hand_eye = true;

    auto res = optimize_bundle<Camera<BrownConradyd>>(observations, cams, {init_g_T_c}, b_T_t, opts);
    std::cout << res.report << std::endl;

    EXPECT_LT((res.g_T_c[0].translation() - g_T_c.translation()).norm(),1e-3);
    Eigen::AngleAxisd diff(res.g_T_c[0].linear()*g_T_c.linear().transpose());
    EXPECT_LT(diff.angle(),1e-3);
    EXPECT_LT(res.reprojection_error, 0.01);
}

TEST(OptimizeBundle, SingleCameraTargetPose) {
    CameraMatrix K{100.0,100.0,64.0,48.0};
    Camera<BrownConradyd> cam(K, Eigen::VectorXd::Zero(5));
    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity();
    g_T_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_c.translation() = Eigen::Vector3d(0.1,0.0,0.05);

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.translation() = Eigen::Vector3d(0.2,0.0,0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1,-0.1},{0.1,-0.1},{0.1,0.1},{-0.1,0.1},
                                     {0.5,0.5},{-1.0,-1.0},{2.0,2.0},{2.5,0.5}};
    std::vector<Camera<BrownConradyd>> cams{cam};
    auto poses = make_circle_poses(8, 0.1, 0.3, 0.05, 0.1, 0.5);
    auto observations = make_bundle_observations(cams, {g_T_c}, b_T_t, obj, poses);
    Eigen::Affine3d init_b_T_t = b_T_t;
    init_b_T_t.translation() += Eigen::Vector3d(0.01,-0.02,0.03);

    BundleOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = true;
    opts.optimize_hand_eye = false;

    auto res = optimize_bundle<Camera<BrownConradyd>>(observations, cams, {g_T_c}, init_b_T_t, opts);

    EXPECT_LT((res.b_T_t.translation() - b_T_t.translation()).norm(), 1e-3);
    Eigen::AngleAxisd diff(res.b_T_t.linear() * b_T_t.linear().transpose());
    EXPECT_LT(diff.angle(), 1e-3);
}

TEST(OptimizeBundle, TwoCamerasHandEyeExtrinsics) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};
    Camera<BrownConradyd> cam0(K, Eigen::VectorXd::Zero(5));
    Camera<BrownConradyd> cam1(K, Eigen::VectorXd::Zero(5));

    Eigen::Affine3d g_T_c0 = Eigen::Affine3d::Identity();
    g_T_c0.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_c0.translation() = Eigen::Vector3d(0.1,0.0,0.05);

    Eigen::Affine3d c1_T_c0 = Eigen::Affine3d::Identity();
    c1_T_c0.translation() = Eigen::Vector3d(0.05,0.0,0.0);
    c1_T_c0.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Affine3d g_T_c1 = g_T_c0 * c1_T_c0.inverse();

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.translation() = Eigen::Vector3d(0.2,0.0,0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1,-0.1},{0.1,-0.1},{0.1,0.1},{-0.1,0.1},
                                     {0.5,0.5},{-1.0,-1.0},{2.0,2.0},{2.5,0.5}};

    std::vector<Camera<BrownConradyd>> cams{cam0, cam1};
    auto poses = make_circle_poses(8, 0.1, 0.3, 0.05, 0.1, 0.5);
    auto observations = make_bundle_observations(
        cams, {g_T_c0, g_T_c1}, b_T_t, obj, poses);
    Eigen::Affine3d init_g_T_c1 = g_T_c1;
    init_g_T_c1.translation() += Eigen::Vector3d(0.01,-0.01,0.0);
    init_g_T_c1.linear() = g_T_c1.linear() * Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Affine3d init_g_T_c0 = g_T_c0;
    init_g_T_c0.translation() += Eigen::Vector3d(-0.01,0.02,-0.02);

    BundleOptions opts;
    opts.optimize_intrinsics=false;
    opts.optimize_target_pose=false;
    opts.optimize_hand_eye=true;

    auto res = optimize_bundle<Camera<BrownConradyd>>(observations, cams, {init_g_T_c0, init_g_T_c1}, b_T_t, opts);

    std::cout << "True g_T_c0 translation: " << g_T_c0.translation().transpose() << std::endl;
    std::cout << "Result g_T_c0 translation: " << res.g_T_c[0].translation().transpose() << std::endl;
    std::cout << "True g_T_c1 translation: " << g_T_c1.translation().transpose() << std::endl;
    std::cout << "Result g_T_c1 translation: " << res.g_T_c[1].translation().transpose() << std::endl;

    EXPECT_LT((res.g_T_c[0].translation() - g_T_c0.translation()).norm(),1e-3);
    Eigen::AngleAxisd diff(res.g_T_c[0].linear() * g_T_c0.linear().transpose());
    EXPECT_LT(diff.angle(),1e-3);

    EXPECT_LT((res.g_T_c[1].translation() - g_T_c1.translation()).norm(),1e-3);
    Eigen::AngleAxisd diff2(res.g_T_c[1].linear() * g_T_c1.linear().transpose());
    EXPECT_LT(diff2.angle(), 1e-3);
}
