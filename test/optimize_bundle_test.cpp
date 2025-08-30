#include <gtest/gtest.h>

// std
#include <numbers>
#include <cmath>

#include "calib/bundle.h"

using namespace calib;

static PlanarView make_view(const std::vector<Eigen::Vector2d>& obj,
                            const std::vector<Eigen::Vector2d>& img) {
    PlanarView view(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) {
        view[i].object_xy = obj[i];
        view[i].image_uv = img[i];
    }
    return view;
}

static Eigen::Affine3d compute_camera_T_target(
    const Eigen::Affine3d& b_T_t,
    const Eigen::Affine3d& g_T_c,
    const Eigen::Affine3d& b_T_g) {
    Eigen::Affine3d c_T_t = g_T_c.inverse() * b_T_g.inverse() * b_T_t;
    return c_T_t;
}

TEST(OptimizeBundle, SingleCameraHandEye) {
    CameraMatrix K{100.0,100.0,64.0,48.0};
    Camera<DualDistortion> cam(K, Eigen::VectorXd::Zero(5));

    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity();
    g_T_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_c.translation() = Eigen::Vector3d(0.1,0.0,0.05);

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.translation() = Eigen::Vector3d(0.2,0.0,0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1,-0.1},{0.1,-0.1},{0.1,0.1},{-0.1,0.1},
                                     {0.5,0.5},{-1.0,-1.0},{2.0,2.0},{2.5,0.5}};

    std::vector<BundleObservation> observations;
    for (int i=0;i<8;++i){
        double angle = i * std::numbers::pi/4.0;
        Eigen::Affine3d b_T_g = Eigen::Affine3d::Identity();
        b_T_g.translation() = Eigen::Vector3d(0.1*std::cos(angle),0.1*std::sin(angle),0.3+0.05*i);
        Eigen::Matrix3d rot = Eigen::AngleAxisd(0.1*i, Eigen::Vector3d(std::cos(angle),std::sin(angle),0.5).normalized()).toRotationMatrix();
        b_T_g.linear() = rot;

        Eigen::Affine3d c_T_t = compute_camera_T_target(b_T_t, g_T_c, b_T_g);
        std::vector<Eigen::Vector2d> img; img.reserve(obj.size());
        for (const auto& xy: obj){
            Eigen::Vector3d P(xy.x(),xy.y(),0.0);
            P = c_T_t * P;
            Eigen::Vector2d uv = cam.project(P);
            img.push_back(uv);
        }
        observations.push_back({make_view(obj,img), b_T_g, 0});
    }

    std::vector<Camera<DualDistortion>> cams{cam};
    Eigen::Affine3d init_g_T_c = g_T_c;
    init_g_T_c.translation() += Eigen::Vector3d(0.01,-0.01,0.02);

    BundleOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = false;
    opts.optimize_hand_eye = true;

    auto res = optimize_bundle(observations, cams, {init_g_T_c}, b_T_t, opts);

    EXPECT_LT((res.g_T_c[0].translation() - g_T_c.translation()).norm(),1e-3);
    Eigen::AngleAxisd diff(res.g_T_c[0].linear()*g_T_c.linear().transpose());
    EXPECT_LT(diff.angle(),1e-3);
    EXPECT_LT(res.reprojection_error, 0.01);
}

TEST(OptimizeBundle, SingleCameraTargetPose) {
    CameraMatrix K{100.0,100.0,64.0,48.0};
    Camera<DualDistortion> cam(K, Eigen::VectorXd::Zero(5));
    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity();
    g_T_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_c.translation() = Eigen::Vector3d(0.1,0.0,0.05);

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.translation() = Eigen::Vector3d(0.2,0.0,0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1,-0.1},{0.1,-0.1},{0.1,0.1},{-0.1,0.1},
                                     {0.5,0.5},{-1.0,-1.0},{2.0,2.0},{2.5,0.5}};
    std::vector<BundleObservation> observations;
    for (int i = 0; i < 8; ++i) {
        double angle = i * std::numbers::pi/4.0;
        Eigen::Affine3d b_T_g = Eigen::Affine3d::Identity();
        b_T_g.translation() = Eigen::Vector3d(0.1*std::cos(angle),0.1*std::sin(angle),0.3+0.05*i);
        Eigen::Matrix3d rot = Eigen::AngleAxisd(0.1*i, Eigen::Vector3d(std::cos(angle),std::sin(angle),0.5).normalized()).toRotationMatrix();
        b_T_g.linear() = rot;
        Eigen::Affine3d c_T_t = compute_camera_T_target(b_T_t, g_T_c, b_T_g);
        std::vector<Eigen::Vector2d> img;
        img.reserve(obj.size());

        for (const auto& xy : obj) {
            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            P = c_T_t * P;
            img.push_back(cam.project(P));
        }
        observations.push_back({make_view(obj,img), b_T_g,0});
    }

    std::vector<Camera<DualDistortion>> cams{cam};
    Eigen::Affine3d init_b_T_t = b_T_t;
    init_b_T_t.translation() += Eigen::Vector3d(0.01,-0.02,0.03);

    BundleOptions opts;
    opts.optimize_intrinsics = false;
    opts.optimize_target_pose = true;
    opts.optimize_hand_eye = false;

    auto res = optimize_bundle(observations, cams, {g_T_c}, init_b_T_t, opts);

    EXPECT_LT((res.b_T_t.translation() - b_T_t.translation()).norm(), 1e-3);
    Eigen::AngleAxisd diff(res.b_T_t.linear() * b_T_t.linear().transpose());
    EXPECT_LT(diff.angle(), 1e-3);
}

TEST(OptimizeBundle, TwoCamerasHandEyeExtrinsics) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};
    Camera<DualDistortion> cam0(K, Eigen::VectorXd::Zero(5));
    Camera<DualDistortion> cam1(K, Eigen::VectorXd::Zero(5));

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

    std::vector<BundleObservation> observations;
    for (int i = 0; i < 8; ++i) {
        double angle = i * std::numbers::pi/4.0;
        Eigen::Affine3d b_T_g = Eigen::Affine3d::Identity();
        b_T_g.translation() = Eigen::Vector3d(0.1*std::cos(angle),0.1*std::sin(angle),0.3+0.05*i);
        Eigen::Matrix3d rot = Eigen::AngleAxisd(0.1*i, Eigen::Vector3d(std::cos(angle),std::sin(angle),0.5).normalized()).toRotationMatrix();
        b_T_g.linear() = rot;

        Eigen::Affine3d c0_T_t = compute_camera_T_target(b_T_t, g_T_c0, b_T_g);
        Eigen::Affine3d c1_T_t = compute_camera_T_target(b_T_t, g_T_c1, b_T_g);
        std::vector<Eigen::Vector2d> img0_vec;
        img0_vec.reserve(obj.size());
        for(const auto& xy:obj){
            Eigen::Vector3d P(xy.x(),xy.y(),0.0);
            P = c0_T_t * P;
            img0_vec.push_back(cam0.project(P));
        }
        observations.push_back({make_view(obj,img0_vec), b_T_g,0});
        std::vector<Eigen::Vector2d> img1_vec;
        img1_vec.reserve(obj.size());
        for(const auto& xy:obj){
            Eigen::Vector3d P(xy.x(),xy.y(),0.0);
            P = c1_T_t * P;
            img1_vec.push_back(cam1.project(P));
        }
        observations.push_back({make_view(obj,img1_vec), b_T_g,1});
    }

    std::vector<Camera<DualDistortion>> cams{cam0, cam1};
    Eigen::Affine3d init_g_T_c1 = g_T_c1;
    init_g_T_c1.translation() += Eigen::Vector3d(0.01,-0.01,0.0);
    init_g_T_c1.linear() = g_T_c1.linear() * Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    Eigen::Affine3d init_g_T_c0 = g_T_c0;
    init_g_T_c0.translation() += Eigen::Vector3d(-0.01,0.02,-0.02);

    BundleOptions opts;
    opts.optimize_intrinsics=false;
    opts.optimize_target_pose=false;
    opts.optimize_hand_eye=true;

    auto res = optimize_bundle(observations, cams, {init_g_T_c0, init_g_T_c1}, b_T_t, opts);

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
