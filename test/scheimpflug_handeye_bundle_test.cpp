#include <gtest/gtest.h>
#include <numbers>
#include <vector>
#include "calib/handeye.h"
#include "calib/bundle.h"
#include "calib/scheimpflug.h"

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
    auto c_T_t = g_T_c.inverse() * b_T_g.inverse() * b_T_t;
    return c_T_t;
}

TEST(ScheimpflugHandEye, ReprojectionRefinement) {
    CameraMatrix K{ 100.0, 100.0, 64.0, 48.0 };
    Camera cam(K, Eigen::VectorXd::Zero(5));
    const double taux = 0.02;
    const double tauy = -0.015;
    ScheimpflugCamera sc(cam, taux, tauy);

    Eigen::Affine3d g_T_c = Eigen::Affine3d::Identity();
    g_T_c.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
    g_T_c.translation() = Eigen::Vector3d(0.1, 0.0, 0.05);

    Eigen::Affine3d b_T_t = Eigen::Affine3d::Identity();
    b_T_t.translation() = Eigen::Vector3d(0.2, 0.0, 0.0);

    std::vector<Eigen::Vector2d> obj{{-0.1, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {-0.1, 0.1},
                                     {0.5, 0.5}, {-1.0, -1.0}, {2.0, 2.0}, {2.5, 0.5}};
    std::vector<Eigen::Affine3d> b_T_g; std::vector<PlanarView> obs;
    for (int i = 0; i < 8; ++i) {
        double angle = i * std::numbers::pi/4.0;
        Eigen::Affine3d btg = Eigen::Affine3d::Identity();
        btg.translation() = Eigen::Vector3d(
            0.1*std::cos(angle),
            0.1*std::sin(angle),
            0.3+0.05*i
        );
        btg.linear() = Eigen::AngleAxisd(
            0.1 * i,
            Eigen::Vector3d(
                std::cos(angle),
                std::sin(angle),
                0.5
            ).normalized()
        ).toRotationMatrix();

        b_T_g.push_back(btg);
        Eigen::Affine3d c_T_t = compute_camera_T_target(b_T_t, g_T_c, btg);
        std::vector<Eigen::Vector2d> img; img.reserve(obj.size());
        for (const auto& xy: obj){
            Eigen::Vector3d P(xy.x(),xy.y(),0); P = c_T_t * P;
            img.push_back(sc.project(P));
        }
        obs.push_back(make_view(obj,img));
    }
    Intrinsics init_intr{K.fx, K.fy, K.cx, K.cy, 0, 0, 0, 0, 0};
    init_intr.tau_x = taux + 0.01;
    init_intr.tau_y = tauy - 0.01;
    Eigen::Affine3d init_g_T_c = g_T_c;
    init_g_T_c.translation() += Eigen::Vector3d(0.01,-0.01,0.02);

    ReprojRefineOptions opts;
    opts.refine_intrinsics = true;
    opts.refine_distortion = false;
    opts.use_distortion = false;
    opts.verbose = false;

    auto res = refine_hand_eye_reprojection_scheimpflug(b_T_g, obs, init_intr, init_g_T_c, opts);
    EXPECT_LT((res.g_T_r.translation() - g_T_c.translation()).norm(), 1e-3);
    Eigen::AngleAxisd diff(res.g_T_r.linear() * g_T_c.linear().transpose());
    EXPECT_LT(diff.angle(), 1e-3);
    EXPECT_NEAR(res.intr.tau_x, taux, 1e-3);
    EXPECT_NEAR(res.intr.tau_y, tauy, 1e-3);
}

TEST(ScheimpflugBundle, SingleCamera) {
    CameraMatrix K{100.0, 100.0, 64.0, 48.0};
    Camera cam(K, Eigen::VectorXd::Zero(5));
    const double taux = 0.02;
    const double tauy = -0.015;
    ScheimpflugCamera sc(cam, taux, tauy);

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
        Eigen::Affine3d btg = Eigen::Affine3d::Identity();
        btg.translation() = Eigen::Vector3d(0.1*std::cos(angle),0.1*std::sin(angle),0.3+0.05*i);
        btg.linear() = Eigen::AngleAxisd(0.1*i, Eigen::Vector3d(std::cos(angle),std::sin(angle),0.5).normalized()).toRotationMatrix();
        Eigen::Affine3d c_T_t = compute_camera_T_target(b_T_t, g_T_c, btg);
        std::vector<Eigen::Vector2d> img; img.reserve(obj.size());
        for (const auto& xy: obj){
            Eigen::Vector3d P(xy.x(),xy.y(),0); P = c_T_t * P; img.push_back(sc.project(P));
        }
        observations.push_back({make_view(obj,img), btg, 0});
    }
    std::vector<ScheimpflugCamera> cams{sc};
    Eigen::Affine3d init_g_T_c = g_T_c;
    init_g_T_c.translation() += Eigen::Vector3d(0.01,-0.01,0.02);

    BundleOptions opts;
    opts.optimize_intrinsics = true;
    opts.optimize_target_pose = false;
    opts.optimize_hand_eye = true;
    opts.verbose = false;

    auto res = optimize_bundle_scheimpflug(observations, cams, {init_g_T_c}, b_T_t, opts);
    EXPECT_LT((res.g_T_c[0].translation()-g_T_c.translation()).norm(),1e-3);
    Eigen::AngleAxisd diff(res.g_T_c[0].linear()*g_T_c.linear().transpose());
    EXPECT_LT(diff.angle(),1e-3);
    EXPECT_NEAR(res.cameras[0].tau_x, taux, 1e-3);
    EXPECT_NEAR(res.cameras[0].tau_y, tauy, 1e-3);
}
