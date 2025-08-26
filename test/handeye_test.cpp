// test_hand_eye.cpp
#include <gtest/gtest.h>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// std
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <numbers>

#include "calibration/handeye.h"

using namespace vitavision;

static inline double deg2rad(double d) { return d * std::numbers::pi / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / std::numbers::pi; }

static double rotation_angle(const Eigen::Matrix3d& R) {
    double c = (R.trace() - 1.0) * 0.5;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c); // [0,pi]
}

static Eigen::Matrix3d axis_angle_to_R(const Eigen::Vector3d& axis, double angle){
    if (angle < 1e-16) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

static Eigen::Affine3d make_pose(const Eigen::Vector3d& t, const Eigen::Vector3d& axis, double angle) {
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = axis_angle_to_R(axis, angle);
    T.translation() = t;
    return T;
}

// Compose pose A * B
static Eigen::Affine3d compose(const Eigen::Affine3d& A, const Eigen::Affine3d& B){
    Eigen::Affine3d C = Eigen::Affine3d::Identity();
    C.linear() = A.linear() * B.linear();
    C.translation() = A.linear() * B.translation() + A.translation();
    return C;
}

// Inverse
static Eigen::Affine3d inv(const Eigen::Affine3d& T){
    Eigen::Affine3d Ti = Eigen::Affine3d::Identity();
    Ti.linear() = T.linear().transpose();
    Ti.translation() = -Ti.linear() * T.translation();
    return Ti;
}

// Project with simple pinhole + optional Brown 5
static Eigen::Vector2d project_point(
    const Eigen::Vector3d& Pc, const Intrinsics& K)
{
    double x = Pc.x() / Pc.z();
    double y = Pc.y() / Pc.z();
    double xd = x, yd = y;
    if (K.use_distortion) {
        double r2 = x*x + y*y, r4 = r2*r2, r6 = r4*r2;
        double radial = 1.0 + K.k1*r2 + K.k2*r4 + K.k3*r6;
        double x_tan = 2.0*K.p1*x*y + K.p2*(r2 + 2.0*x*x);
        double y_tan = K.p1*(r2 + 2.0*y*y) + 2.0*K.p2*x*y;
        xd = radial*x + x_tan;
        yd = radial*y + y_tan;
    }
    return Eigen::Vector2d(K.fx*xd + K.cx, K.fy*yd + K.cy);
}

// RNG helpers
struct RNG final {
    std::mt19937 gen;
    explicit RNG(uint32_t seed=0xC001C0DE) : gen(seed) {}
    double uni(double a, double b) {
        std::uniform_real_distribution<double> d(a,b);
        return d(gen);
    }
    double gauss(double stddev) {
        std::normal_distribution<double> n(0.0, stddev);
        return n(gen);
    }
    Eigen::Vector3d rand_unit_axis() {
        // Marsaglia method
        double z = uni(-1.0, 1.0);
        double t = uni(0.0, 2.0 * std::numbers::pi);
        double r = std::sqrt(1.0 - z*z);
        return {r*std::cos(t), r*std::sin(t), z};
    }
};

struct SimulatedHandEye final {
    // Ground truth
    Eigen::Affine3d g_T_c_gt;  // ^gT_c
    Eigen::Affine3d b_T_t_gt;  // ^bT_t
    Intrinsics K_gt;

    // Data
    std::vector<Eigen::Affine3d> b_T_g;   // ^bT_g per frame
    std::vector<Eigen::Affine3d> c_T_t;   // ^cT_t per frame (derived)
    std::vector<Eigen::Vector3d> obj_pts; // target points in t-frame
    std::vector<PlanarView> observations;  // per frame observations

    // Build a sequence with non-degenerate motions
    void make_sequence(size_t n_frames, RNG& rng) {
        b_T_g.clear(); c_T_t.clear();
        b_T_g.reserve(n_frames);
        // Start at identity, apply random *cumulative* motions (rot >= few deg)
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        b_T_g.push_back(T);
        for (size_t k=1; k<n_frames; ++k) {
            double ang = deg2rad(rng.uni(5.0, 25.0));     // 5–25 deg steps
            Eigen::Vector3d ax = rng.rand_unit_axis();
            Eigen::Vector3d dt(rng.uni(-0.10, 0.10), rng.uni(-0.10, 0.10), rng.uni(-0.10, 0.10));
            Eigen::Affine3d d = make_pose(dt, ax, ang);
            T = compose(T, d);
            b_T_g.push_back(T);
        }
        // Derive c_T_t from chain: c_T_tk = (g_T_c)^{-1} * (b_T_gk)^{-1} * b_T_t
        for (size_t k=0; k<n_frames; ++k) {
            c_T_t.push_back( compose( inv(g_T_c_gt), compose( inv(b_T_g[k]), b_T_t_gt ) ) );
        }
    }

    // Generate grid points on Z=0 plane (target frame)
    void make_target_grid(int rows, int cols, double spacing) {
        obj_pts.clear();
        obj_pts.reserve(rows*cols);
        double x0 = -0.5*(cols-1)*spacing;
        double y0 = -0.5*(rows-1)*spacing;
        for (int r=0; r<rows; ++r) {
            for (int c=0; c<cols; ++c) {
                obj_pts.emplace_back(x0 + c*spacing, y0 + r*spacing, 0.0);
            }
        }
    }

    // Render pixels with optional Gaussian noise (pixels)
    void render_pixels(double noise_px = 0.0, RNG* rng = nullptr) {
        observations.clear(); observations.resize(c_T_t.size());
        for (size_t k=0; k < c_T_t.size(); ++k) {
            const auto& Tct = c_T_t[k];
            observations[k].reserve(obj_pts.size());
            for (const auto& P : obj_pts) {
                Eigen::Vector3d Pc = Tct.linear()*P + Tct.translation();
                Eigen::Vector2d uv = project_point(Pc, K_gt);
                if (noise_px > 0.0 && rng) {
                    uv.x() += rng->gauss(noise_px);
                    uv.y() += rng->gauss(noise_px);
                }
                observations[k].push_back({ {P.x(), P.y()}, {uv.x(), uv.y()} });
            }
        }
    }
};

// ---------- TESTS ----------

TEST(TsaiLenzAllPairsWeighted, RecoversGroundTruthWithNoise) {
    RNG rng(123);
    // Ground truth X (hand-eye)
    Eigen::Vector3d axX = rng.rand_unit_axis();
    double angX = deg2rad(12.0);
    Eigen::Affine3d X_gt = make_pose(Eigen::Vector3d(0.03, -0.02, 0.10), axX, angX);

    // Ground truth ^bT_t
    Eigen::Vector3d axT = rng.rand_unit_axis();
    double angT = deg2rad(20.0);
    Eigen::Affine3d b_T_t_gt = make_pose(Eigen::Vector3d(0.40, 0.10, 0.60), axT, angT);

    // Intrinsics (not used by linear step, but for later)
    Intrinsics K; K.fx=900; K.fy=920; K.cx=640; K.cy=360; K.use_distortion=false;

    // Sim data
    SimulatedHandEye sim{X_gt, b_T_t_gt, K};
    sim.make_sequence(/*n_frames*/ 20, rng);
    sim.make_target_grid(7, 10, 0.02);  // 7x10 grid, 20 mm spacing
    sim.render_pixels(/*noise_px*/ 0.2, &rng);

    // Build camera_T_target from sim.c_T_t
    const auto& base_T_gripper = sim.b_T_g;
    const auto& camera_T_target = sim.c_T_t;

    // Estimate with all-pairs weighted Tsai-Lenz
    Eigen::Affine3d X_est = estimate_hand_eye_tsai_lenz_allpairs_weighted(
        base_T_gripper, camera_T_target, /*min_angle_deg*/1.0);

    double rot_err = rad2deg(rotation_angle(X_est.linear().transpose() * X_gt.linear()));
    double trans_err = (X_est.translation() - X_gt.translation()).norm();

    EXPECT_LT(rot_err, 10);   // ~10 deg. TODO: is it too large?
    EXPECT_LT(trans_err, 0.005); // ~5 mm
}

TEST(TsaiLenzAllPairsWeighted, ThrowsOnDegenerateSmallMotions) {
    // All poses identical -> no valid pairs
    std::vector<Eigen::Affine3d> b_T_g(5, Eigen::Affine3d::Identity());
    std::vector<Eigen::Affine3d> c_T_t(5, Eigen::Affine3d::Identity());
    EXPECT_THROW({
        estimate_hand_eye_tsai_lenz_allpairs_weighted(b_T_g, c_T_t, /*min_angle_deg*/2.0);
    }, std::runtime_error);
}

TEST(TsaiLenzAllPairsWeighted, InvariantToBaseFrameLeftMultiply) {
    RNG rng(77);
    // Make simple ground truth and sequence
    Eigen::Affine3d X_gt = make_pose(Eigen::Vector3d(0.01, 0.02, 0.12), Eigen::Vector3d(0,0,1), deg2rad(15));
    Eigen::Affine3d b_T_t_gt = make_pose(Eigen::Vector3d(0.3, 0.2, 0.7), Eigen::Vector3d(1,0,0), deg2rad(10));
    Intrinsics K; K.fx=1000; K.fy=1000; K.cx=640; K.cy=360; K.use_distortion=false;

    SimulatedHandEye sim{X_gt, b_T_t_gt, K};
    sim.make_sequence(12, rng);
    sim.make_target_grid(6, 8, 0.03);
    sim.render_pixels(0.0, nullptr);

    auto base_T_gripper = sim.b_T_g;
    auto camera_T_target = sim.c_T_t;

    // Left-multiply base poses by a fixed transform
    Eigen::Affine3d B = make_pose(Eigen::Vector3d(0.5, -0.1, 0.2), Eigen::Vector3d(0.3,0.7,0.2).normalized(), deg2rad(25));
    std::vector<Eigen::Affine3d> base_T_gripper2 = base_T_gripper;
    for (auto& T : base_T_gripper2) T = compose(B, T);

    Eigen::Affine3d X1 = estimate_hand_eye_tsai_lenz_allpairs_weighted(base_T_gripper,  camera_T_target, 1.0);
    Eigen::Affine3d X2 = estimate_hand_eye_tsai_lenz_allpairs_weighted(base_T_gripper2, camera_T_target, 1.0);

    double rot_err = rad2deg(rotation_angle(X1.linear().transpose() * X2.linear()));
    double trans_err = (X1.translation() - X2.translation()).norm();

    EXPECT_LT(rot_err, 1e-6);
    EXPECT_LT(trans_err, 1e-9);
}

TEST(CeresAXXBRefine, ImprovesOverInitializer) {
    RNG rng(2024);
    // Ground truth
    Eigen::Affine3d X_gt = make_pose(Eigen::Vector3d(0.02, -0.01, 0.09), rng.rand_unit_axis(), deg2rad(10.0));
    Eigen::Affine3d b_T_t_gt = make_pose(Eigen::Vector3d(0.25, 0.05, 0.55), rng.rand_unit_axis(), deg2rad(18.0));
    Intrinsics K; K.fx=950; K.fy=960; K.cx=640; K.cy=360; K.use_distortion=false;

    // Data
    SimulatedHandEye sim{X_gt, b_T_t_gt, K};
    sim.make_sequence(18, rng);
    sim.make_target_grid(6, 9, 0.025);
    sim.render_pixels(0.0, nullptr);
    const auto& base_T_gripper = sim.b_T_g;
    const auto& camera_T_target = sim.c_T_t;

    // Initializer: perturb X
    Eigen::Affine3d X0 = X_gt;
    {
        Eigen::Vector3d ax = rng.rand_unit_axis();
        double dang = deg2rad(2.0);
        X0.linear() = axis_angle_to_R(ax, dang) * X0.linear();
        X0.translation() += Eigen::Vector3d(0.01, -0.005, 0.004); // ~ centimeter
    }

    double err0_rot = rad2deg(rotation_angle(X0.linear().transpose()*X_gt.linear()));
    double err0_tr  = (X0.translation() - X_gt.translation()).norm();

    RefinementOptions ro;
    ro.max_iterations = 60;
    ro.huber_delta = 1.0;
    ro.verbose = false;

    Eigen::Affine3d Xr = refine_hand_eye(base_T_gripper, camera_T_target, X0, ro);

    double err1_rot = rad2deg(rotation_angle(Xr.linear().transpose()*X_gt.linear()));
    double err1_tr  = (Xr.translation() - X_gt.translation()).norm();

    // Should improve both
    EXPECT_LT(err1_rot, err0_rot);
    EXPECT_LT(err1_tr,  err0_tr);
    EXPECT_LT(err1_rot, 0.05);     // ~0.05 deg
    EXPECT_LT(err1_tr,  0.002);    // ~2 mm
}

TEST(ReprojectionRefine, RecoversXAndIntrinsics_NoDistortion) {
    RNG rng(7);
    // GT
    Eigen::Affine3d X_gt = make_pose(
        Eigen::Vector3d(0.03, 0.00, 0.12),
        Eigen::Vector3d(0,1,0), deg2rad(8.0)
    );
    Eigen::Affine3d b_T_t_gt = make_pose(
        Eigen::Vector3d(0.5, -0.1, 0.8),
        Eigen::Vector3d(1,0,0), deg2rad(14.0)
    );

    Intrinsics K_gt {
        .fx = 1000,
        .fy = 1005,
        .cx = 640,
        .cy = 360,
        .use_distortion = false
    };

    // Data
    SimulatedHandEye sim{X_gt, b_T_t_gt, K_gt};
    sim.make_sequence(25, rng);
    sim.make_target_grid(8, 11, 0.02);
    sim.render_pixels(0.3, &rng); // 0.3 px noise

    // Bad initial intrinsics and X
    Intrinsics K0 = {
        .fx = K_gt.fx * 0.97,
        .fy = K_gt.fy * 1.03,
        .cx = K_gt.cx + 5.0,
        .cy = K_gt.cy - 4.0,
        .use_distortion = K_gt.use_distortion
    };
    Eigen::Affine3d X0 = X_gt;
    X0.translation() += Eigen::Vector3d(-0.01, 0.006, -0.004);
    X0.linear() = axis_angle_to_R(Eigen::Vector3d(0.3,0.7,-0.2).normalized(), deg2rad(2.0)) * X0.linear();

    // Options: refine intrinsics (no distortion)
    ReprojRefineOptions opts;
    opts.refine_intrinsics = true;
    opts.refine_distortion = false;
    opts.use_distortion = false;
    opts.huber_delta_px = 1.0;
    opts.max_iterations = 100;
    opts.verbose = false;

    auto result = refine_hand_eye_reprojection(sim.b_T_g, sim.observations, K0, X0, opts);
    const auto& X = result.g_T_r;
    const auto& Kf = result.intr;
    const auto& b_T_t_est = result.b_T_t;

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

    // Recovered base->target should be close
    double bt_rot = rad2deg(rotation_angle(b_T_t_est.linear().transpose() * b_T_t_gt.linear()));
    double bt_tr  = (b_T_t_est.translation() - b_T_t_gt.translation()).norm();
    EXPECT_LT(bt_rot, 0.10);
    EXPECT_LT(bt_tr,  0.004);
}

TEST(ReprojectionRefine, DistortionRecoveryOptional) {
    RNG rng(99);
    // GT with distortion
    Intrinsics K_gt; K_gt.fx=900; K_gt.fy=905; K_gt.cx=640; K_gt.cy=360;
    K_gt.k1=-0.12; K_gt.k2=0.02; K_gt.p1=0.0005; K_gt.p2=-0.0007; K_gt.k3=0.001;
    K_gt.use_distortion = true;

    Eigen::Affine3d X_gt = make_pose(Eigen::Vector3d(0.015, -0.002, 0.10), Eigen::Vector3d(0.2,0.5,0.1).normalized(), deg2rad(9.0));
    Eigen::Affine3d b_T_t_gt = make_pose(Eigen::Vector3d(0.35, 0.15, 0.65), Eigen::Vector3d(0.6,-0.1,0.2).normalized(), deg2rad(16.0));

    SimulatedHandEye sim{X_gt, b_T_t_gt, K_gt};
    sim.make_sequence(22, rng);
    sim.make_target_grid(7, 10, 0.022);
    sim.render_pixels(0.25, &rng);

    // Start from wrong K (no distortion) and perturbed X
    Intrinsics K0 = K_gt; K0.use_distortion = true;
    K0.k1 = K0.k2 = K0.k3 = K0.p1 = K0.p2 = 0.0; // start at zero
    Eigen::Affine3d X0 = X_gt;
    X0.translation() += Eigen::Vector3d(0.01, 0.006, -0.003);
    X0.linear() = axis_angle_to_R(Eigen::Vector3d(0.1,0.8,0.1).normalized(), deg2rad(2.0)) * X0.linear();

    ReprojRefineOptions opts;
    opts.refine_intrinsics = true;
    opts.refine_distortion = true;
    opts.use_distortion = true;
    opts.huber_delta_px = 1.0;
    opts.max_iterations = 120;
    opts.verbose = false;

    auto result = refine_hand_eye_reprojection(sim.b_T_g, sim.observations, K0, X0, opts);
    const auto& X = result.g_T_r;
    const auto& Kf = result.intr;

    // Check X quality
    double rot_err = rad2deg(rotation_angle(X.linear().transpose() * X_gt.linear()));
    double tr_err = (X.translation() - X_gt.translation()).norm();
    EXPECT_LT(rot_err, 2.5);
    EXPECT_LT(tr_err,  0.02);

    // Distortion parameters should move toward GT (not necessarily perfect)
    EXPECT_NEAR(Kf.k1, K_gt.k1, 0.15);
    EXPECT_NEAR(Kf.k2, K_gt.k2, 0.03);
    EXPECT_NEAR(Kf.p1, K_gt.p1, 0.001);
    EXPECT_NEAR(Kf.p2, K_gt.p2, 0.001);
    EXPECT_NEAR(Kf.k3, K_gt.k3, 0.002);
}

TEST(ReprojectionRefine, InputValidation) {
    // Mismatched sizes should throw
    std::vector<Eigen::Affine3d> base_T_gripper(3, Eigen::Affine3d::Identity());
    std::vector<PlanarView> observations(2); // wrong (should be 3)
    Intrinsics K; K.fx=1000; K.fy=1000; K.cx=640; K.cy=360; K.use_distortion=false;
    Eigen::Affine3d X0 = Eigen::Affine3d::Identity();
    ReprojRefineOptions opts;

    EXPECT_THROW({
        refine_hand_eye_reprojection(base_T_gripper, observations, K, X0, opts);
    }, std::runtime_error);
}
