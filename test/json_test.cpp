#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "calib/intrinsics.h"
#include "calib/json.h"
#include "calib/serialization.h"

using namespace calib;

TEST(JsonReflection, CameraMatrixRoundTrip) {
    CameraMatrix cam{100.0, 110.0, 10.0, 20.0, 1.5};

    nlohmann::json j = cam;
    CameraMatrix parsed = j.get<CameraMatrix>();

    EXPECT_DOUBLE_EQ(parsed.fx, cam.fx);
    EXPECT_DOUBLE_EQ(parsed.fy, cam.fy);
    EXPECT_DOUBLE_EQ(parsed.cx, cam.cx);
    EXPECT_DOUBLE_EQ(parsed.cy, cam.cy);
    EXPECT_DOUBLE_EQ(parsed.skew, cam.skew);
}

TEST(JsonSerialization, ObservationRoundTrip) {
    Observation<double> o{1.0,2.0,3.0,4.0};
    nlohmann::json j = o;
    auto o2 = j.get<Observation<double>>();
    EXPECT_DOUBLE_EQ(o2.x, o.x);
    EXPECT_DOUBLE_EQ(o2.y, o.y);
    EXPECT_DOUBLE_EQ(o2.u, o.u);
    EXPECT_DOUBLE_EQ(o2.v, o.v);
}

TEST(JsonSerialization, IntrinsicsResultRoundTrip) {
    IntrinsicsOptimizationResult<Camera<BrownConradyd>> res;
    res.camera.K = CameraMatrix{100,100,0,0,0};
    res.camera.distortion.coeffs = Eigen::VectorXd::Zero(5);
    res.covariance = Eigen::MatrixXd::Identity(5,5);
    res.view_errors = {0.1, 0.2};
    res.report = "ok";
    res.c_se3_t = {Eigen::Affine3d::Identity()};
    nlohmann::json j = res;
    auto r2 = j.get<IntrinsicsOptimizationResult<Camera<BrownConradyd>>>();
    EXPECT_NEAR(r2.camera.K.fx, 100, 1e-9);
    EXPECT_EQ(r2.report, "ok");
    EXPECT_EQ(r2.view_errors.size(), 2u);
}
