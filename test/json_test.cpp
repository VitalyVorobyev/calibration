#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <calibration/intrinsics.h>
#include <calibration/json.h>
#include <calibration/serialization.h>

using vitavision::CameraMatrix;

TEST(JsonReflection, CameraMatrixRoundTrip) {
    CameraMatrix cam{100.0, 110.0, 10.0, 20.0};

    nlohmann::json j = cam;
    CameraMatrix parsed = j.get<CameraMatrix>();

    EXPECT_DOUBLE_EQ(parsed.fx, cam.fx);
    EXPECT_DOUBLE_EQ(parsed.fy, cam.fy);
    EXPECT_DOUBLE_EQ(parsed.cx, cam.cx);
    EXPECT_DOUBLE_EQ(parsed.cy, cam.cy);
}

TEST(JsonSerialization, ObservationRoundTrip) {
    vitavision::Observation<double> o{1.0,2.0,3.0,4.0};
    nlohmann::json j = o;
    auto o2 = j.get<vitavision::Observation<double>>();
    EXPECT_DOUBLE_EQ(o2.x, o.x);
    EXPECT_DOUBLE_EQ(o2.y, o.y);
    EXPECT_DOUBLE_EQ(o2.u, o.u);
    EXPECT_DOUBLE_EQ(o2.v, o.v);
}

TEST(JsonSerialization, IntrinsicResultRoundTrip) {
    vitavision::IntrinsicOptimizationResult res;
    res.camera.K = vitavision::CameraMatrix{100,100,0,0};
    res.covariance = Eigen::Matrix4d::Identity();
    res.reprojection_error = 0.5;
    res.summary = "ok";
    nlohmann::json j = res;
    auto r2 = j.get<vitavision::IntrinsicOptimizationResult>();
    EXPECT_NEAR(r2.camera.K.fx, 100, 1e-9);
    EXPECT_EQ(r2.summary, "ok");
    EXPECT_NEAR(r2.covariance(0,0), 1.0, 1e-9);
}
