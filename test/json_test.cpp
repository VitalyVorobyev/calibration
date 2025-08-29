#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <calibration/intrinsics.h>
#include <calibration/json.h>

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
