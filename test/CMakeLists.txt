include(GoogleTest)

if(WIN32)
  add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
  add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
endif()

# Test for homography estimation
add_executable(homography_test homography_test.cpp)
target_link_libraries(homography_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
)
gtest_discover_tests(homography_test)

# Test for distortion calculation
add_executable(distortion_test distortion_test.cpp)
target_link_libraries(distortion_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
)
gtest_discover_tests(distortion_test)

# Test for camera intrinsics optimization
add_executable(intrinsics_test intrinsics_test.cpp)
target_link_libraries(intrinsics_test 
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(intrinsics_test)

# Test for planar pose estimation
add_executable(planarpose_test planarpose_test.cpp)
target_link_libraries(planarpose_test PRIVATE
    calibration
    GTest::gtest_main
    GTest::gmock_main
    Eigen3::Eigen
    Ceres::ceres
)
gtest_discover_tests(planarpose_test)

# Test for full camera calibration
add_executable(calib_test calib_test.cpp)
target_link_libraries(calib_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(calib_test)

# Test for extrinsic pose optimization
add_executable(extrinsics_test extrinsics_test.cpp)
target_link_libraries(extrinsics_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(extrinsics_test)

# Test for joint intrinsics/extrinsics optimization
add_executable(joint_test joint_test.cpp)
target_link_libraries(joint_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(joint_test)

# Test for hand-eye calibration
add_executable(handeye_test handeye_test.cpp)
target_link_libraries(handeye_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(handeye_test)

# Test for line scan calibration
add_executable(linescan_test linescan_test.cpp)
target_link_libraries(linescan_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(linescan_test)

# Test for automatic JSON serialization
add_executable(json_test json_test.cpp)
target_link_libraries(json_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
    PRIVATE nlohmann_json::nlohmann_json
)
gtest_discover_tests(json_test)

# Test for bundle optimisation
add_executable(bundle_test bundle_test.cpp)
target_link_libraries(bundle_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(bundle_test)

# Test for Scheimpflug camera projection
add_executable(scheimpflug_test scheimpflug_test.cpp)
target_link_libraries(scheimpflug_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
)
gtest_discover_tests(scheimpflug_test)

# Tests for Scheimpflug hand-eye and bundle adjustment
add_executable(scheimpflug_bundle_test scheimpflug_bundle_test.cpp)
target_link_libraries(scheimpflug_bundle_test
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
    PRIVATE Ceres::ceres
)
gtest_discover_tests(scheimpflug_bundle_test)
