include(GoogleTest)

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
)
gtest_discover_tests(intrinsics_test)

# Test for planar pose estimation
add_executable(planarpose_test planarpose_test.cpp)
target_link_libraries(planarpose_test 
    PRIVATE calibration
    PRIVATE GTest::gtest_main
    PRIVATE GTest::gmock_main
    PRIVATE Eigen3::Eigen
)
gtest_discover_tests(planarpose_test)
