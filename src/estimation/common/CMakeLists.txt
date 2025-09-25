add_library(calib_estimation_common INTERFACE)

target_include_directories(calib_estimation_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../detail>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../residuals>)

target_link_libraries(calib_estimation_common INTERFACE
    calib_core
    calib_models
    Ceres::ceres)

add_library(calib::estimation_common ALIAS calib_estimation_common)

install(TARGETS calib_estimation_common
    EXPORT calibTargets)
