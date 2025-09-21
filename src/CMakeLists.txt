set(CALIB_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

set(CALIB_ESTIMATION_SOURCES
    estimation/bundle.cpp
    estimation/extrinsics.cpp
    estimation/extrinsicsdlt.cpp
    estimation/handeye.cpp
    estimation/handeyedlt.cpp
    estimation/homography.cpp
    estimation/homographyestimator.cpp
    estimation/intrinsics.cpp
    estimation/intrinsicsdlt.cpp
    estimation/intrinsicssemidlt.cpp
    estimation/planarpose.cpp
    estimation/posefromhomography.cpp
    estimation/zhang.cpp
)

set(CALIB_PIPELINE_SOURCES
    pipeline/linescan.cpp
)

set(CALIB_UTILS_SOURCES
    utils/planar_intrinsics_utils.cpp
)

add_library(calib_core INTERFACE)
target_include_directories(calib_core INTERFACE
    $<BUILD_INTERFACE:${CALIB_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(calib_core INTERFACE Eigen3::Eigen)
add_library(calib::core ALIAS calib_core)

add_library(calib_models INTERFACE)
target_include_directories(calib_models INTERFACE
    $<BUILD_INTERFACE:${CALIB_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(calib_models INTERFACE Eigen3::Eigen)
add_library(calib::models ALIAS calib_models)

add_library(calib_io INTERFACE)
target_include_directories(calib_io INTERFACE
    $<BUILD_INTERFACE:${CALIB_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(calib_io INTERFACE nlohmann_json::nlohmann_json)
add_library(calib::io ALIAS calib_io)

add_library(calib_estimation STATIC ${CALIB_ESTIMATION_SOURCES})
target_include_directories(calib_estimation
    PUBLIC
        $<BUILD_INTERFACE:${CALIB_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/estimation
        ${CMAKE_CURRENT_SOURCE_DIR}/estimation/detail)
target_link_libraries(calib_estimation
    PUBLIC
        calib_core
        calib_models
        Ceres::ceres)
calib_set_default_warnings(calib_estimation)
add_library(calib::estimation ALIAS calib_estimation)

add_library(calib_pipeline STATIC ${CALIB_PIPELINE_SOURCES})
target_include_directories(calib_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${CALIB_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/estimation
        ${CMAKE_CURRENT_SOURCE_DIR}/estimation/detail)
target_link_libraries(calib_pipeline
    PUBLIC
        calib_core
        calib_models
        calib_estimation
        Ceres::ceres)
calib_set_default_warnings(calib_pipeline)
add_library(calib::pipeline ALIAS calib_pipeline)

add_library(calib_utils STATIC ${CALIB_UTILS_SOURCES})
target_include_directories(calib_utils
    PUBLIC
        $<BUILD_INTERFACE:${CALIB_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/estimation
        ${CMAKE_CURRENT_SOURCE_DIR}/estimation/detail)
target_link_libraries(calib_utils
    PUBLIC
        calib_core
        calib_models
        calib_estimation
        calib_io
        nlohmann_json::nlohmann_json)
calib_set_default_warnings(calib_utils)
add_library(calib::utils ALIAS calib_utils)

add_library(calib INTERFACE)
target_link_libraries(calib INTERFACE
    calib_core
    calib_models
    calib_estimation
    calib_pipeline
    calib_utils
    calib_io)
add_library(calib::calib ALIAS calib)
add_library(calibration ALIAS calib)

include(CMakePackageConfigHelpers)

install(TARGETS
    calib_core
    calib_models
    calib_io
    calib_estimation
    calib_pipeline
    calib_utils
    calib
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CALIB_INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/calibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/calibConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/calib)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/calibConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/calibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/calibConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/calib)

install(EXPORT calibTargets
    FILE calibTargets.cmake
    NAMESPACE calib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/calib)
