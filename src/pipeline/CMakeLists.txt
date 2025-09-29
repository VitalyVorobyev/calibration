set(SOURCES
    pipeline.cpp
    loaders.cpp
    planar_intrinsics.cpp
    extrinsics.cpp
    linescan.cpp
    handeye.cpp
    stages/intrinsic_stage.cpp
    stages/stereo_stage.cpp
    stages/handeye_stage.cpp
    stages/bundle_stage.cpp
    stages/detail/bundle_utils.cpp
    stages/detail/planar_utils.cpp
)

add_library(calib_pipeline STATIC ${SOURCES})

target_include_directories(calib_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../estimation)

target_link_libraries(calib_pipeline
    PUBLIC
        calib_models
        calib_estimation_optim
        calib_datasets
        calib_reports
        calib_io
        Ceres::ceres)

calib_set_default_warnings(calib_pipeline)
add_library(calib::pipeline ALIAS calib_pipeline)

install(TARGETS calib_pipeline
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
