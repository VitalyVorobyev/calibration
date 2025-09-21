set(SOURCES
    linescan.cpp
    pipeline.cpp
    stages.cpp
    loaders.cpp
)

add_library(calib_pipeline STATIC ${SOURCES})

target_include_directories(calib_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../estimation
        ${CMAKE_CURRENT_SOURCE_DIR}/../estimation/detail)

target_link_libraries(calib_pipeline
    PUBLIC
        calib_core
        calib_models
        calib_estimation_optim
        Ceres::ceres)

calib_set_default_warnings(calib_pipeline)
add_library(calib::pipeline ALIAS calib_pipeline)

install(TARGETS calib_pipeline
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
