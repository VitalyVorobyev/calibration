set(SOURCES
    bundle.cpp
    extrinsics.cpp
    handeye.cpp
    homography.cpp
    intrinsics.cpp
    intrinsicssemidlt.cpp
    planarpose.cpp
)

add_library(calib_estimation_optim STATIC ${SOURCES})

set(ESTIMATION_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_include_directories(calib_estimation_optim
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${ESTIMATION_ROOT}
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(calib_estimation_optim
    PUBLIC
        calib_models
        calib_estimation_linear
        Ceres::ceres
    PRIVATE
        calib_estimation_common)

calib_set_default_warnings(calib_estimation_optim)
add_library(calib::estimation_optim ALIAS calib_estimation_optim)

install(TARGETS calib_estimation_optim
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
