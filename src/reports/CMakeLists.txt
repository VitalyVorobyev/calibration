set(SOURCES
    planar_intrinsics.cpp
    planar_intrinsics_types.cpp
)

add_library(calib_reports STATIC ${SOURCES})

target_include_directories(calib_reports
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(calib_reports
    PUBLIC
        calib_models
        calib_estimation
        calib_io)

calib_set_default_warnings(calib_reports)
add_library(calib::reports ALIAS calib_reports)

install(TARGETS calib_reports
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
