set(SOURCES
    planar_intrinsics_utils.cpp
)

add_library(calib_utils STATIC ${SOURCES})

target_include_directories(calib_utils
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../estimation
        ${CMAKE_CURRENT_SOURCE_DIR}/../estimation/detail)

target_link_libraries(calib_utils
    PUBLIC
        calib_core
        calib_models
        calib_estimation
        calib_io
        nlohmann_json::nlohmann_json)

calib_set_default_warnings(calib_utils)
add_library(calib::utils ALIAS calib_utils)

install(TARGETS calib_utils
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
