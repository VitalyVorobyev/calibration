add_library(calib_core INTERFACE)
target_include_directories(calib_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(calib_core INTERFACE Eigen3::Eigen)
add_library(calib::core ALIAS calib_core)

install(TARGETS calib_core
    EXPORT calibTargets)
