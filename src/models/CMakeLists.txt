add_library(calib_models INTERFACE)
target_include_directories(calib_models INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(calib_models INTERFACE Eigen3::Eigen)
add_library(calib::models ALIAS calib_models)

install(TARGETS calib_models
    EXPORT calibTargets)
