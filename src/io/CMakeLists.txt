add_library(calib_io INTERFACE)
target_include_directories(calib_io INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(calib_io INTERFACE nlohmann_json::nlohmann_json)
add_library(calib::io ALIAS calib_io)

install(TARGETS calib_io
    EXPORT calibTargets)
