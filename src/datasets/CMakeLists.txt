set(SOURCES
    planar.cpp
)

add_library(calib_datasets STATIC ${SOURCES})

target_include_directories(calib_datasets
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(calib_datasets
    PUBLIC
        calib_core
        calib_io
        nlohmann_json::nlohmann_json)

calib_set_default_warnings(calib_datasets)
add_library(calib::datasets ALIAS calib_datasets)

install(TARGETS calib_datasets
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
