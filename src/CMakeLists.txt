add_subdirectory(models)
add_subdirectory(io)
add_subdirectory(estimation)
add_subdirectory(reports)
add_subdirectory(pipeline)

add_library(calib INTERFACE)
target_link_libraries(calib INTERFACE
    calib_models
    calib_estimation
    calib_reports
    calib_pipeline
    calib_io)
add_library(calib::calib ALIAS calib)
add_library(calibration ALIAS calib)

include(CMakePackageConfigHelpers)

install(TARGETS
    calib
    EXPORT calibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
