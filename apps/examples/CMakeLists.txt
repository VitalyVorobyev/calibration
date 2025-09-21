add_executable(calib_example_homography main.cpp)
target_link_libraries(calib_example_homography PRIVATE calib::calib)
calib_set_default_warnings(calib_example_homography)

add_executable(calib_example_planar_intrinsics planar_intrinsics.cpp)
target_link_libraries(calib_example_planar_intrinsics PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_planar_intrinsics)

install(TARGETS
    calib_example_homography
    calib_example_planar_intrinsics
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
    planar_features_schema.json
    planar_intrinsics_config.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/calib/examples)
