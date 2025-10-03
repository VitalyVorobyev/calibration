add_executable(calib_example_homography homography.cpp)
target_link_libraries(calib_example_homography PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_homography)

add_executable(calib_example_planar_intrinsics planar_intrinsics.cpp)
target_link_libraries(calib_example_planar_intrinsics PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_planar_intrinsics)

add_executable(calib_example_pipeline calibration_pipeline.cpp)
target_link_libraries(calib_example_pipeline PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_pipeline)

add_executable(calib_example_intrinsic_extrinsic intrinsic_extrinsic_pipeline.cpp)
target_link_libraries(calib_example_intrinsic_extrinsic PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_intrinsic_extrinsic)

add_executable(calib_example_bundle bundle_pipeline.cpp)
target_link_libraries(calib_example_bundle PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_bundle)

add_executable(calib_example_linescan linescan_calibration.cpp)
target_link_libraries(calib_example_linescan PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(calib_example_linescan)

install(TARGETS
    calib_example_homography
    calib_example_planar_intrinsics
    calib_example_pipeline
    calib_example_intrinsic_extrinsic
    calib_example_bundle
    calib_example_linescan
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
    planar_features_schema.json
    planar_intrinsics_config.json
    ${PROJECT_SOURCE_DIR}/apps/examples/intrinsic_extrinsic_input.json
    ${PROJECT_SOURCE_DIR}/apps/examples/intrinsic_extrinsic_multicam_input.json
    ${PROJECT_SOURCE_DIR}/apps/examples/bundle_pipeline_input.json
    ${PROJECT_SOURCE_DIR}/apps/examples/linescan_input.json
    ${PROJECT_SOURCE_DIR}/apps/examples/homography_input.json
    ${PROJECT_SOURCE_DIR}/schemas/calib_dataset.schema.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/calib/examples)
