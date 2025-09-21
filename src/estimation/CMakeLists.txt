add_subdirectory(linear)
add_subdirectory(optim)

add_library(calib_estimation INTERFACE)
target_link_libraries(calib_estimation INTERFACE
    calib_estimation_linear
    calib_estimation_optim)
add_library(calib::estimation ALIAS calib_estimation)

install(TARGETS calib_estimation
    EXPORT calibTargets)
