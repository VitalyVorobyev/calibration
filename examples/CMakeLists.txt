
add_executable(homography main.cpp)
target_link_libraries(homography PRIVATE calibration)

add_executable(planar_intrinsics planar_intrinsics.cpp)
target_link_libraries(planar_intrinsics PRIVATE calibration CLI11::CLI11)

# install executables
install(TARGETS homography planar_intrinsics
    EXPORT calibrationTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
