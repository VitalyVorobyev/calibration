
add_executable(homography main.cpp)
target_link_libraries(homography PRIVATE calibration)

add_executable(charuco_intrinsics charuco_intrinsics.cpp)
target_link_libraries(charuco_intrinsics PRIVATE calibration CLI11::CLI11)

# install executables
install(TARGETS homography charuco_intrinsics
    EXPORT calibrationTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
