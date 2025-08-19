
set(TARGET homography)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE calibration)

# install executable
install(TARGETS ${TARGET}
    EXPORT calibrationTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
