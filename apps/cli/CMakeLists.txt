set(TARGET calib_cli)
add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE calib::calib CLI11::CLI11)
calib_set_default_warnings(${TARGET})

install(TARGETS ${TARGET}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
