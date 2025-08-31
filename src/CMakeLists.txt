
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/calibration/*.h)

set(TARGET calibration)
add_library(${TARGET} STATIC ${SOURCES} ${HEADERS})
set_target_properties(${TARGET} PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${TARGET} PUBLIC Eigen3::Eigen)
target_link_libraries(${TARGET} PUBLIC Ceres::ceres)
target_link_libraries(${TARGET} PUBLIC nlohmann_json::nlohmann_json)

# set max warning level if unix, warnings as errors
if(UNIX)
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if (MSVC)
    # Only for cl.exe:
    target_compile_options(${TARGET} PUBLIC /Zc:__cplusplus /bigobj)
endif()

# install
install(TARGETS ${TARGET}
        EXPORT vitavision-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include)
