cmake_minimum_required(VERSION 3.14)
project(calibration CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# Optional: Find Doxygen for documentation generation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    # Add documentation target
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen found: documentation target 'doc' available")
else()
    message(STATUS "Doxygen not found: documentation will not be built")
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(test)
add_subdirectory(app)
