cmake_minimum_required(VERSION 3.16)
project(calibration VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CALIB_BUILD_APPS "Build CLI calibration tools" ON)
option(CALIB_BUILD_EXAMPLES "Build example applications" ON)
option(CALIB_BUILD_TESTS "Build unit and integration tests" ON)
option(CALIB_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(CALIB_ENABLE_WERROR "Promote warnings to errors on supported compilers" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/Dependencies.cmake)
include(cmake/Warnings.cmake)

# Header-only Boost.PFR via FetchContent (no system Boost dependency)
include(FetchContent)
FetchContent_Declare(
    boost_pfr
    GIT_REPOSITORY https://github.com/boostorg/pfr.git
    GIT_TAG 2.2.0
)
FetchContent_MakeAvailable(boost_pfr)

find_package(Doxygen QUIET)
if (CALIB_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

add_subdirectory(src)

if (CALIB_BUILD_APPS)
    add_subdirectory(apps/cli)
endif()

if (CALIB_BUILD_EXAMPLES)
    add_subdirectory(apps/examples)
endif()

if (CALIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    message(STATUS "Doxygen found: documentation target 'doc' available")
else()
    message(STATUS "Doxygen not found: documentation will not be built")
endif()
