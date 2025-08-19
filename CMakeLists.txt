cmake_minimum_required(VERSION 3.14)
project(calibration CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_subdirectory(src)
add_subdirectory(examples)
