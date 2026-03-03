# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(AvalonNeuralSimulator VERSION 5040.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(avalon_neural_core avalon_neural_core.cpp)

# Executables
add_executable(avalon_test main.cpp)
target_link_libraries(avalon_test avalon_neural_core)

add_executable(avalon_benchmark benchmark.cpp)
target_link_libraries(avalon_benchmark avalon_neural_core)

add_executable(bio_stress_test bio_stress_test.cpp)
target_link_libraries(bio_stress_test avalon_neural_core)
