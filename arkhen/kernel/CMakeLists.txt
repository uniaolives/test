cmake_minimum_required(VERSION 3.14)
project(arkhe_kernel)
set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

add_library(arkhe_kernel SHARED
    src/quantum_history.cpp
    src/ffi.cpp
)

target_include_directories(arkhe_kernel PUBLIC include)
target_link_libraries(arkhe_kernel Eigen3::Eigen)

add_executable(test_quantum_history src/test_quantum_history.cpp)
target_link_libraries(test_quantum_history arkhe_kernel Eigen3::Eigen)
