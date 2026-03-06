cmake_minimum_required(VERSION 3.16)
project(arkhen-core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(pybind11 REQUIRED)

# Libraries
add_library(arkhe_cpp SHARED
    cpp/src/kernel.cpp
)

target_include_directories(arkhe_cpp PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Python Module
pybind11_add_module(arkhe_core
    python/bindings/arkhe_py.cpp
)
target_link_libraries(arkhe_core PRIVATE arkhe_cpp)

# Main executable
add_executable(arkhen_node cpp/src/main.cpp)
target_link_libraries(arkhen_node arkhe_cpp)
target_link_libraries(arkhen_node ${CMAKE_BINARY_DIR}/libarkhen_rust.so)
