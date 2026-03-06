cmake_minimum_required(VERSION 3.16)
project(arkhen-core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
add_library(arkhe_cpp SHARED
    cpp/src/kernel.cpp
)

target_include_directories(arkhe_cpp PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Main executable
add_executable(arkhen_node cpp/src/main.cpp)
target_link_libraries(arkhen_node arkhe_cpp)

# Link to Rust library
target_link_libraries(arkhen_node ${CMAKE_BINARY_DIR}/libarkhen_rust.so)
