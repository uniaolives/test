cmake_minimum_required(VERSION 3.16)
project(arkhen-core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
add_library(arkhe_cpp SHARED
    cpp/src/kernel.cpp
    cpp/src/nexus_node.cpp
)

target_include_directories(arkhe_cpp PUBLIC
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Main executable
add_executable(arkhen_node cpp/src/main.cpp)
target_link_libraries(arkhen_node arkhe_cpp)

# Link to Rust library
target_link_libraries(arkhen_node ${CMAKE_BINARY_DIR}/libarkhen_rust.so)

# Python bindings with pybind11
execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND11_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "-I" "" PYBIND11_INCLUDES_CLEAN ${PYBIND11_INCLUDES})
string(REPLACE " " ";" PYBIND11_INCLUDES_LIST ${PYBIND11_INCLUDES_CLEAN})

find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(arkhe_core cpp/src/python_bindings.cpp)
else()
    # Manual fallback
    add_library(arkhe_core MODULE cpp/src/python_bindings.cpp)
    set_target_properties(arkhe_core PROPERTIES PREFIX "" OUTPUT_NAME "arkhe_core" SUFFIX ".so")
    target_include_directories(arkhe_core PRIVATE ${PYBIND11_INCLUDES_LIST})
endif()
target_link_libraries(arkhe_core PRIVATE arkhe_cpp)

# Tests
enable_testing()
# Tests use the internal headers
add_executable(test_topology cpp/tests/test_topology.cpp)
target_include_directories(test_topology PRIVATE ${CMAKE_SOURCE_DIR}/cpp/tests)
target_link_libraries(test_topology arkhe_cpp)
add_test(NAME topology_test COMMAND test_topology)
