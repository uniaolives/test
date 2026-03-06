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

# Python bindings with pybind11
# Use execute_process to get pybind11 includes if find_package fails or to be more robust in Docker
execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE PYBIND11_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE)
# execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

# Since we might not have pybind11 CMake files, we'll manually set up the module if needed,
# but usually pybind11_add_module is preferred if find_package works.
# To satisfy the reviewer's concern about find_package failing in Docker:
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(arkhe_core cpp/src/python_bindings.cpp)
else()
    # Manual fallback
    add_library(arkhe_core MODULE cpp/src/python_bindings.cpp)
    set_target_properties(arkhe_core PROPERTIES PREFIX "" OUTPUT_NAME "arkhe_core")
    target_compile_options(arkhe_core PRIVATE ${PYBIND11_INCLUDES})
endif()
target_link_libraries(arkhe_core PRIVATE arkhe_cpp)

# Tests
enable_testing()
# Use the downloaded catch.hpp
target_include_directories(arkhe_cpp PRIVATE ${CMAKE_SOURCE_DIR}/cpp/tests)
add_executable(test_topology cpp/tests/test_topology.cpp)
target_include_directories(test_topology PRIVATE ${CMAKE_SOURCE_DIR}/cpp/tests)
target_link_libraries(test_topology arkhe_cpp)
add_test(NAME topology_test COMMAND test_topology)
