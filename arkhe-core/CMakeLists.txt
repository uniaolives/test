cmake_minimum_required(VERSION 3.16)
project(arkhen-core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(pybind11 QUIET)

# Libraries
add_library(arkhe_cpp SHARED
    cpp/src/kernel.cpp
    cpp/src/nexus_node.cpp
    cpp/src/arkhe_vessel.cpp
)

target_include_directories(arkhe_cpp PUBLIC
    ${PROJECT_SOURCE_DIR}/cpp/include
)

# Python Module
if(pybind11_FOUND)
    pybind11_add_module(arkhe_core
        cpp/src/python_bindings.cpp
    )
    target_link_libraries(arkhe_core PRIVATE arkhe_cpp)
endif()

# Main executable
add_executable(arkhen_node cpp/src/main.cpp)
target_link_libraries(arkhen_node arkhe_cpp)

# Ensure data directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

# Tests
enable_testing()

include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.9
)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2
)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.11.2
)
FetchContent_MakeAvailable(catch2 json httplib)

# Genesis Node
add_executable(arkhe_genesis cpp/src/arkhe_genesis.cpp)
find_package(SQLite3 REQUIRED)
target_link_libraries(arkhe_genesis PRIVATE nlohmann_json::nlohmann_json httplib::httplib SQLite::SQLite3)

# Tests
add_executable(test_topology cpp/tests/test_topology.cpp)
target_link_libraries(test_topology arkhe_cpp Catch2::Catch2)
add_test(NAME topology_test COMMAND test_topology)
