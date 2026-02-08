# CMakeLists.txt - Avalon Neural Simulator [V2.1 INTERSTELLAR EDITION]
cmake_minimum_required(VERSION 3.20)
project(AvalonNeuralHPC VERSION 5040.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flags de Otimização
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -pthread")

# Target Principal
add_library(avalon_neural_core
    avalon_neural_core.cpp
    pulsar_sync.cpp
    collective_manifestation.cpp
)

# Executáveis
add_executable(avalon_test main.cpp)
target_link_libraries(avalon_test avalon_neural_core)

add_executable(avalon_benchmark benchmark.cpp)
target_link_libraries(avalon_benchmark avalon_neural_core)
