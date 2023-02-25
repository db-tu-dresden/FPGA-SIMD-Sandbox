cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)

project(GROUP-COUNT)
set(CMAKE_BINARY_DIR "bin")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_compile_options(-O3 -fno-tree-vectorize -march=native -mavx512f)

add_executable(main src/main.cpp)
target_sources(main PRIVATE src/helper.cpp src/LinearProbing_avx512.cpp src/LinearProbing_scalar.cpp)
