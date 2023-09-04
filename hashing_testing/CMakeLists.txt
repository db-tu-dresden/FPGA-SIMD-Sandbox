cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(hashing)

option(USE_AVX512 "use horizontal printing provided by terminal.cpp" OFF)

if(USE_AVX512)
    add_definitions(-DUSE_AVX512)
    add_compile_options(-mavx512cd)
endif()

add_compile_options(-O3 -fno-tree-vectorize)


include_directories(hashing PUBLIC 
    "${PROJECT_SOURCE_DIR}/include/"
)

add_subdirectory("src/")

add_executable(hashing src/main/main_benchmark.cpp)
add_executable(datagen src/main/datagen_test.cpp)
# add_executable(datagen_test src/main/main_datagen_test.cpp)

target_link_libraries(hashing PUBLIC TABLE FILE GCH LP HASH_FUNCTIONS DATAGEN)
target_link_libraries(datagen PUBLIC HASH_FUNCTIONS DATAGEN)
# target_link_libraries(CHAINED)
if(USE_AVX512)
    target_link_libraries(hashing PUBLIC LCP LP_HORIZONTAL LP_VERTICAL)
endif()