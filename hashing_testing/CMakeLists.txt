cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(hashing_tsl)

include(tools/tsl/tsl.cmake)

create_tsl(
  TSLGENERATOR_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools/tsl"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tools/tsl"
  USE_CONCEPTS
)



option(USE_AVX512 "use horizontal printing provided by terminal.cpp" OFF)

if(USE_AVX512)
add_definitions(-DUSE_AVX512)
add_compile_options(-mavx512cd)
endif()

add_compile_options(-O3 -fno-tree-vectorize)


include_directories(hashing_tsl PUBLIC 
"${CMAKE_CURRENT_SOURCE_DIR}/include/"
)

# include_directories(hashing PUBLIC ${TSL_INCLUDE_DIRECTORY})
include_directories(hashing_tsl PUBLIC ${TSL_INCLUDE_DIRECTORY})

add_subdirectory("src/")

# add_executable(hashing src/main/main_benchmark.cpp)
add_executable(hashing_tsl src/main/tsl_benchmark.cpp)
add_executable(datagen src/main/datagen_test.cpp)
add_executable(tsl_testing src/main/main_tsl_test.cpp)

# add_executable(datagen_test src/main/main_datagen_test.cpp)
target_compile_options(hashing_tsl PUBLIC -lnuma)
# target_link_libraries(hashing PUBLIC TABLE FILE GCH LP HASH_FUNCTIONS DATAGEN tsl)
target_link_libraries(hashing_tsl PUBLIC TABLE FILE GCH LP LCP LP_HORIZONTAL LP_VERTICAL HASH_FUNCTIONS DATAGEN tsl "numa")
target_link_libraries(tsl_testing tsl)
# target_include_directories(hashing PUBLIC ${TSL_INCLUDE_DIRECTORY} "${TSL_INCLUDE_DIRECTORY}/../supplementary/runtime/cpu/include/")
# target_link_libraries(hashing tsl)

target_link_libraries(datagen PUBLIC HASH_FUNCTIONS DATAGEN)
# target_link_libraries(CHAINED)
if(USE_AVX512)
    # target_link_libraries(hashing PUBLIC LCP LP_HORIZONTAL LP_VERTICAL)
endif()