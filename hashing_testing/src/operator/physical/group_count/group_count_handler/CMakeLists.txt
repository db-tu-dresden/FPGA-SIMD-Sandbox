

# if(USE_AVX512)
#     target_compile_definitions(GCH -DUSE_AVX512)
# endif()

add_library(GCH group_count_algorithms.cpp)

