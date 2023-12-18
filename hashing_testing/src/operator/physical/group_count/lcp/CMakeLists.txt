add_library(LCP avx512_gc_aosov_v1.cpp avx512_gc_aosov_v2.cpp avx512_gc_aosov_v3.cpp avx512_gc_aosov_v4.cpp avx512_gc_soaov_v1.cpp avx512_gc_soaov_v2.cpp tsl_gc_lcp_soa.cpp)
# include_directories(LCP PUBLIC ${TSL_INCLUDE_DIRECTORY})
target_compile_options(LCP PUBLIC -Wno-undef)
target_link_libraries(LCP LP tsl)