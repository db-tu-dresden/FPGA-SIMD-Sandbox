#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

# g++ -mavx512f -O3 -fno-tree-vectorize -o test src/main/main.cpp \
#     src/operator/physical/gc/scalar_gc_soa.cpp \
#     src/operator/physical/gc/avx512_gc_soa_v1.cpp \
#     src/operator/physical/gc/avx512_gc_soa_v2.cpp \
#     src/operator/physical/gc/avx512_gc_soa_v3.cpp \
#     src/operator/physical/gc/avx512_gc_soaov_v1.cpp

g++ -mavx512f -mavx512cd -O3 -fno-tree-vectorize -o collision \
    src/main/main_collision.cpp \
    src/operator/physical/group_count/lp/*.cpp \
    src/operator/physical/group_count/lp_horizontal/*.cpp \
    src/operator/physical/group_count/lcp/*.cpp \
    src/operator/physical/group_count/lp_vertical/*.cpp \
    src/operator/physical/group_count/chained/*.cpp \
    src/main/benchmark/table.cpp


# g++ -mavx512f -mavx512cd -O3 -fno-tree-vectorize -o gc_test src/main/gc_testing_main.cpp \
#     src/operator/physical/gc/scalar_gc_soa.cpp \
#     src/operator/physical/gc/avx512_gc_soa_v1.cpp \
#     src/operator/physical/gc/avx512_gc_soa_v2.cpp \
#     src/operator/physical/gc/avx512_gc_soa_v3.cpp \
#     src/operator/physical/gc/avx512_gc_soaov_v1.cpp \
#     src/operator/physical/gc/avx512_gc_soa_collision_v1.cpp