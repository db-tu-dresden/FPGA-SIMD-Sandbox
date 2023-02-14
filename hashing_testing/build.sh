#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

# g++ -mavx512f -O3 -fno-tree-vectorize -o test src/main/main.cpp \
#     src/operator/physical/group_count/scalar_group_count.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_v1.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_v2.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_v3.cpp \
#     src/operator/physical/group_count/avx512_group_count_soaov_v1.cpp

g++ -mavx512f -mavx512cd -O3 -fno-tree-vectorize -o collision src/main/main_collision.cpp \
    src/operator/physical/group_count/scalar_group_count.cpp \
    src/operator/physical/group_count/avx512_group_count_soa_v1.cpp \
    src/operator/physical/group_count/avx512_group_count_soa_v2.cpp \
    src/operator/physical/group_count/avx512_group_count_soa_v3.cpp \
    src/operator/physical/group_count/avx512_group_count_soaov_v1.cpp \
    src/operator/physical/group_count/avx512_group_count_soa_collision_v1.cpp


# g++ -mavx512f -mavx512cd -O3 -fno-tree-vectorize -o group_count_test src/main/group_count_testing_main.cpp \
#     src/operator/physical/group_count/scalar_group_count.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_v1.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_v2.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_v3.cpp \
#     src/operator/physical/group_count/avx512_group_count_soaov_v1.cpp \
#     src/operator/physical/group_count/avx512_group_count_soa_collision_v1.cpp