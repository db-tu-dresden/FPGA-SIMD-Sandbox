#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

g++ src/main/main.cpp src/operator/physical/group_count/scalar_group_count.cpp src/operator/physical/group_count/avx512_group_count_soa_v1.cpp src/operator/physical/group_count/avx512_group_count_soa_v2.cpp src/operator/physical/group_count/avx512_group_count_soa_v3.cpp -mavx512f