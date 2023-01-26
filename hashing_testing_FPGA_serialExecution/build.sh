#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

g++ -std=c++14 -O3 src/main/main.cpp -o test \
    src/operator/physical/group_count/scalar_group_count.cpp \
    src/operator/physical/group_count/fpga_group_count_soa_v1.cpp \
    src/operator/physical/group_count/fpga_group_count_soa_v2.cpp \
    src/operator/physical/group_count/fpga_group_count_soa_v3.cpp \
    src/operator/physical/group_count/fpga_group_count_soaov_v1.cpp