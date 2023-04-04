#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

# old  g++ -std=c++14 -O3 -o main src_load512bit/main.cpp \ ...

# new intel compiler with debug flags for gdb-oneapi debugger: icpx -fsycl -O3 -g -o main src_load512bit/main.cpp \

# new intel compiler without debug flags:  icpx -fsycl -O3 -o main src_load512bit/main.cpp \

#icpx -fsycl -O3 -g -o main src_load512bit/main.cpp \





icpx -fsycl -O3 -o main src_load512bit/main.cpp \
       helper/helper_kernel.cpp \
       helper/helper_main.cpp \
       src_load512bit/kernel.cpp \
       src_load512bit/LinearProbing_scalar.cpp \















## old project - not meassured on CPU for evaluation
# build 2048bit_permanent project (scalar + LinarProbing v1, v2, v3, v4)
#   icpx -fsycl -O3 -o main z_old_projects/src_2048bit_permanent/main.cpp \
#       helper/helper_kernel.cpp \
#       helper/helper_main.cpp \
#       z_old_projects/src_2048bit_permanent/kernel.cpp \
#       z_old_projects/src_2048bit_permanent/LinearProbing_scalar.cpp \

## old project - not meassured on CPU for evaluation
# build inner_512bit project (scalar + LinarProbing v1, v2, v3, v4, v5)
#   icpx -fsycl -O3 -o main z_old_projects/src_inner512bits/main.cpp \
#       helper/helper_kernel.cpp \
#       helper/helper_main.cpp \
#       z_old_projects/src_inner512bits/kernel.cpp \
#       z_old_projects/src_inner512bits/LinearProbing_scalar.cpp \