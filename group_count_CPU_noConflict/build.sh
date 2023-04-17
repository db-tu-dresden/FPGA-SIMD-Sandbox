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
       src_load512bit/kernel.cpp

