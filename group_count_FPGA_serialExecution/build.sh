#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

# old  g++ -std=c++14 -O3 -o main src/main.cpp \ ...

# new intel compiler with debug flags for gdb-oneapi debugger: icpx -fsycl -O3 -g -o main src/main.cpp \

# new intel compiler without debug flags:  icpx -fsycl -O3 -o main src/main.cpp \


#icpx -fsycl -O3 -o main src/main.cpp \
icpx -fsycl -O3 -g -o main src/main.cpp \
    src/helper_kernel.cpp \
    src/helper_main.cpp \
    src/kernel.cpp \
    src/LinearProbing_scalar.cpp \
