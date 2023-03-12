#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

# old  g++ -std=c++14 -O3 -o main src/main.cpp \ ...

# new intel compiler with debug flags for gdb-oneapi debugger: icpx -fsycl -O3 -g -o main src/main.cpp \

# new intel compiler without debug flags:  icpx -fsycl -O3 -o main src/main.cpp \

#icpx -fsycl -O3 -g -o main src_inner512bits/main.cpp \


# build 2048bit_permanent project (scalar + LinarProbing v1, v2, v3, v4)
#   icpx -fsycl -O3 -o main src_2048bit_permanent/main.cpp \
#       helper/helper_kernel.cpp \
#       helper/helper_main.cpp \
#       src_2048bit_permanent/kernel.cpp \
#       src_2048bit_permanent/LinearProbing_scalar.cpp \

# build inner_512bit project (scalar + LinarProbing v1, v2, v3, v4, v5)
#   icpx -fsycl -O3 -o main src_inner512bits/main.cpp \
#       helper/helper_kernel.cpp \
#       helper/helper_main.cpp \
#       src_inner512bits/kernel.cpp \
#       src_inner512bits/LinearProbing_scalar.cpp \



icpx -fsycl -O3 -o main src_inner512bits/main.cpp \
    helper/helper_kernel.cpp \
    helper/helper_main.cpp \
    src_inner512bits/kernel.cpp \
    src_inner512bits/LinearProbing_scalar.cpp \