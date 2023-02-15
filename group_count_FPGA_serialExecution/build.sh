#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

 g++ -std=c++14 -O3 -o main src/main.cpp \
    src/helper_kernel.cpp \
    src/helper_main.cpp \
    src/kernel.cpp \
    src/LinearProbing_scalar.cpp \
