#!/bin/bash

#projectRoot="$(pwd)"
#buildDir="${projectRoot}/build"

#cmake -S "$projectRoot" -B "$buildDir" 

icpx -fsycl -O3 -o main main.cpp \
    ../../helper/helper_kernel.cpp \
     ../../helper/helper_main.cpp \
    kernel.cpp    