//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>


SYCL_EXTERNAL float SyclSquare(float x) { return x * x; }
SYCL_EXTERNAL float SyclSqrt(float x) { return sqrt(x); }
SYCL_EXTERNAL int SyclMult(int x, int y) { return x * y; }

#define N 16

SYCL_EXTERNAL std::array<int, 16> SyclBubbleSort(
        std::array<int, 16> x
){

        // Input registers
        int a[N];
        
        #pragma unroll
        for (int i = 0; i < N; i++)
                a[i] = x[i];

        // bubble sort
        int tmp = 0;
        
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            #pragma unroll
            for (int j = 0; j < N - i - 1; ++j) {
                if (a[j] > a[j + 1]) {
                    tmp = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = tmp;
                }
            }
        }

        // Output registers
        std::array<int, 16> a0;
        
        #pragma unroll
        for (int i = 0; i < N; i++)
                a0[i] = a[i];
    
        return a0;
}

