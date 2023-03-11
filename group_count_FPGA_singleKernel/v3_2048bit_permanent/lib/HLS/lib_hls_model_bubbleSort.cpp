//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
###############################
## Created: Intel Corporation 
##          Christian Faerber
##          PSG CE EMEA TS-FAE 
##          June 2022
###############################
*/

#include <CL/sycl.hpp>
#include <stdio.h>
#include <iostream>
#include <array>

#define N 16

SYCL_EXTERNAL extern "C" std::array<u_int32_t,N> bubble_sort(
        std::array<u_int32_t,N> x
){


        // unflatten array
        u_int32_t a[N];
        for (int i = 0; i < N; i++)
                a[i] = x[i];



        // bubble sort
        u_int32_t tmp = 0;

        for (int i = 0; i < N-1; i++)
                {

                for (int ii = 0; ii < N-1; ii++)
                        {

                                if(a[i] < a[i+1])
                                        {

                                                tmp = a[i];
                                                a[i] = a[i+1];
                                                a[i+1] = tmp;


                                        }

                        }

                }

        // copy array of int32 to array of ac_int for slice operation
        u_int32_t a0[N];
        for (int i = 0; i < N; i++)
                a0[i] = a[i];


        return a0;

}