#include <HLS/hls.h>
#include <HLS/ac_int.h>
#include <stdio.h>

#define N 16

component hls_always_run_component ac_int<32*N, false> bubble_sort(
        ac_int<32*N, false> x
){


        // unflatten array
        u_int32_t a[N];
        for (int i = 0; i < N; i++)
                a[i] = x.slc<32>(i*32);



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
        ac_int<32, false> a0[N];
        for (int i = 0; i < N; i++)
                a0[i] = a[i];


        // flatten
        ac_int<32*N, false> y;
        for (int i = 0; i < N; i++)
                y.set_slc(i*32, a0[i]);

        return y;

}
