#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "helper.cpp"


/**
 * declare some (global) basic masks and arrays
 */ 
__mmask16 oneMask = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
__mmask16 zeroMask = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
__m512i zeroM512iArray = _mm512_setzero_epi32();

/**
 * Main function of the AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param arr the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512(uint32_t* arr, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    printf("###########################\n");
    printf("BEGIN of LinearProbingAVX512() function - AVX512 section: \n");

    //initalize hash array with zeros
    for (int i=0; i<HSIZE;i++) {
        hashVec[i]=0;
        countVec[i]=0;
    }

    // iterate over input data
    int p = 0;
    while (p < dataSize) {

        // @todo logic of linear probing algorithm




        p++;
    }
    printf("END of LinearProbingAVX512() function - AVX512 section: \n");
    printf("###########################\n");
}