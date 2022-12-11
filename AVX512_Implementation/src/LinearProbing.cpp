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

/*
*   This is a proprietary AVX512 implementation of Bala Gurumurthy's LinearProbing approach. 
*   The logical approach refers to the procedure described in the paper.
*
*   Link to paper "SIMD Vectorized Hashing for Grouped Aggregation"
*   https://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/Gurumurthy:ADBIS18.pdf
*
*   Link to the Gitlab-repository of Bala Gurumurthy:
*   https://git.iti.cs.ovgu.de/bala/SIMD-Parallel-Hash-Based-Aggregation
*/

#define HSIZE 5000000


// simple multiplicative hashing function
unsigned int hash(int key) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

// print function for vector of type __m512i
void print512_num(__m512i var) {
    uint32_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Content of __m512i Array: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}

/* 
 * function to print an integer in bit-wise notation 
 * Assumes little endian
 * print-result:    p16, p15, p14, p13, p12, p11, p10, p09, p08, p07, p06, p05, p04, p03, p02, p01
 */
void printBits(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u ", byte);
        }
    }
    puts("");
}

/*
 * Main function of the LinearProbing algorithm regarding the logical program flow.
 * The algorithm uses the LinearProbing approach to perform a group-count aggregation.
 * @param arr[] the input data array
 * @param dataSize number of elements of the input data array 
 */
int vectorizedLinearProbing(unsigned int arr[], int dataSize) {
    printf("###########################\n");
    printf("Start of AVX512-Implemention of LinearProbing algorithm: \n");
    printf("Received value of dataSize: %i \n", dataSize); 
    printf("element [1] of vector: %i \n", arr[1]); 

    // start of algorithm logic
    __mmask16 oneMask = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512i zeroM512iArray = _mm512_setzero_epi32();

    /* 
     * create two vector to realize Hashmap
     * hasVec store value of k at position hash(k)
     * countVec store the count of occurence of k at position hash(k)
     * calloc : allocate memory and initialize all elements with '0'
     */
    unsigned int *hashVec = (unsigned int *)calloc((((unsigned long)((unsigned int)1300000077*dataSize)* HSIZE)>>32),sizeof(unsigned int));
    unsigned int *countVec = (unsigned int *)calloc((((unsigned long)((unsigned int)1300000077*dataSize)* HSIZE)>>32),sizeof(unsigned int));

    /* Insert the first element in the empty hashmap.
     * Since the data structures have only just been initialized,
     * the first element can be inserted directly
     */ 
    int i = 0;
    hashVec[hash(arr[i])] = arr[i];
    countVec[hash(arr[i])] = countVec[hash(arr[i])] + 1;

    // do the following commands for every element of arr[]
    for(i=1;i<dataSize; i+=1){
        // broadcast element i of arr[] to vector of type __m512i
        // broadcastCurrentValue contains sixteen times value of arr[i]
        __m512i broadcastCurrentValue = _mm512_set1_epi32(arr[i]);
print512_num(broadcastCurrentValue);

        /*
         * Load a vector of the type__m512i with the following addresses, 
         * starting from the start position hashVec[hash(arr[i])]
         * load the following elements of HashVec using the memory addresses from nextHashVecElements ()
         * realized here in one step, variable "nextHashVecElements" does not exist
         */
        __m512i nextHashVecElements = _mm512_mask_loadu_epi32(zeroM512iArray, oneMask, &hashVec[hash(arr[i])]);
print512_num(nextHashVecElements);

        // compare vector with broadcast value against vector with following elements for equality
        __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, nextHashVecElements);
printBits(sizeof(compareRes), &compareRes);

        /*
         * case distinction regarding the content of the mask "compareRes"
         *
         * case "a":
         * if compareRes != 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 --> identical value found (compared to broadcastCurrentValue).
         * --> Determine the position in hashVec[] (where the value is stored) and 
         * add +1 to the current count at this position in countVec 
         * (value has already been recorded, number must be increased)
         *
         * case "b":
         * if compareRes = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  --> value from "broadcastCurrentValue" not found in current set
         * case "b1": compare to zero: if result contains at least one "1" --> free space available
         *      @todo next step?
         * case "b2": no free space in curent set --> go +16 memory locations forward
        */ 
        int mask = (int)compareRes;
        // case "a"
        if (mask != 0) {
            printf("Result in mask: %i \n", mask); 
// @todo add_mask(...)

        } else {        
            // case "b1"
            __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(zeroM512iArray, nextHashVecElements);
            if((int)checkForFreeSpace == 0) {
                // no free space available in the current set
// @todo next step?

            } else {
                // free slots available in the current set
// @todo next step?

            }


        }




        // DELETE THIS BREAK - ONLY FOR TESTING!!
        break;
    }







    /* 
     * TODO :   implement error-handling: 
     *          return 1 if the program ends without errors 
     *          return 0 if an error has occurred       
     */
    return 1;
}