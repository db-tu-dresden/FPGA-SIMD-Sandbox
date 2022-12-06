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
    printf("Numerical: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
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
    
    /* 
     * create two vector to realize Hashmap
     * hasVect store value of k at position hash(k)
     * countVect store the count of occurence of k at position hash(k)
     * calloc : allocate memory and initialize all elements with '0'
     */
    unsigned int *hashVect = (unsigned int *)calloc((((unsigned long)((unsigned int)1300000077*dataSize)* HSIZE)>>32),sizeof(unsigned int));
    unsigned int *countVect = (unsigned int *)calloc((((unsigned long)((unsigned int)1300000077*dataSize)* HSIZE)>>32),sizeof(unsigned int));

    /* Insert the first element in the empty hashmap.
     * Since the data structures have only just been initialized,
     * the first element can be inserted directly
     */ 
    int i = 0;
    hashVect[hash(arr[i])] = arr[i];
    countVect[hash(arr[i])] = countVect[hash(arr[i])] + 1;

    // do the following commands for every element of arr[]
    for(i=1;i<dataSize; i+=1){
        __m512i broadcastCurrentValue = _mm512_set1_epi32(arr[i]);
        print512_num(broadcastCurrentValue);


        // DELETE THIS BREAK - ONLY FOR TESTING!!
        break;
    }


/*
    printf("element 1: %i \n", arr[1]); 
    printf("hash of element 1: %i \n", hash(arr[1])); 
    printf("hash of element 1: %i \n", 1300000077*arr[1]* HSIZE); 
    printf("element 2: %i \n", arr[2]); 
    printf("hash of element 2: %i \n", hash(arr[2])); 
    printf("hash of element 2: %i \n", 1300000077*arr[2]* HSIZE); 
    printf("element 3: %i \n", arr[3]); 
    printf("hash of element 3: %i \n", hash(arr[3])); 
    printf("hash of element 3: %i \n", 1300000077*arr[3]* HSIZE); 
*/
    // __m512i zero = _mm512_setzero_epi32();
    // print512_num(zero);
    return 1;
}