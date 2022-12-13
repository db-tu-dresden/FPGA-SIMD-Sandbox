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
__m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

/**
 * Main function of the AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param arr the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    printf("###########################\n");
    printf("BEGIN of LinearProbingAVX512() function - AVX512 section: \n");

    //initalize hash array with zeros
    for (int i=0; i<HSIZE;i++) {
        hashVec[i]=0;
        countVec[i]=0;
    }

    /**
     * iterate over input data
     * @param p current element of input data array
     **/ 
    int p = 0;
    while (p < dataSize) {
        uint32_t inputValue = input[p];
        uint32_t hash_key = hashx(input[p],HSIZE);
        // cout << "current pair input/hash: " << input[p]<<" "<<hash_key<<endl;

        /**
         * broadcast element p of input[] to vector of type __m512i
         * broadcastCurrentValue contains sixteen times value of input[i]
         **/
        __m512i broadcastCurrentValue = _mm512_set1_epi32(input[p]);

        // Load 16 consecutive elements from hashVec, starting from position hash_key
        __m512i nextElements = _mm512_mask_loadu_epi32(zeroM512iArray, oneMask, &hashVec[hash_key%HSIZE]);

        // compare vector with broadcast value against vector with following elements for equality
        __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, nextElements);
       
        /**
         * case distinction regarding the content of the mask "compareRes"
         * 
         * CASE (A):
         * inputValue does match one of the keys in nextElements (key match)
         * just increment the associated count entry in countVec
         **/ 
        int mask = _mm512_mask2int(compareRes);
        if (mask == 1) {
            // cout << "CASE A:" <<endl;

            __m512i nextCounts = _mm512_mask_loadu_epi32(zeroM512iArray, oneMask, &countVec[hash_key%HSIZE]);
            nextCounts = _mm512_mask_add_epi32(nextCounts, compareRes , nextCounts, oneM512iArray);
            // print512_num(nextCounts);

            uint32_t val[16];
            memcpy(val, &nextCounts, sizeof(val));
            for(int i=0; i<16; i++) {
                if(val[i] != 0) {
                    // cout << val[i] <<endl;
                    countVec[(hash_key+i)%HSIZE] = val[i];
                    // alternativ addition:
                    // countVec[(hash_key+i)%HSIZE] = countVec[(hash_key+i)%HSIZE] + 1;
                    break;
                };
            }
            p++;
        }   else {
            // cout << "CASE B: " <<endl;
             /**
              * CASE (B): 
              * --> inputValue does NOT match any of the keys in nextElements (no key match)
              * --> compare "nextElements" with zero
              * CASE (B1):   resulting mask of this comparison is not 0
              *             --> insert inputValue into next possible slot       
              *             
              * CASE (B2):  resulting mask of this comparison is 0
              *             --> no free slot in current 16-slot array
              *             --> load next +16 elements (add +16 to hash_key and re-iterate through while-loop without incrementing p)
              *             --> attention for the overflow of hashVec & countVec ! (% HSIZE, continuation at position 0)
              **/ 

            // __m512i freeSlots is used as a helper; contains the informations of __mmask16 checkForFreeSpace
            // @todo    find a method to be able to access the individual bits of the mask "checkForFreeSpace" directly
            //          and thus to process their information directly
            __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(zeroM512iArray, nextElements);
            int innerMask = _mm512_mask2int(checkForFreeSpace);
            if(innerMask != 0) {                // CASE B1          
                __m512i freeSlots = _mm512_setzero_epi32();
                freeSlots = _mm512_mask_add_epi32(freeSlots, checkForFreeSpace , freeSlots, oneM512iArray);
                print512_num(freeSlots);


                uint32_t val[16];
                memcpy(val, &freeSlots, sizeof(val));
                for(int i=0; i<16; i++) {              
                    if(val[i] == 1) {
                        cout << "Free slot found! Insert inputValue: " << inputValue << " at Position hashVec[(hash_key+i)%HSIZE]:" << ((hash_key+i)%HSIZE) << endl;

                        // insert key at new position and increment corresponding count
                        hashVec[(hash_key+i)%HSIZE] = inputValue;
                        countVec[(hash_key+i)%HSIZE] = countVec[(hash_key+i)%HSIZE] + 1;
                        break;
                    };
                }
            p++;
            }   else    {                   // CASE B2   
                /**
                 * @todo : error-handling: what, if there is no free slot (bad global settings!)
                 * @todo : avoid infinite loop!
                */
                hash_key = (hash_key+16) % HSIZE;
            }
        }
        
        /*
        if(p==30) {
            break;
        };     */
    }
    for(int i=0; i<16; i++) {
        cout << "Endresult value / count: " << hashVec[i] << "  " << countVec[i] <<endl;
    };
    printf("END of LinearProbingAVX512() function - AVX512 section: \n");
    printf("###########################\n");
}