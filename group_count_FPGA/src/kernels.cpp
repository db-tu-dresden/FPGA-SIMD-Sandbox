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

#include "primitives.hpp"
#include "kernels.hpp"
#include "helper_kernel.cpp"

class kernels;

/**
 * declare some (global) basic masks and arrays
 */ 
uint32_t one = 1;
uint32_t zero = 0;
fpvec<uint32_t> oneMask = set1(one);
fpvec<uint32_t> zeroMask = set1(zero);
fpvec<uint32_t> zeroM512iArray = set1(zero);
fpvec<uint32_t> oneM512iArray = set1(one);

/**
 * Variant 1 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant1(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE) {
    /** 
     * define example register
     * fpvec<uint32_t> testReg;
     * 
     * example function call from primitives.hpp
     * testReg = cvtu32_mask16(n);
     */

    /**
     * iterate over input data
     * @param p current element of input data array
     **/ 
    int p = 0;
    while (p < dataSize) {
      // get single value from input at position p
      uint32_t inputValue = input[p];

      // compute hash_key of the input value
      uint32_t hash_key = hashx(inputValue,HSIZE);

      // broadcast inputValue into a SIMD register
      fpvec<uint32_t> broadcastCurrentValue = set1(inputValue);

      while (1) {
        // Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
        int32_t overflow = (hash_key + 16) - HSIZE;
        overflow = overflow < 0? 0: overflow;
        uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
        fpvec<uint32_t> overflow_correction_mask = cvtu32_mask16(overflow_correction_mask_i);
      
        // Load 16 consecutive elements from hashVec, starting from position hash_key
        fpvec<uint32_t> nextElements = mask_loadu(oneMask, hashVec, hash_key, HSIZE);
                       
        // compare vector with broadcast value against vector with following elements for equality
        fpvec<uint32_t> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
    
        /**
        * case distinction regarding the content of the mask "compareRes"
        * 
        * CASE (A):
        * inputValue does match one of the keys in nextElements (key match)
        * just increment the associated count entry in countVec
        **/ 
        if (mask2int(compareRes) == 1) {
          // load cout values from the corresponding location                
          fpvec<uint32_t> nextCounts = mask_loadu(oneMask, countVec, hash_key, HSIZE);
                    
          // increment by one at the corresponding location
          nextCounts = mask_add_epi32(nextCounts, compareRes, nextCounts, oneM512iArray);
                
          // selective store of changed value
          mask_storeu_epi32(countVec, hash_key, HSIZE, compareRes,nextCounts);
          p++;
          break;
          }   else {
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
            fpvec<uint32_t> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_correction_mask, zeroMask, nextElements);
            uint32_t innerMask = mask2int(checkForFreeSpace);
            if(innerMask != 0) {                // CASE B1    
              fpvec<uint32_t> mask1 = knot(checkForFreeSpace);

              // compute position of the emtpy slot   
              uint32_t pos = (32-clz_onceBultin(mask1))%16;

              // use 
              hashVec[hash_key+pos] = (uint32_t)inputValue;
              countVec[hash_key+pos]++;
              p++;
              break;
            } else    {                   // CASE B2   
              hash_key += 16;
              if(hash_key >= HSIZE){
                hash_key = 0;
              }
            }
          }
      } 
    }
}   

/**
 * Variant 2 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant2(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE) {

}  

/**
 * Variant 3 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant3(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE) {

}  