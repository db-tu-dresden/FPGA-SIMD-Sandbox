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

#include "helper.hpp"
#include "global_settings.hpp"

#define EMPTY_SPOT 0

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	OVERVIEW about functions in LinearProbing_avx512.cpp
//
//	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 - without aligned_start; version descbribed in paper
// 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
//	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
//	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
// 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 
// 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///// declare some (global) basic masks and arrays
__mmask16 oneMask = 0xFFFF;
__mmask16 zeroMask = 0x0000;
__mmask16 testMask = 0x0002;
__m512i zeroM512iArray = _mm512_setzero_epi32();
__m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 1 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512Variant1(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
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
        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);

        while (1) {
            // Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
            int32_t overflow = (hash_key + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);
            
            // Load 16 consecutive elements from hashVec, starting from position hash_key
            __m512i nextElements = _mm512_maskz_loadu_epi32(oneMask, &hashVec[hash_key]);
                       
            // compare vector with broadcast value against vector with following elements for equality
            __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
    
            /**
                * case distinction regarding the content of the mask "compareRes"
                * 
                * CASE (A):
                * inputValue does match one of the keys in nextElements (key match)
                * just increment the associated count entry in countVec
            **/ 
            if (compareRes != 0) {
                // load cout values from the corresponding location                
                __m512i nextCounts = _mm512_mask_loadu_epi32(zeroM512iArray, oneMask, &countVec[hash_key]);
                    
                // increment by one at the corresponding location
                nextCounts = _mm512_mask_add_epi32(nextCounts, compareRes, nextCounts, oneM512iArray);
                
                // selective store of changed value
                _mm512_mask_storeu_epi32(&countVec[hash_key],compareRes,nextCounts);
                p++;
                break;
            }   
            else {
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

                __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(), nextElements);
                uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                if(innerMask != 0) {            // CASE B1 
                    // __mmask16 mask1 = _mm512_knot(innerMask);    // not used anymore      
                    // compute position of the emtpy slot   
                    // uint32_t pos = (32-__builtin_clz(mask1))%16;
                    uint32_t pos = __builtin_ctz(checkForFreeSpace);

                    // use 
                    hashVec[hash_key+pos] = (uint32_t)inputValue;
                    countVec[hash_key+pos]++;
                    p++;
                    break;
                }   else    {                   // CASE B2   
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
 * Variant 2 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512Variant2(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    /**
     * iterate over input data
     * @param p current element of input data array
     **/ 
    int p = 0;
    while (p < dataSize) {

        // get input value
        uint32_t inputValue = input[p];

        // compute hash_key for the input value
        uint32_t hash_key = hashx(inputValue,HSIZE);

        // compute the aligned start position within the hashMap based the hash_key
        uint32_t aligned_start = (hash_key/16)*16;
        uint32_t remainder = hash_key - aligned_start; // should be equal to hash_key % 16

        /**
        * broadcast element p of input[] to vector of type __m512i
        * broadcastCurrentValue contains sixteen times value of input[i]
        **/
        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);
        while (1) {

            int32_t overflow = (aligned_start + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);

            int32_t cutlow = 16 - remainder; // should be in a range from 1-16
            uint32_t cutlow_mask_i = (1 << cutlow) -1;
            cutlow_mask_i <<= remainder;
            // __mmask16 cutlow_mask = _cvtu32_mask16(cutlow_mask_i); // unused

            uint32_t combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
            __mmask16 overflow_and_cutlow_mask = _cvtu32_mask16(combined_mask_i);

            // Load 16 consecutive elements from hashVec, starting from position hash_key
            __m512i nextElements = _mm512_load_epi32(&hashVec[aligned_start]);
            
            // compare vector with broadcast value against vector with following elements for equality
            __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
    
    
            /**
             * case distinction regarding the content of the mask "compareRes"
             * 
             * CASE (A):
             * inputValue does match one of the keys in nextElements (key match)
             * just increment the associated count entry in countVec
            **/
            if (compareRes != 0) {
                // compute the matching position indicated by a one within the compareRes mask
                // the position can be calculated two ways.
                // example: 00010000 is our matching mask
                // we could count the leading zeros and get the position like 7 - leadingzeros
                // we calculate the trailing zeros and get the position implicitly 
                uint32_t matchPos = __builtin_ctz(compareRes); 

                // WE COULD DO THIS LIKE VARIANT ONE.
                //  This would mean we wouldn't calculate the match pos since it is clear already.
                // increase the counter in countVec
                countVec[aligned_start+matchPos]++;
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

                // checkForFreeSpace. A free space is indicated by 1.
                __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, _mm512_setzero_epi32(),nextElements);
                uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                if(innerMask != 0) {                // CASE B1    

                    // __mmask16 mask1 = _mm512_knot(innerMask);
                    // uint32_t pos = ((64/sizeof(uint32_t))-__builtin_clz(mask1))%16;
                    
                    //this does not calculate the correct position. we should rather look at trailing zeros.
                    uint32_t pos = __builtin_ctz(checkForFreeSpace);
                    
                    hashVec[aligned_start+pos] = (uint32_t)inputValue;
                    countVec[aligned_start+pos]++;
                    p++;
                    break;
                }   else    {                   // CASE B2
                    //aligned_start = (aligned_start+16) % HSIZE;
                    // since we now use the overflow mask we can do this to change our position
                    // we ALSO need to set the remainder to 0.
                    remainder = 0;
                    aligned_start += 16;
                    if(aligned_start >= HSIZE){
                        aligned_start = 0;
                    }
                }
            }
        }
    }

}

/**
 * Variant 3 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512Variant3(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    /**
     * iterate over input data
     * @param p current element of input data array
     **/ 
    int p = 0;
    while (p < dataSize) {

        // load 16 input values
        __m512i iValues = _mm512_load_epi32(&input[p]);

        //iterate over the input values
        int i=0;
        while (i<16) {

            // broadcast single value from input at postion i into a new SIMD register
            __m512i broadcastCurrentValue = _mm512_permutexvar_epi32(_mm512_set1_epi32((uint32_t)i),iValues);

            uint32_t inputValue = (uint32_t)broadcastCurrentValue[0];
            uint32_t hash_key = hashx(inputValue,HSIZE);

            // compute the aligned start position within the hashMap based the hash_key
            uint32_t aligned_start = (hash_key/16)*16;
            uint32_t remainder = hash_key - aligned_start; // should be equal to hash_key % 16
         
            while (1) {
                int32_t overflow = (aligned_start + 16) - HSIZE;
                overflow = overflow < 0? 0: overflow;
                uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
                __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);

                int32_t cutlow = 16 - remainder; // should be in a range from 1-16
                uint32_t cutlow_mask_i = (1 << cutlow) -1;
                cutlow_mask_i <<= remainder;
                // __mmask16 cutlow_mask = _cvtu32_mask16(cutlow_mask_i); // unused

                uint32_t combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
                __mmask16 overflow_and_cutlow_mask = _cvtu32_mask16(combined_mask_i);

                // Load 16 consecutive elements from hashVec, starting from position hash_key
                __m512i nextElements = _mm512_load_epi32(&hashVec[aligned_start]);
            
                // compare vector with broadcast value against vector with following elements for equality
                __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
    
                /**
                * case distinction regarding the content of the mask "compareRes"
                * 
                * CASE (A):
                * inputValue does match one of the keys in nextElements (key match)
                * just increment the associated count entry in countVec
                **/ 
                if (compareRes != 0) {
                    // compute the matching position indicated by a one within the compareRes mask
                    // the position can be calculated two ways.
                    // example: 00010000 is our matching mask
                    // we could count the leading zeros and get the position like 7 - leadingzeros
                    // we calculate the trailing zeros and get the position implicitly 
                    uint32_t matchPos = __builtin_ctz(compareRes); 
                    
                    // WE COULD DO THIS LIKE VARIANT ONE.
                    // This would mean we wouldn't calculate the match pos since it is clear already.                
                    // increase the counter in countVec
                    countVec[aligned_start+matchPos]++;
                    i++;
                    break;
                }   
                else {
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

                    // checkForFreeSpace. A free space is indicated by 1.
                    __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, _mm512_setzero_epi32(),nextElements);
                    uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                    if(innerMask != 0) {                // CASE B1    
                        // __mmask16 mask1 = _mm512_knot(innerMask);   
                        // uint32_t pos = (32-__builtin_clz(mask1))%16;

                        //this does not calculate the correct position. we should rather look at trailing zeros.
                        uint32_t pos = __builtin_ctz(checkForFreeSpace);
                        
                        hashVec[aligned_start+pos] = (uint32_t)inputValue;
                        countVec[aligned_start+pos]++;
                        i++;
                        break;
                    }   
                    else    {                   // CASE B2                    
                        //aligned_start = (aligned_start+16) % HSIZE;
                        // since we now use the overflow mask we can do this to change our position
                        // we ALSO need to set the remainder to 0.
                        remainder = 0;
                        aligned_start += 16;
                        if(aligned_start >= HSIZE){
                            aligned_start = 0;
                        }
                    }
                }
            }
        }
        p+=16;
    }
}

/**
 * Variant 4 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512Variant4(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
  
// TODO create a stack allocated m512i array with implicit size. it should be small enough for the heap.
// note we need 2 of these. with this we can go around 

    //// declare the basic hash- and count-map structure for this approach and some function intern variables
    const size_t m_elements_per_vector = (512 / 8) / sizeof(uint32_t);
    const size_t m_HSIZE_v = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
    const size_t m_HSIZE = HSIZE;

    __m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512i hash_map[m_HSIZE_v];
    __m512i count_map[m_HSIZE_v];

    // loading data. On the first exec this should result in only 0 vals.
    for(size_t i = 0; i < m_HSIZE_v; i++){
        size_t h = i * m_elements_per_vector;
        hash_map[i] = _mm512_load_epi32(&hashVec[h]);
        count_map[i] = _mm512_load_epi32(&countVec[h]);
    }

    //creating writing masks
    __mmask16 masks[17];
    masks[0] = _cvtu32_mask16(0);
    for(size_t i = 1; i <= 16; i++){
        masks[i] = _cvtu32_mask16(1 << (i-1));
    }

    int p = 0;
    while (p < dataSize) {

        uint32_t inputValue = input[p];
        uint32_t hash_key = hashx(inputValue, m_HSIZE_v);

        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);

        while(1) {
            // compare vector with broadcast value against vector with following elements for equality
            __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, hash_map[hash_key]);

            // found match
            if (compareRes > 0) {
                count_map[hash_key] = _mm512_mask_add_epi32(count_map[hash_key], compareRes, count_map[hash_key], oneM512iArray);

                p++;
                break;
            } else { // no match found
                // deterime free position within register
                __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), hash_map[hash_key]);
                if(checkForFreeSpace > 0) {                // CASE B1    
                    uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
                    
                    //store key
                    hash_map[hash_key] = _mm512_mask_set1_epi32(hash_map[hash_key], masks[pos], inputValue);
                    //set count to one
                    count_map[hash_key] = _mm512_mask_set1_epi32(count_map[hash_key], masks[pos], 1);
                    p++;
                    break;
                }   else    { // CASE B2
                    hash_key = (hash_key + 1) % m_HSIZE_v;
                }
            }
        }
    }

    //store data
    for(size_t i = 0; i < m_HSIZE_v; i++){
        size_t h = i * m_elements_per_vector;
        _mm512_store_epi32(&hashVec[h], hash_map[i]);
        _mm512_store_epi32(&countVec[h], count_map[i]);
    }
}

/**
 * Variant 5 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingAVX512Variant5(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
     __mmask16 all_mask = 0xFFFF;

    const __m512i one = _mm512_set1_epi32 (1);
    const __m512i zero = _mm512_setzero_epi32();

    uint32_t *buffer = reinterpret_cast< uint32_t* >( _mm_malloc( 16 * sizeof( uint32_t ), 64 ) );
    size_t p = 0;
    while(p + 16 <= dataSize){

        // load the to aggregate data
        __m512i input_value = _mm512_load_epi32(&input[p]);
        // how much the given count should be increased for the given input.
        __m512i input_add = _mm512_set1_epi32(1);

        // search for conflicts
        __m512i conflicts = _mm512_conflict_epi32(input_value);
        // masked to indicate were there is a conflict in the input_values and were not.
        __mmask16 no_conflicts_mask = _mm512_cmpeq_epi32_mask(zero, conflicts);
        __mmask16 negativ_no_conflicts_mask = _mm512_knot(no_conflicts_mask);

        // we need to store the conflicts so we can interprete them as masks. and access them.
        // we are only interested in the enties that are not zero. That means the conflict cases.
        _mm512_mask_compressstoreu_epi32(buffer, negativ_no_conflicts_mask, conflicts);
        size_t conflict_count = __builtin_popcount((uint32_t)(negativ_no_conflicts_mask));
        // add at all the places where the conflict masks indicates that there is an overlap
        for(size_t i = 0; i < conflict_count; i++){
            input_add = _mm512_mask_add_epi32(input_add, (__mmask16)(buffer[i]), input_add, one);
        }
        // we override the value and what to add with zero in the positions where we have a conflict.
        // NOTE: This steps might not be necessary.
        input_value = _mm512_mask_set1_epi32(input_value, negativ_no_conflicts_mask, 0);
        input_add = _mm512_mask_set1_epi32(input_add, negativ_no_conflicts_mask, 0);

        // now we can calculate the hashes.
        // for this we can store the input_value hash it and load it
        // OR we use the input and hash it save it in to buffer and than make a maskz load for the hashed data
        // OR we have a simdifyed Hash Algorithm! For the most cases we would need an avx... mod. 
        // _mm512_store_epi32(buffer, input_value);
        for(size_t i = 0; i < 16; i++){
            buffer[i] = hashx(input[p + i], HSIZE);
        }
        __m512i hash_map_position = _mm512_maskz_load_epi32(no_conflicts_mask, buffer); // these are the hash values

        do{
            // now we can gather the data from the different positions where we have no conflicts.
            __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero, no_conflicts_mask, hash_map_position, hashVec, 4);
            // with these we can calculate the different possible hits. Real hits and empty positions.
            __mmask16 foundPos = _mm512_mask_cmpeq_epi32_mask(no_conflicts_mask, input_value, hash_map_value);
            __mmask16 foundEmpty = _mm512_mask_cmpeq_epi32_mask(no_conflicts_mask, zero, hash_map_value);

            if(foundPos != 0){//A
                // Now we have to gather the count. IMPORTANT! the count is a 32bit integer. 
                    // FOR NOW THIS IS CORRECT BUT MIGHT CHANGE LATER!
                // For 64bit integers we would need to find a different solution!
                __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero, foundPos, hash_map_position, countVec, 4);
                // on this count we can know add the pre calculated values. and scatter it back to their positions
                hash_map_value = _mm512_maskz_add_epi32(foundPos, hash_map_value, input_add);
                _mm512_mask_i32scatter_epi32(countVec, foundPos, hash_map_position, hash_map_value, 4);
                
                // finaly we remove the entries we just saved from the no_conflicts_mask such that the work to be done shrinkes.
                no_conflicts_mask = _mm512_kandn(foundPos, no_conflicts_mask);
            }
            if(foundEmpty != 0){//B1
                // now we have to check for conflicts to prevent two different entries to write to the same position.
                __m512i saveConflicts = _mm512_maskz_conflict_epi32(foundEmpty, hash_map_position);
                __m512i empty = _mm512_set1_epi32(foundEmpty);
                saveConflicts = _mm512_and_epi32(saveConflicts, empty);

                __mmask16 to_save_data = _mm512_cmpeq_epi32_mask(zero, saveConflicts);
                to_save_data = _mm512_kand(to_save_data, foundEmpty);

                // with the cleaned mask we can now save the data.
                _mm512_mask_i32scatter_epi32(hashVec, to_save_data, hash_map_position, input_value, 4);
                _mm512_mask_i32scatter_epi32(countVec, to_save_data, hash_map_position, input_add, 4);
                
                //and again we need to remove the data from the todo list
                no_conflicts_mask = _mm512_kandn(to_save_data, no_conflicts_mask);
            }
            
            // afterwards we add one on the current positions of the still to be handled values.
            hash_map_position = _mm512_maskz_add_epi32(no_conflicts_mask, hash_map_position, one);
            // Since there isn't a modulo operation we have to check if the values are bigger or equal the HSIZE AND IF we have to set them to zero
            __mmask16 tobig = _mm512_mask_cmp_epi32_mask(no_conflicts_mask, hash_map_position, _mm512_set1_epi32(HSIZE), _MM_CMPINT_NLT);
            hash_map_position = _mm512_mask_set1_epi32(hash_map_position, tobig, 0);

            // we repeat this for one vector as long as their is still a value to be saved.
        }while(no_conflicts_mask != 0);
        p += 16;
    }

    //scalar remainder
    while(p < dataSize){
        int error = 0;
        // get the possible possition of the element.
        uint32_t hash_key = hashx(input[p], HSIZE);
        
        while(1){
            // get the value of this position
            uint32_t value = hashVec[hash_key];
            
            // Check if it is the correct spot
            if(value == input[p]){
                countVec[hash_key]++;
                break;
            
            // Check if the spot is empty
            }else if(value == EMPTY_SPOT){
                hashVec[hash_key] = input[p];
                countVec[hash_key] = 1;
                break;
            
            }
            else{
                //go to the next spot
                hash_key = (hash_key + 1) % HSIZE;
                //we assume that the hash_table is big enough
            }
        }
        p++;
    }
}


/*
std::cout<<"input_value:"<<std::endl;
print512_num(input_value);
std::cout<<" "<<std::endl;

std::cout<<"no_conflicts_mask: "<<std::endl;
printBits(sizeof(no_conflicts_mask), &no_conflicts_mask);
std::cout<<" "<<std::endl;
*/