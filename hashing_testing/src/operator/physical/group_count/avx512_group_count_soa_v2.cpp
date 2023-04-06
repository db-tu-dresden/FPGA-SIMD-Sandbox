#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soa_v2.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoA_v2<T>::AVX512_group_count_SoA_v2(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_group_count_SoA_v2<T>::~AVX512_group_count_SoA_v2(){
    // free(this->m_hash_vec);
    // free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoA_v2<T>::identify(){
    return "AVX512 Group Count SoA Version 2";
}


template <typename T>
void AVX512_group_count_SoA_v2<T>::create_hash_table(T* input, size_t data_size){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    while(p < data_size){
        int error = 0;
        // get the possible possition of the element.
        T hash_key = this->m_hash_function(input[p], HSIZE);
        while(1){
            
            // get the value of this position
            T value = this->m_hash_vec[hash_key];
            
            // Check if it is the correct spot
            if(value == input[p]){
                this->m_count_vec[hash_key]++;
                break;
            // Check if the spot is empty
            }else if(value == EMPTY_SPOT){
                this->m_hash_vec[hash_key] = input[p];
                this->m_count_vec[hash_key] = 1;
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


template <>
void AVX512_group_count_SoA_v2<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;

    __mmask16 oneMask = 0xFFFF;
    __m512i zeroM512iArray = _mm512_setzero_epi32();
    __m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

    int p = 0;
    while (p < data_size) {

        // get input value
        uint32_t inputValue = input[p];

        // compute hash_key for the input value
        uint32_t hash_key = this->m_hash_function(inputValue,HSIZE);

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

//WE COULD DO THIS LIKE VARIANT ONE.
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
                    /**
                    * @todo : error-handling: what, if there is no free slot (bad global settings!)
                    *       : this case is not implemented yet 
                    * @todo : avoid infinite loop!
                    */
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

template class AVX512_group_count_SoA_v2<uint32_t>;
template class AVX512_group_count_SoA_v2<uint64_t>;