#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "operator/physical/group_count/lp_horizontal/avx512_gc_soa_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_SoA_v1<T>::AVX512_gc_SoA_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_gc_SoA<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_gc_SoA_v1<T>::~AVX512_gc_SoA_v1(){
    // free(this->m_hash_vec);
    // free(this->m_count_vec);
}

template <typename T>
void AVX512_gc_SoA_v1<T>::create_hash_table(T* input, size_t data_size){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    while(p < data_size){
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


// only an 32 bit uint implementation rn, because we don't use the tvl rn. As soon as we use the TVL we should reform this code to support it.
template <>
void AVX512_gc_SoA_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;

    __mmask16 oneMask = 0xFFFF;
    __m512i zero_512i = _mm512_setzero_epi32();
    __m512i one_512i = _mm512_set1_epi32(1);
    __m512i two_512i = _mm512_set1_epi32(2);
    __m512i seq_512i = _mm512_setr_epi32 (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512i seq_512i_x2 = _mm512_setr_epi32 (0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);

    size_t p = 0;
    while(p < data_size){
        // Set up of the values and keys.
        uint32_t inputValue = input[p];
        uint32_t hash_key = this->m_hash_function(inputValue,HSIZE);
        __m512i value_vector = _mm512_set1_epi32(inputValue);

        // We have three cases:
        //      A: We found the Value we searched for.
        //      B: We found an empty spot.
        //      C: We found neither so we continue the search.
        while(1){
            // it is possible that our index overflows. With this masked we can correct it.
            int32_t overflow = (hash_key + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);
            
            //load dat
            __m512i nextElements = _mm512_maskz_loadu_epi32(overflow_correction_mask, &hashVec[hash_key]);   
            __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, value_vector, nextElements);

            if(compareRes != 0){
            // A    -   Increment Count
                __m512i nextCounts = _mm512_mask_loadu_epi32(zero_512i, oneMask, &countVec[hash_key]);
                nextCounts = _mm512_mask_add_epi32(nextCounts, compareRes, nextCounts, one_512i);
                _mm512_mask_storeu_epi32(&countVec[hash_key],compareRes,nextCounts);
                break;
            }else{
                __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(), nextElements);
                uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                if(innerMask != 0){
                // B    -   Register Value at the first empty position and set count to 1
                    __mmask16 mask1 = _mm512_knot(innerMask);
                    uint32_t pos = __builtin_ctz(checkForFreeSpace);
                    hashVec[hash_key+pos] = (uint32_t)inputValue;
                    countVec[hash_key+pos]++;
                    break;
                }else{
                // C    -   Increase hashkey to find the next spot
                    hash_key += 16;
                    if(hash_key >= HSIZE){
                        hash_key = 0;
                    }
                }
            }
        }
        p++;
    }
}

template class AVX512_gc_SoA_v1<uint32_t>;
template class AVX512_gc_SoA_v1<uint64_t>;