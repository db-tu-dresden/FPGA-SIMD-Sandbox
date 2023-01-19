#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soa_v3.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoA_v3<T>::AVX512_group_count_SoA_v3(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_group_count_SoA_v3<T>::~AVX512_group_count_SoA_v3(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoA_v3<T>::identify(){
    return "AVX512 Group Count SoA Version 3";
}


template <typename T>
void AVX512_group_count_SoA_v3<T>::create_hash_table(T* input, size_t data_size){
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
void AVX512_group_count_SoA_v3<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;

    __mmask16 oneMask = 0xFFFF;
    __m512i zeroM512iArray = _mm512_setzero_epi32();
    __m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

    int p = 0;
    while (p < data_size) {

        uint32_t inputValue = input[p];

        uint32_t hash_key = this->m_hash_function(inputValue,HSIZE);

        uint32_t aligned_start = (hash_key/16)*16;
        uint32_t remainder = hash_key - aligned_start; // should be equal to hash_key % 16

        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);
        while (1) {

            int32_t overflow = (aligned_start + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);

            // int32_t cutlow = 16 - remainder; // should be in a range from 1-16
            // uint32_t cutlow_mask_i = 0xFFFF;
            // cutlow_mask_i <<= remainder;

            // uint32_t combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
            // __mmask16 overflow_and_cutlow_mask = _cvtu32_mask16(combined_mask_i);

            __m512i nextElements = _mm512_load_epi32(&hashVec[aligned_start]);
            
            __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
    
    
            if (compareRes != 0) {
                uint32_t matchPos = __builtin_ctz(compareRes); 
                countVec[aligned_start+matchPos]++;
                p++;
                break;
            }   else {
                __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(),nextElements);
                uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                if(innerMask != 0) {                // CASE B1    

                    uint32_t pos = __builtin_ctz(checkForFreeSpace);
                    
                    hashVec[aligned_start+pos] = (uint32_t)inputValue;
                    countVec[aligned_start+pos]++;
                    p++;
                    break;
                }   else    {                   // CASE B2

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

template <typename T>
T AVX512_group_count_SoA_v3<T>::get(T input){
    size_t rounds = 0;
    size_t HSIZE = this->m_HSIZE;
    
    size_t ele_count = (512/8) / sizeof(T);

    T hash_key = this->m_hash_function(input, HSIZE);
    uint64_t aligned_start = (hash_key/ele_count)*ele_count;
    bool empty_spot = false;
    bool roundtrip = false;

    while(rounds <= 1){
        T value = this->m_hash_vec[aligned_start];
        
        if(value == input){
            return this->m_count_vec[aligned_start];
        }else if(value == EMPTY_SPOT){
            empty_spot == true;
            aligned_start++;
        }else{
            aligned_start = (aligned_start + 1) % HSIZE;
            rounds += (aligned_start == 0);
            roundtrip |= (aligned_start == 0);
        }
        if(((aligned_start + 1) % ele_count == 0 || roundtrip) && empty_spot){
            return 0;
        }
    }
    return 0;
}


template class AVX512_group_count_SoA_v3<uint32_t>;
template class AVX512_group_count_SoA_v3<uint64_t>;