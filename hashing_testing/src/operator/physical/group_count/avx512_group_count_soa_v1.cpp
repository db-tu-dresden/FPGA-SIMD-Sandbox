#include <stdlib.h>
#include <stdint.h>
#include <iostream>

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

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soa_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoA_v1<T>::AVX512_group_count_SoA_v1(size_t HSIZE, T (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_group_count_SoA_v1<T>::~AVX512_group_count_SoA_v1(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoA_v1<T>::identify(){
    return "AVX512 Group Count SoA Version 1";
}


template <typename T>
void AVX512_group_count_SoA_v1<T>::create_hash_table(T* input, size_t dataSize){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    while(p < dataSize){
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
void print512_num(__m512i var) ;
template <>
void AVX512_group_count_SoA_v1<uint32_t>::create_hash_table(uint32_t* input, size_t dataSize){
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;

    __mmask16 oneMask = 0xFFFF;
    __m512i zeroM512iArray = _mm512_setzero_epi32();
    __m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
            std::cout << "onemask:\t";this->printBits(2,&oneMask);

    size_t p = 0;
    this->print(true);
    while(p < dataSize){
        uint32_t inputValue = input[p];
        uint32_t hash_key = this->m_hash_function(inputValue,HSIZE);
        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);
        std::cout << "inputValue:\t" << inputValue << "\thash_key\t"<< hash_key <<std::endl;
        while(1){
            int32_t overflow = (hash_key + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);
            
            std::cout << "overflow:\t";this->printBits(2,&overflow_correction_mask_i);
            __m512i nextElements = _mm512_maskz_loadu_epi32(overflow_correction_mask, &hashVec[hash_key]);   
            __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
            std::cout << "compareV:\t";this->printBits(2,&compareRes);
            
            print512_num(broadcastCurrentValue);
            print512_num(nextElements);

            if(compareRes != 0){
                std::cout << "MATCH" << std::endl;
                __m512i nextCounts = _mm512_mask_loadu_epi32(zeroM512iArray, oneMask, &countVec[hash_key]);
                nextCounts = _mm512_mask_add_epi32(nextCounts, compareRes, nextCounts, oneM512iArray);
                _mm512_mask_storeu_epi32(&countVec[hash_key],compareRes,nextCounts);
                break;
            }else{
                print512_num(_mm512_setzero_epi32());
                __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(), nextElements);
                uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                std::cout << "compareZ:\t"; this->printBits(2,&innerMask);
                if(innerMask != 0){
                    __mmask16 mask1 = _mm512_knot(innerMask);
                    uint32_t pos = __builtin_ctz(checkForFreeSpace);
                    hashVec[hash_key+pos] = (uint32_t)inputValue;
                    countVec[hash_key+pos]++;
                    std::cout << "place at:\t"<< hash_key + pos << std::endl;
                    break;
                }else{   
                    std::cout << "not found at:\n";
                    hash_key += 16;
                    if(hash_key >= HSIZE){
                        hash_key = 0;
                    }
                }
            }
        }
        this->print(true);
        std::cout << "\n";

        p++;
    }
}

// print function for vector of type __m512i
void print512_num(__m512i var) {
    uint32_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Content of __m512i Array: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}

template class AVX512_group_count_SoA_v1<uint32_t>;
template class AVX512_group_count_SoA_v1<uint64_t>;