#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "fpga_group_count_soa_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
FPGA_group_count_SoA_v1<T>::FPGA_group_count_SoA_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
FPGA_group_count_SoA_v1<T>::~FPGA_group_count_SoA_v1(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string FPGA_group_count_SoA_v1<T>::identify(){
    return "FPGA Group Count SoA Version 1";
}


template <typename T>
void FPGA_group_count_SoA_v1<T>::create_hash_table(T* input, size_t data_size){
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
void FPGA_group_count_SoA_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;

    uint32_t one = 1;
    uint32_t zero = 0;
    fpvec<uint32_t> oneMask = set1(one);
    fpvec<uint32_t> zeroMask = set1(zero);  
    fpvec<uint32_t> zeroM512iArray = set1(zero);
    fpvec<uint32_t> oneM512iArray = set1(one);

    size_t p = 0;
    while(p < data_size){
        // Set up of the values and keys.
        uint32_t inputValue = input[p];
        uint32_t hash_key = this->m_hash_function(inputValue,HSIZE);
        fpvec<uint32_t> broadcastCurrentValue = set1(inputValue);

        // We have three cases:
        //      A: We found the Value we searched for.
        //      B: We found an empty spot.
        //      C: We found neither so we continue the search.
        while(1){
            // it is possible that our index overflows. With this masked we can correct it.
            int32_t overflow = (hash_key + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            fpvec<uint32_t> overflow_correction_mask = cvtu32_mask16(overflow_correction_mask_i);
            
            //load data
            fpvec<uint32_t> nextElements = mask_loadu(overflow_correction_mask, hashVec, hash_key, HSIZE);   
            fpvec<uint32_t> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);

            if((mask2int(compareRes)) != 0){
            // A    -   Increment Count
                fpvec<uint32_t> nextCounts = mask_loadu(oneMask, countVec, hash_key, HSIZE);
                nextCounts = mask_add_epi32(nextCounts, compareRes, nextCounts, oneM512iArray);
                mask_storeu_epi32(countVec, hash_key, HSIZE, compareRes,nextCounts);
                break;
            }else{
                fpvec<uint32_t> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_correction_mask, zeroMask, nextElements);
                uint32_t innerMask = mask2int(checkForFreeSpace);
                if(innerMask != 0){
                // B    -   Register Value at the first empty position and set count to 1
                    //fpvec<uint32_t> mask1 = knot(innerMask);    // not used anymore?
                    uint32_t pos = ctz_onceBultin(checkForFreeSpace);
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

template class FPGA_group_count_SoA_v1<uint32_t>;
template class FPGA_group_count_SoA_v1<uint64_t>;