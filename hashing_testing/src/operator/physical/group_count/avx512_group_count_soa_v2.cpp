#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "avx512_group_count_soa_v2.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoA_v2<T>::AVX512_group_count_SoA_v2(size_t HSIZE, T (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_group_count_SoA_v2<T>::~AVX512_group_count_SoA_v2(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoA_v2<T>::identify(){
    return "Scalar Group Count Soa Version 1";
}


template <typename T>
void AVX512_group_count_SoA_v2<T>::create_hash_table(T* input, size_t dataSize){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    while(p < dataSize){
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

template class AVX512_group_count_SoA_v2<uint32_t>;
template class AVX512_group_count_SoA_v2<uint64_t>;