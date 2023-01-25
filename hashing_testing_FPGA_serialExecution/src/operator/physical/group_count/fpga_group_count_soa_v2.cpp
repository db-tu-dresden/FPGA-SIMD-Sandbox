#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "fpga_group_count_soa_v2.hpp"

#define EMPTY_SPOT 0

template <typename T>
FPGA_group_count_SoA_v2<T>::FPGA_group_count_SoA_v2(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
FPGA_group_count_SoA_v2<T>::~FPGA_group_count_SoA_v2(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string FPGA_group_count_SoA_v2<T>::identify(){
    return "FPGA Group Count SoA Version 2";
}


template <typename T>
void FPGA_group_count_SoA_v2<T>::create_hash_table(T* input, size_t data_size){
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
void FPGA_group_count_SoA_v2<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
   
}

template class FPGA_group_count_SoA_v2<uint32_t>;
template class FPGA_group_count_SoA_v2<uint64_t>;