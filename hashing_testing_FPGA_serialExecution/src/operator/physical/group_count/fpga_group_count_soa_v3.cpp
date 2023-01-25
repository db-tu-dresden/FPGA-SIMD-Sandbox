#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "fpga_group_count_soa_v3.hpp"

#define EMPTY_SPOT 0

template <typename T>
FPGA_group_count_SoA_v3<T>::FPGA_group_count_SoA_v3(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
FPGA_group_count_SoA_v3<T>::~FPGA_group_count_SoA_v3(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string FPGA_group_count_SoA_v3<T>::identify(){
    return "FPGA Group Count SoA Version 3";
}


template <typename T>
void FPGA_group_count_SoA_v3<T>::create_hash_table(T* input, size_t data_size){
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
void FPGA_group_count_SoA_v3<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    
}

template <typename T>
T FPGA_group_count_SoA_v3<T>::get(T input){
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


template class FPGA_group_count_SoA_v3<uint32_t>;
template class FPGA_group_count_SoA_v3<uint64_t>;