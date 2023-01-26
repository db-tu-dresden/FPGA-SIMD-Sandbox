#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "fpga_group_count_soaov_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
FPGA_group_count_SoAoV_v1<T>::FPGA_group_count_SoAoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  Scalar_group_count<T>(HSIZE, hash_function, true)
{
    this->m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE_v = (HSIZE + this->m_elements_per_vector - 1) / this->m_elements_per_vector;
}

template <typename T>
FPGA_group_count_SoAoV_v1<T>::~FPGA_group_count_SoAoV_v1(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string FPGA_group_count_SoAoV_v1<T>::identify(){
    return "FPGA Group Count SoAoV Version 1";
}


// only an 32 bit uint implementation rn, because we don't use the TVL. As soon as we use the TVL we should reform this code to support it.
template <>
void FPGA_group_count_SoAoV_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    



}
/*
void print512i(__m512i a, bool newline){
    uint32_t *res = (uint32_t*) aligned_alloc(64, 1*sizeof(__m512i));
    _mm512_store_epi32 (res, a);
    for(uint32_t i = 0; i < 16; i++){
        std::cout << res[i] << "\t";
    }
    free(res);
    if(newline){
        std::cout << std::endl;
    }

}
*/

template class FPGA_group_count_SoAoV_v1<uint32_t>;
// template class FPGA_group_count_SoAoV_v1<uint64_t>;