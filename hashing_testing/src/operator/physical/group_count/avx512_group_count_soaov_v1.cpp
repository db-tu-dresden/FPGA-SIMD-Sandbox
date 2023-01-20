#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soaov_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoAoV_v1<T>::AVX512_group_count_SoAoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  Scalar_group_count<T>(HSIZE, hash_function, true)
{
    // std::cout << "Before problem!\t"<< HSIZE << std::endl;
    m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE_v = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
}

template <typename T>
AVX512_group_count_SoAoV_v1<T>::~AVX512_group_count_SoAoV_v1(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoAoV_v1<T>::identify(){
    return "AVX512 Group Count SoAoV Version 1";
}




// only an 32 bit uint implementation rn, because we don't use the TVL. As soon as we use the TVL we should reform this code to support it.
template <>
void AVX512_group_count_SoAoV_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    
//TODO create a stack allocated m512i array with implicit size. it should be small enough for the heap.
// note we need 2 of these. with this we can go around 

    __m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    
    __m512i hash_map[this->m_HSIZE_v];
    __m512i count_map[this->m_HSIZE_v];

    // we need to load the data into the arrays. For the case that the group is over multiple datas.
    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        size_t h = i * this->m_elements_per_vector;
        hash_map[i] = _mm512_load_epi32(&this->m_hash_vec[h]);
        count_map[i] = _mm512_load_epi32(&this->m_count_vec[h]);
    }



    __mmask16 masks[17];
    masks[0] = _cvtu32_mask16(0);
    for(size_t i = 1; i <= 16; i++){
        masks[i] = _cvtu32_mask16(1 << (i-1));
    }

    int p = 0;
    while (p < data_size) {

        // get current input value
        uint32_t inputValue = input[p];

        // determine hash_key based on sizeRegister*16
        //uint32_t hash_key = hashx(inputValue,sizeRegister*16);
        uint32_t hash_key = this->m_hash_function(inputValue, this->m_HSIZE_v);

        // determine array position of the hash_key
        //uint32_t arrayPos = (hash_key/16);
        uint32_t arrayPos = hash_key;

        /**
        * broadcast element p of input[] to vector of type __m512i
        * broadcastCurrentValue contains sixteen times value of input[i]
        **/
        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);
        // this->print(true);
        // std::cout << "\nval "<< inputValue << "\thash_key " << hash_key << "\n\n";
        while(1) {
            // compare vector with broadcast value against vector with following elements for equality
            __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, hash_map[arrayPos]);

            // found match
            if (compareRes > 0) {
                count_map[arrayPos] = _mm512_mask_add_epi32(count_map[arrayPos], compareRes, count_map[arrayPos], oneM512iArray);

                p++;
                break;

            } else { // no match found
                // deterime free position within register
                __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), hash_map[arrayPos]);
                if(checkForFreeSpace > 0) {                // CASE B1    
                    uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
                    
                    //store key
                    hash_map[arrayPos] = _mm512_mask_set1_epi32(hash_map[arrayPos],masks[pos],inputValue);
                    //set count to one
                    count_map[arrayPos] = _mm512_mask_set1_epi32(count_map[arrayPos],masks[pos],1);
                    p++;
                    break;
                }   else    { 
                    arrayPos = (arrayPos + 1)%this->m_HSIZE_v;
                }
            }
        }
    }

    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        size_t h = i * this->m_elements_per_vector;
        _mm512_store_epi32(&this->m_hash_vec[h], hash_map[i]);
        _mm512_store_epi32(&this->m_count_vec[h], count_map[i]);
    }

}

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




template class AVX512_group_count_SoAoV_v1<uint32_t>;
// template class AVX512_group_count_SoAoV_v1<uint64_t>;