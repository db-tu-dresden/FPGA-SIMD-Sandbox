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
    :  Group_count<T>(HSIZE, hash_function)
{
    // std::cout << "Before problem!\t"<< HSIZE << std::endl;
    m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
    this->help = (T*) aligned_alloc(64, 2 * sizeof(__m512i));
    this->m_hash_vec = (__m512i*) aligned_alloc(512, m_elements_per_vector*sizeof(__m512i));
    this->m_count_vec = (__m512i*) aligned_alloc(512, m_elements_per_vector*sizeof(__m512i));
    
    for(size_t i = 0; i < this->m_HSIZE; i++){
        this->m_hash_vec[i] = _mm512_setzero_si512();
        this->m_count_vec[i] = _mm512_setzero_si512();
    }

}

template <typename T>
AVX512_group_count_SoAoV_v1<T>::~AVX512_group_count_SoAoV_v1(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
    free(this->help);
}


template <typename T>
std::string AVX512_group_count_SoAoV_v1<T>::identify(){
    return "AVX512 Group Count SoAoV Version 1";
}

//needs to do a memcopy of the vectors into an array such stat we can use it in a scalar way
// template <typename T>
// void AVX512_group_count_SoAoV_v1<T>::create_hash_table(T* input, size_t data_size){
//     size_t p = 0;
//     size_t HSIZE = this->m_HSIZE;
//     // Iterate over input 
//     while(p < data_size){
//         // get the possible possition of the element.
//         T hash_key = this->m_hash_function(input[p], HSIZE);
//         while(1){
//             // get the value of this position
//             T value = this->m_hash_vec[hash_key];
            
//             // Check if it is the correct spot
//             if(value == input[p]){
//                 this->m_count_vec[hash_key]++;
//                 break;
//             // Check if the spot is empty
//             }else if(value == EMPTY_SPOT){
//                 this->m_hash_vec[hash_key] = input[p];
//                 this->m_count_vec[hash_key] = 1;
//                 break;
//             }
//             else{
//                 //go to the next spot
//                 hash_key = (hash_key + 1) % HSIZE;
//                 //we assume that the hash_table is big enough
//             }
//         }
//         p++;
//     }
// }


// only an 32 bit uint implementation rn, because we don't use the TVL. As soon as we use the TVL we should reform this code to support it.
template <>
void AVX512_group_count_SoAoV_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    
    __m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    
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
        uint32_t hash_key = this->m_hash_function(inputValue, this->m_HSIZE);

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
            __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, this->m_hash_vec[arrayPos]);

            // found match
            if (compareRes > 0) {
                this->m_count_vec[arrayPos] = _mm512_mask_add_epi32( this->m_count_vec[arrayPos], compareRes ,  this->m_count_vec[arrayPos], oneM512iArray);

                p++;
                break;

            } else { // no match found
                // deterime free position within register
                __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), this->m_hash_vec[arrayPos]);
                if(checkForFreeSpace > 0) {                // CASE B1    
                    uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
                    
                    //store key
                    this->m_hash_vec[arrayPos] = _mm512_mask_set1_epi32(this->m_hash_vec[arrayPos],masks[pos],inputValue);
                    //set count to one
                    this->m_count_vec[arrayPos] = _mm512_mask_set1_epi32(this->m_count_vec[arrayPos],masks[pos],1);
                    p++;
                    break;
                }   else    { 
                    arrayPos = (arrayPos + 1)%this->m_HSIZE;
                }
            }
        }
    }
}

template <typename T>
void AVX512_group_count_SoAoV_v1<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    if(horizontal){

        for(size_t i = 0; i < HSIZE; i++){
            print512i(m_hash_vec[i]);
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
            print512i(m_count_vec[i]);
        }
    }
    else{
        std::cout << "not printable!\n";
    }
}


template <typename T>
T AVX512_group_count_SoAoV_v1<T>::get(T input){

    size_t hash = this->m_hash_function(input, this->m_HSIZE);
    size_t k = 0; 
    __m512i val = _mm512_set1_epi32(input);
    while(true){
        __mmask16 compareRes1 = _mm512_cmpeq_epi32_mask(val, this->m_hash_vec[hash]);
        __mmask16 compareRes2 = _mm512_cmpeq_epi32_mask(val, _mm512_setzero_si512());

        uint32_t val_match_pos = __builtin_ctz(compareRes1);
        
        if(compareRes1 != 0){
        
            _mm512_store_epi32 (help, m_count_vec[hash]);
            T count = help[val_match_pos];

            return count;
        }else if(compareRes2 != 0){
            return 0;
        }
        hash = (hash + 1) % this->m_HSIZE;
        // if(k++ > this->m_HSIZE * 3){
        //     std::cout << input << " lead to an error!" << std::endl;
        //     this->print(true);
        //     return 0;
        // }
    }
    return 0;
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