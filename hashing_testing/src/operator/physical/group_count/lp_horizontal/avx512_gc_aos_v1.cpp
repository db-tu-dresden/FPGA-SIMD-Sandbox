#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_gc_aos_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_AoS_v1<T>::AVX512_gc_AoS_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_gc_AoS<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_gc_AoS_v1<T>::~AVX512_gc_AoS_v1(){
    // free(this->m_hash_vec);
    // free(this->m_count_vec);
}


template <typename T>
std::string AVX512_gc_AoS_v1<T>::identify(){
    return "AVX512 Group Count AoS Version 1";
}



// only an 32 bit uint implementation rn, because we don't use the tvl rn. As soon as we use the TVL we should reform this code to support it.
template <>
void AVX512_gc_AoS_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    // uint32_t* hashVec = this->m_hash_vec;
    // uint32_t* countVec = this->m_count_vec;

    __mmask16 oneMask = 0xFFFF;
    __m512i zero_512i = _mm512_setzero_epi32();
    __m512i one_512i = _mm512_set1_epi32(1);
    __m512i two_512i = _mm512_set1_epi32(2);
    __m512i seq_512i = _mm512_setr_epi32 (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    // __m512i seq_512i_x2 = _mm512_setr_epi32 (0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);

    size_t index = 0;
    for(size_t p = 0; p < data_size; p++){
        // Set up of the values and keys.
        uint32_t inputValue = input[p];
        uint32_t hash_key = this->m_hash_function(inputValue,HSIZE);

        __m512i value_vector = _mm512_set1_epi32(inputValue);

        while(1){
            // it is possible that our index overflows. With this masked we can correct it.
            int32_t overflow = (hash_key + 16) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
            __mmask16 overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);
            
            index = hash_key << 1;

            //load dat
            __m512i ht_values = _mm512_mask_i32gather_epi32(zero_512i, overflow_correction_mask, seq_512i, &m_hash_table[index], 8);// with 8 we say that we want a different stride. might not work correctly 
            
            __mmask16 check_match = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, value_vector, ht_values);
            __mmask16 check_free = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(), ht_values);

            if(check_match > 0){
                // __m512i ht_count = _mm512_mask_i32gather_epi32(zero_512i, overflow_correction_mask, seq_512i, &m_hash_table[index + 1], 8);
                // ht_count = _mm512_mask_add_epi32(ht_count, check_match, ht_count, one_512i);
                // _mm512_mask_i32scatter_epi32(&countVec[index + 1], check_match, seq_512i, ht_count, 8);
                uint32_t pos = __builtin_ctz(check_match) << 1;
                m_hash_table[index + pos + 1]++;
                break;

            }else if(check_free > 0){// doing this totally scalar because why not. we could do this with gather and scatter but it does not seem to be necessary.
                uint32_t pos = __builtin_ctz(check_free) << 1;
                m_hash_table[index + pos] = (uint32_t)inputValue;
                m_hash_table[index + pos + 1] = 1;
                break;

            }
            hash_key += 16;
            hash_key = hash_key > HSIZE? 0: hash_key;
        }
    }
}

template class AVX512_gc_AoS_v1<uint32_t>;
// template class AVX512_gc_AoS_v1<uint64_t>;