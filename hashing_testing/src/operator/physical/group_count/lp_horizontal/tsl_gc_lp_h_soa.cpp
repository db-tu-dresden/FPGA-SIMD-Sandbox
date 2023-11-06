#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "operator/physical/group_count/lp_horizontal/tsl_gc_lp_h_soa.hpp"

#define EMPTY_SPOT 0


template <class SimdT, typename T>
TSL_gc_LP_H_SoA<SimdT, T>::TSL_gc_LP_H_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_gc_SoA<T>(HSIZE, hash_function)
{}


template <class SimdT, typename T>
TSL_gc_LP_H_SoA<SimdT, T>::~TSL_gc_LP_H_SoA(){
    // free(this->m_hash_vec);
    // free(this->m_count_vec);
}

// todo:
//  test
// implement masked load
// implement masked store
// test ctz
template <class SimdT, typename T>
void TSL_gc_LP_H_SoA<SimdT, T>::create_hash_table(T* input, size_t data_size){

    using ps = tsl::simd<T, SimdT>;
    using vec_t = typename ps::register_type;
    using mask_t = typename ps::mask_type;
    using imask_t = typename ps::imask_type;


    size_t HSIZE = this->m_HSIZE;
    T* hashVec = this->m_hash_vec;
    T* countVec = this->m_count_vec;


    size_t vec_elements = 1; //TODO!

    vec_t zero_vec = tsl::set1<ps>(0);
    vec_t one_vec = tsl::set1<ps>(1);
    vec_t seq_vec = tsl::sequence<ps>();
    vec_t seq_x2_vec = tsl::add<ps>(seq_vec, seq_vec);
    
    mask_t oneMask = tsl::equal<ps>(zero_vec, zero_vec);
    // mask_t oneMask = tsl::to_mask<ps>( tsl::integral_all_true<ps>() );


    // __mmask16 oneMask = 0xFFFF;
    // __m512i zero_512i = _mm512_setzero_epi32();
    // __m512i one_512i = _mm512_set1_epi32(1);
    // __m512i two_512i = _mm512_set1_epi32(2);
    // __m512i seq_512i = _mm512_setr_epi32 (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    // __m512i seq_512i_x2 = _mm512_setr_epi32 (0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);

    for(size_t p = 0; p < data_size; p++){

        // Set up of the values and keys.
        T inputValue = input[p];
        T hash_key = this->m_hash_function(inputValue, HSIZE);
        vec_t value_vec = tsl::set1<ps>(inputValue);
        // __m512i value_vector = _mm512_set1_epi32(inputValue);

        // We have three cases:
        //      A: We found the Value we searched for.
        //      B: We found an empty spot.
        //      C: We found neither so we continue the search.
        while(1){
            int64_t overflow = (hash_key + vec_elements) - HSIZE;
            overflow = overflow < 0? 0: overflow;
            imask_t overflow_correction_imask = tsl::to_integral<ps>(oneMask) >> overflow;
            mask_t overflow_correction_mask = tsl::to_mask<ps>(overflow_correction_imask);

            // it is possible that our index overflows. With this masked we can correct it.
            // int64_t overflow = (hash_key + vec_elements) - HSIZE;
            // overflow = overflow < 0? 0: overflow;
            // uint32_t overflow_correction_mask_i = (1 << (vec_elements-overflow)) - 1; 
            // mask_t overflow_correction_mask = _cvtu32_mask16(overflow_correction_mask_i);
            
            //load data
            vec_t next_elements = tsl::loadu<ps>(overflow_correction_mask, &hashVec[hash_key]);
            mask_t compare_res = tsl::equal<ps>(overflow_correction_mask, value_vec, next_elements);
            // __m512i nextElements = _mm512_maskz_loadu_epi32(overflow_correction_mask, &hashVec[hash_key]);   
            // __mmask16 compareRes = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, value_vector, nextElements);
            // __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(), nextElements);

            // if(compareRes != 0){
            if(tsl::to_integral(compare_res) != 0){
            // A    -   Increment Count
                mask_t next_counts = tsl::loadu<ps>(overflow_correction_mask, &countVec[hash_key]);
                next_counts = tsl::add<ps>(compare_res, next_counts, one_vec);
                tsl::storeu<ps>(&countVec[hash_key], compare_res, next_counts);


                // __m512i nextCounts = _mm512_mask_loadu_epi32(zero_512i, oneMask, &countVec[hash_key]);
                // nextCounts = _mm512_mask_add_epi32(nextCounts, compareRes, nextCounts, one_512i);
                // _mm512_mask_storeu_epi32(&countVec[hash_key], compareRes, nextCounts);
                break;
            }else{
                mask_t check_for_free_space = tsl::equal<ps>(overflow_correction_mask, zero_vec, next_elements);
                imask_t check_for_free_space_i = tsl::to_integral<ps>(check_for_free_space);

                // __mmask16 checkForFreeSpace = _mm512_mask_cmpeq_epi32_mask(overflow_correction_mask, _mm512_setzero_epi32(), nextElements);
                // uint32_t innerMask = _mm512_mask2int(checkForFreeSpace);
                if(check_for_free_space_i != 0){
                // B    -   Register Value at the first empty position and set count to 1
                    // __mmask16 mask1 = _mm512_knot(innerMask);
                    uint64_t pos = tsl::tzc<ps>(check_for_free_space_i);
                    
                    // uint32_t pos = __builtin_ctz(check_for_free_space);
                    hashVec[hash_key + pos] = (uint32_t)inputValue;
                    countVec[hash_key + pos] = 1;
                    break;
                }else{
                // C    -   Increase hashkey to find the next spot
                    hash_key += vec_elements;
                    if(hash_key >= HSIZE){
                        hash_key = 0;
                    }
                }
            }
        }
    }
}

// template class TSL_gc_LP_H_SoA<uint32_t>;
// template class TSL_gc_LP_H_SoA<uint64_t>;