#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soa_collision_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoA_collision_v1<T>::AVX512_group_count_SoA_collision_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_group_count_SoA_collision_v1<T>::~AVX512_group_count_SoA_collision_v1(){
    free(this->m_hash_vec);
    free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoA_collision_v1<T>::identify(){
    return "AVX512 Group Count SoA Version 3";
}


template <typename T>
void AVX512_group_count_SoA_collision_v1<T>::create_hash_table(T* input, size_t data_size){
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
void AVX512_group_count_SoA_collision_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;

    __mmask16 all_mask = 0xFFFF;

    const __m512i one = _mm512_set1_epi32 (1);
    const __m512i zero = _mm512_setzero_epi32();

    uint32_t *buffer = reinterpret_cast< uint32_t* >( _mm_malloc( 16 * sizeof( uint32_t ), 64 ) );
    size_t p = 0;
    while(p + 16 < data_size){

        // load the to aggregate data
        __m512i input_value = _mm512_maskz_loadu_epi32(all_mask, &input[p]);
        // how much the given count should be increased for the given input.
        __m512i input_add = _mm512_set1_epi32(1);

        // search for conflicts
        __m512i conflicts = _mm512_conflict_epi32(input_value);
        // masked to indicate were there is a conflict in the input_values and were not.
        __mmask16 no_conflicts_mask = _mm512_cmpeq_epi32_mask(zero, conflicts);
        __mmask16 negativ_no_conflicts_mask = _mm512_knot(no_conflicts_mask);

        // we need to store the conflicts so we can interprete them as masks. and access them.
        // we are only interested in the enties that are not zero. That means the conflict cases.
        _mm512_mask_compressstoreu_epi32(buffer, negativ_no_conflicts_mask, conflicts);
        size_t conflict_count = __builtin_popcount((uint32_t)(negativ_no_conflicts_mask));
        // add at all the places where the conflict masks indicates that there is an overlap
        for(size_t i = 0; i < conflict_count; i++){
            input_add = _mm512_mask_add_epi32(input_add, (__mmask16)(buffer[i]), input_add, one);
        }
        
        // we override the value and what to add with zero in the positions where we have a conflict.
        // NOTE: This steps might not be necessary.
        input_value = _mm512_mask_set1_epi32(input_value, negativ_no_conflicts_mask, 0);
        input_add = _mm512_mask_set1_epi32(input_add, negativ_no_conflicts_mask, 0);

        // now we can calculate the hashes.
        // for this we can store the input_value hash it and load it
        // OR we use the input and hash it save it in to buffer and than make a maskz load for the hashed data
        // OR we have a simdifyed Hash Algorithm! For the most cases we would need an avx... mod. 
        _mm512_store_epi32(buffer, input_value);
        for(size_t i = 0; i < 16; i++){
            buffer[i] = this->m_hash_function(buffer[i], HSIZE);
        }
        __m512i hash_map_position = _mm512_maskz_load_epi32(no_conflicts_mask, buffer); // these are the hash values

        do{
            // now we can gather the data from the different positions where we have no conflicts.
            __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero, no_conflicts_mask, hash_map_position, hashVec, 1);
            // with these we can calculate the different possible hits. Real hits and empty positions.
            __mmask16 foundPos = _mm512_mask_cmpeq_epi32_mask(no_conflicts_mask, input_value, hash_map_value);
            __mmask16 foundEmpty = _mm512_mask_cmpeq_epi32_mask(no_conflicts_mask, zero, hash_map_value);

            if(foundPos != 0){//A
                // Now we have to gather the count. IMPORTANT! the count is a 32bit integer. 
                    // FOR NOW THIS IS CORRECT BUT MIGHT CHANGE LATER!
                // For 64bit integers we would need to find a different solution!
                __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero, foundPos, hash_map_position, countVec, 1);
                // on this count we can know add the pre calculated values. and scatter it back to their positions
                hash_map_value = _mm512_maskz_add_epi32(foundPos, hash_map_value, input_add);
                _mm512_mask_i32scatter_epi32(countVec, foundPos, hash_map_position, hash_map_value, 1);
                
                // finaly we remove the entries we just saved from the no_conflicts_mask such that the work to be done shrinkes.
                no_conflicts_mask = _mm512_kandn(foundPos, no_conflicts_mask);
            }
            if(foundEmpty != 0){//B1
                // now we have to check for collisions to prevent two different entries to write to the same position.
                __m512i saveConflicts = _mm512_maskz_conflict_epi32(foundEmpty, hash_map_position);
                __mmask16 to_save_data = _mm512_cmpeq_epi32_mask(zero, saveConflicts);
                to_save_data = _mm512_kand(to_save_data, foundEmpty);

                // with the cleaned mask we can now save the data.
                _mm512_mask_i32scatter_epi32(hashVec, to_save_data, hash_map_position, input_value, 1);
                _mm512_mask_i32scatter_epi32(countVec, to_save_data, hash_map_position, input_add, 1);
                
                //and again we need to remove the data from the todo list
                no_conflicts_mask = _mm512_kandn(to_save_data, no_conflicts_mask);
            }
            
            // afterwards we add one on the current positions of the still to be handled values.
            hash_map_position = _mm512_maskz_add_epi32(no_conflicts_mask, hash_map_position, one);
            // Since there isn't a modulo operation we have to check if the values are bigger or equal the HSIZE AND IF we have to set them to zero
            __mmask16 tobig = _mm512_mask_cmp_epi32_mask(no_conflicts_mask, hash_map_position, _mm512_set1_epi32(HSIZE), _MM_CMPINT_NLT);
            hash_map_position = _mm512_mask_set1_epi32(hash_map_position, tobig, 0);

            // we repeat this for one vector as long as their is still a value to be saved.
        }while(no_conflicts_mask != 0);
        p += 16;
    }

    //scalar remainder
    while(p < data_size){
        int error = 0;
        // get the possible possition of the element.
        uint32_t hash_key = this->m_hash_function(input[p], HSIZE);
        
        while(1){
            // get the value of this position
            uint32_t value = hashVec[hash_key];
            
            // Check if it is the correct spot
            if(value == input[p]){
                countVec[hash_key]++;
                break;
            
            // Check if the spot is empty
            }else if(value == EMPTY_SPOT){
                hashVec[hash_key] = input[p];
                countVec[hash_key] = 1;
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

    // multiple improvements are possible:
    // 1.   we could increase the performance of the worst case first write.
    // 2.   we could absorbe the scalar remainder with overflow masks
    // these would probably have a negative impact on  the overall performance.
}


template class AVX512_group_count_SoA_collision_v1<uint32_t>;
template class AVX512_group_count_SoA_collision_v1<uint64_t>;