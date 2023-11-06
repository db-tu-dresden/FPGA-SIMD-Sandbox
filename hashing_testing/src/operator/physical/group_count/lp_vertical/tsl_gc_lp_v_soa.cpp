#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "operator/physical/group_count/lp_vertical/tsl_gc_lp_v_soa.hpp"

#define EMPTY_SPOT 0


template <class SimdT, typename T>
TSL_gc_LP_V_SoA<SimdT, T>::TSL_gc_LP_V_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_gc_SoA<T>(HSIZE, hash_function)
{}


template <class SimdT, typename T>
TSL_gc_LP_V_SoA<SimdT, T>::~TSL_gc_LP_V_SoA(){
    // free(this->m_hash_vec);
    // free(this->m_count_vec);
}

// todo find annother way to remove limitation of gather, since current implementation limits to type size. 
template <class SimdT, typename T>
void TSL_gc_LP_V_SoA<SimdT, T>::create_hash_table(T* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;
    T* key_array = this->m_hash_vec;
    T* count_array = this->m_count_vec;

    size_t elements_per_vector = 16;//TODO!
    size_t N = 4; //TODO. scale for gather scatter

    vec_t zero_vec = tsl::set1<ps>(0);
    vec_t one_vec = tsl::set1<ps>(1);
    T *buffer = reinterpret_cast< T* >( _mm_malloc( 256 * sizeof( T ), 64 ) );


    size_t p = 0;
    while(p + 16 <= data_size){

        // load the to aggregate data
        vec_t input_value = tsl::load<ps>(&input[p]);
        // how much the given count should be increased for the given input.
        vec_t input_add = tsl::set1<ps>(1);

        // search for conflicts
        vec_t conflicts = tsl::conflict<ps>(input_value);
        // masked to indicate were there is a conflict in the input_values and were not.
        mask_t no_conflicts_mask = tsl::equal<ps>(zero_vec, conflicts);
        mask_t conflicts_mask = tsl::mask_binary_not(no_conflicts_mask);

        // we need to store the conflicts so we can interprete them as masks. and access them.
        // we are only interested in the enties that are not zero. That means the conflict cases.
        tsl::compress_store<ps>(conflicts_mask, buffer, conflicts);
        size_t conflict_count = tsl::pc<ps>(tsl::to_integral<ps>(conflicts_mask)); //__builtin_popcount((uint32_t)(conflicts_mask)); //todo
        // add at all the places where the conflict masks indicates that there is an overlap
        for(size_t i = 0; i < conflict_count; i++){
            input_add = tsl::add<ps>(tsl::to_mask<ps>(buffer[i]),input_add, one_vec);
            // input_add = _mm512_mask_add_epi32(input_add, (__mmask16)(buffer[i]), input_add, one_vec);
        }
        // we override the value and what to add with zero in the positions where we have a conflict.
        // NOTE: This steps might not be necessary.
        input_value = tsl::masked_set1<ps>(input_value, conflicts_mask, 0);
        input_add = tsl::masked_set1<ps>(input_add, conflicts_mask, 0);

        // now we can calculate the hashes.
        // for this we can store the input_value hash it and load it
        // OR we use the input and hash it save it in to buffer and than make a maskz load for the hashed data
        // OR we have a simdifyed Hash Algorithm! For the most cases we would need an avx... mod. 
        // _mm512_store_epi32(buffer, input_value);
        for(size_t i = 0; i < 16; i++){
            buffer[i] = this->m_hash_function(input[p + i], HSIZE);
        }
        // __m512i hash_map_position = _mm512_maskz_load_epi32(no_conflicts_mask, buffer); // these are the hash values
        vec_t hash_map_position = tsl::loadu<ps>(no_conflicts_mask, buffer);
        do{
            // now we can gather the data from the different positions where we have no conflicts.
            vec_t hash_map_value = tsl::gather<ps>(N, no_conflicts_mask, zero_vec, key_array, hash_map_position);

            // __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero_vec, no_conflicts_mask, hash_map_position, key_array, 4);
            // with these we can calculate the different possible hits. Real hits and empty positions.
            mask_t found_pos = tsl::equal<ps>(no_conflicts_mask, input_value, hash_map_value);
            mask_t found_empty = tsl::equal<ps>(no_conflicts_mask, zero_vec, hash_map_value);

            if(found_pos != 0){//A
                // Now we have to gather the count. IMPORTANT! the count is a 32bit integer. 
                    // FOR NOW THIS IS CORRECT BUT MIGHT CHANGE LATER!
                // For 64bit integers we would need to find a different solution!
                vec_t hash_map_value = tsl::gather<ps>(N, found_pos, zero_vec, count_array, hash_map_position);//_mm512_mask_i32gather_epi32(zero_vec, found_pos, hash_map_position, count_array, 4);
                // on this count we can know add the pre calculated values. and scatter it back to their positions
                hash_map_value = tsl::add<ps>(found_pos, hash_map_value, input_add);
                tsl::scatter(N, found_pos, hash_map_value, count_array, hash_map_position);
                // _mm512_mask_i32scatter_epi32(count_array, found_pos, hash_map_position, hash_map_value, 4);

                
                // finaly we remove the entries we just saved from the no_conflicts_mask such that the work to be done shrinkes.
                // no_conflicts_mask = _mm512_kandn(found_pos, no_conflicts_mask);
                no_conflicts_mask = tsl::mask_binary_and<ps>(tsl::mask_binary_not<ps>(found_pos), no_conflicts_mask);
            }
            if(found_empty != 0){//B1
                // now we have to check for conflicts to prevent two different entries to write to the same position.
                __m512i saveConflicts = _mm512_maskz_conflict_epi32(found_empty, hash_map_position);
                __m512i empty = tsl::set1<ps>(tsl::to_integral<ps>(found_empty));
                saveConflicts = tsl::and<ps>(saveConflicts, empty);

                __mmask16 to_save_data = tsl::equal<ps>(zero_vec, saveConflicts);
                to_save_data = tsl::mask_binary_and<ps>(tsl::mask_binary_not<ps>(to_save_data), found_empty);
                // to_save_data = _mm512_kand(to_save_data, found_empty);

                // with the cleaned mask we can now save the data.
                _mm512_mask_i32scatter_epi32(key_array, to_save_data, hash_map_position, input_value, 4);
                _mm512_mask_i32scatter_epi32(count_array, to_save_data, hash_map_position, input_add, 4);
                
                //and again we need to remove the data from the todo list
                // no_conflicts_mask = _mm512_kandn(to_save_data, no_conflicts_mask);
                no_conflicts_mask = tsl::mask_binary_and<ps>(tsl::mask_binary_not<ps>(to_save_data), no_conflicts_mask);
            }
            
            // afterwards we add one on the current positions of the still to be handled values.
            hash_map_position = tsl::add<ps>(no_conflicts_mask, hash_map_position, one_vec);
            // Since there isn't a modulo operation we have to check if the values are bigger or equal the HSIZE AND IF we have to set them to zero
            __mmask16 to_big = _mm512_mask_cmp_epi32_mask(no_conflicts_mask, hash_map_position, _mm512_set1_epi32(HSIZE), _MM_CMPINT_NLT);
            hash_map_position = _mm512_mask_set1_epi32(hash_map_position, to_big, 0);

            // we repeat this for one vector as long as their is still a value to be saved.
        }while(no_conflicts_mask != 0);
        p += 16;
    }


    
    //scalar remainder
    //without special tsl implementation
    while(p < data_size){
        T input_value = input[p];
        size_t hash_key = this->m_hash_function(input_value, HSIZE);
        
        while(1){
            // get the value of this position
            T key_value = key_array[hash_key];
            
            if(key_value == input_value){ // Check if it is the correct spot
                count_array[hash_key]++;
                break;
            
            }
            else if(key_value == EMPTY_SPOT){ // Check if the spot is empty
                key_array[hash_key] = input_value;
                count_array[hash_key] = 1;
                break;
            }
            //go to the next spot
            hash_key = hash_key + 1;
            if(hash_key >= HSIZE){
                hash_key = 0;
            }
        }
        p++;
    }
    free(buffer);
}

// template class TSL_gc_LP_V_SoA<uint32_t>;
// template class TSL_gc_LP_V_SoA<uint64_t>;