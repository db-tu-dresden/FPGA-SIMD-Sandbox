#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soa_conflict_v2.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoA_conflict_v2<T>::AVX512_group_count_SoA_conflict_v2(size_t HSIZE, size_t (*hash_function)(T, size_t))
    : Scalar_group_count<T>(HSIZE, hash_function)
{}

template <typename T>
AVX512_group_count_SoA_conflict_v2<T>::~AVX512_group_count_SoA_conflict_v2(){
    // free(this->m_hash_vec);
    // free(this->m_count_vec);
}


template <typename T>
std::string AVX512_group_count_SoA_conflict_v2<T>::identify(){
    return "AVX512 Group Count SoA conflict Version 2";
}


template <typename T>
void AVX512_group_count_SoA_conflict_v2<T>::create_hash_table(T* input, size_t data_size){
    
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


//_mm512_permutevar_epi32   _mm512_maskz_permutexvar_epi32
//_mm512_mask_blend_epi32
//_mm512_maskz_compress_epi32
//different version where we constantly try to reload data to push in more data
template <>
void AVX512_group_count_SoA_conflict_v2<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    // std::cout << "WOW we get here!\n";
    size_t HSIZE = this->m_HSIZE;
    uint32_t* hashVec = this->m_hash_vec;
    uint32_t* countVec = this->m_count_vec;
    
    const __mmask16 all_mask = 0xFFFF;

    const __m512i one = _mm512_set1_epi32 (1);
    const __m512i zero = _mm512_setzero_epi32();
    const __m512i shift_0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    uint32_t *buffer = reinterpret_cast< uint32_t* >( _mm_malloc( 16 * sizeof( uint32_t ), 64 ) );
    
    __mmask16 raw_data_left_mask = 0;
    int32_t raw_data_left = 0;

    __m512i raw_data = _mm512_setzero_epi32(), raw_hash = _mm512_setzero_epi32();


    __m512i work_todo = _mm512_setzero_epi32(), work_add = _mm512_setzero_epi32(), work_hash = _mm512_setzero_epi32();
    __mmask16 work_todo_mask = 0, no_work_todo_mask;

    // std::cout << "one:\t";  print512i(one);
    // std::cout << "zero:\t"; print512i(zero);
    // std::cout << "shift_0:\t"; print512i(shift_0);
    uint32_t entries_left = __builtin_popcount((uint32_t)(work_todo_mask));
    size_t p = 0;
    while(p + 16 <= data_size || raw_data_left > 0 || entries_left > 0){
        // std::cout << "+++++++++++++++++++++++++++++\n";
        // std::cout << "\traw_mask:\t"; printMask(raw_data_left_mask);
        // std::cout << "\traw_data:\t"; print512i(raw_data);
        // std::cout << "\traw_hash:\t"; print512i(raw_hash);
        
        // std::cout << "\t\t------\n";

        // std::cout << "\twork_mask:\t"; printMask(work_todo_mask);
        // std::cout << "\twork_todo:\t"; print512i(work_todo);
        // std::cout << "\twork_hash:\t"; print512i(work_hash);
        // std::cout << "\twork_add:\t"; print512i(work_add);

        // reload data
        if(raw_data_left <= 0 && p + 16 <= data_size){
            // std::cout << "\t---reload---\n";
            raw_data_left_mask = all_mask;
            raw_data_left = 16;
            raw_data = _mm512_load_epi32(&input[p]);
            for(size_t i = 0; i < 16; i++){
                buffer[i] = this->m_hash_function(input[p + i], HSIZE);
            }
            raw_hash = _mm512_load_epi32(buffer);
            p += 16;

            // std::cout << "\traw_mask:\t"; printMask(raw_data_left_mask);
            // std::cout << "\traw_data:\t"; print512i(raw_data);
            // std::cout << "\traw_hash:\t"; print512i(raw_hash);
        }


        uint32_t entries_to_fill = 16 - entries_left;
        // std::cout << "\tentries left:\t" << entries_left << "\tentries_to_fill:\t" << entries_to_fill << std::endl;

        // std::cout << "\t---compress-and-shuffle---\n";
        //compress 
        if(entries_left < 16){
            work_todo = _mm512_mask_compress_epi32(raw_data, work_todo_mask, work_todo);
            work_hash = _mm512_mask_compress_epi32(raw_hash, work_todo_mask, work_hash);
            work_add  = _mm512_mask_compress_epi32(one, work_todo_mask, work_add);
            //create_new_mask
            work_todo_mask = (1 << entries_left) - 1; 
            __mmask16 help_mask = _mm512_knot(work_todo_mask);
            work_todo_mask = work_todo_mask | raw_data_left_mask;

            raw_data_left = raw_data_left - entries_to_fill;

            if(raw_data_left > 0){
                size_t help_trailing_zero = __builtin_ctz(raw_data_left_mask);
                raw_data_left_mask = raw_data_left_mask >> (help_trailing_zero + entries_to_fill);
                raw_data_left_mask = raw_data_left_mask << (help_trailing_zero + entries_to_fill);

                __m512i help_id = _mm512_maskz_sub_epi32(raw_data_left_mask, shift_0, _mm512_set1_epi32(entries_to_fill));
                raw_data = _mm512_maskz_permutexvar_epi32(raw_data_left_mask, help_id, raw_data);
                raw_hash = _mm512_maskz_permutexvar_epi32(raw_data_left_mask, help_id, raw_hash);
            }else{
                raw_data_left = 0;
                raw_data_left_mask = 0;
                raw_data = zero;
                raw_hash = zero;
            }
        }

        // std::cout << "\traw_mask:\t"; printMask(raw_data_left_mask);
        // std::cout << "\traw_data:\t"; print512i(raw_data);
        // std::cout << "\traw_hash:\t"; print512i(raw_hash);

        // std::cout << "\t\t------\n";
        
        // std::cout << "\twork_mask:\t"; printMask(work_todo_mask);
        // std::cout << "\twork_todo:\t"; print512i(work_todo);
        // std::cout << "\twork_hash:\t"; print512i(work_hash);
        // std::cout << "\twork_add:\t"; print512i(work_add);

        // load the to aggregate data
        // how much the given count should be increased for the given input.

        // search for conflicts
        // std::cout << "\t---conflict-detection-and-cleaning---\n";
        __m512i work_conflicts = _mm512_conflict_epi32(work_todo);
        // std::cout << "\tconflict:\t"; print512i(work_conflicts);
        // masked to indicate were there is a conflict in the input_values and were not.
        work_todo_mask = _mm512_mask_cmp_epi32_mask(work_todo_mask, zero, work_conflicts, _MM_CMPINT_EQ);
        no_work_todo_mask = _mm512_cmp_epi32_mask(zero, work_conflicts, _MM_CMPINT_NE);


        // we need to store the conflicts so we can interprete them as masks. and access them.
        // we are only interested in the enties that are not zero. That means the conflict cases.
        size_t conflict_count = __builtin_popcount((uint32_t)(no_work_todo_mask));
        _mm512_mask_compressstoreu_epi32(buffer, no_work_todo_mask, work_conflicts);
        // add at all the places where the conflict masks indicates that there is an overlap
        for(size_t i = 0; i < conflict_count; i++){
            work_add = _mm512_mask_add_epi32(work_add, (__mmask16)(buffer[i]), work_add, one);
        }
        // we override the value and what to add with zero in the positions where we have a conflict.
        // NOTE: This steps might not be necessary.
        __mmask16 inv_work_todo_mask = _mm512_knot(work_todo_mask);
        work_todo = _mm512_mask_set1_epi32(work_todo, inv_work_todo_mask, 0);
        work_hash = _mm512_mask_set1_epi32(work_hash, inv_work_todo_mask, 0);
        work_add  = _mm512_mask_set1_epi32(work_add, inv_work_todo_mask, 0);


        // std::cout << "\twork_mask:\t"; printMask(work_todo_mask);
        // std::cout << "\twork_todo:\t"; print512i(work_todo);
        // std::cout << "\twork_hash:\t"; print512i(work_hash);
        // std::cout << "\twork_add:\t"; print512i(work_add);


        // now we can gather the data from the different positions where we have no conflicts.
        // with these we can calculate the different possible hits. Real hits and empty positions.
        __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero, work_todo_mask, work_hash, hashVec, 4);
        __mmask16 foundPos = _mm512_mask_cmpeq_epi32_mask(work_todo_mask, work_todo, hash_map_value);
        __mmask16 foundEmpty = _mm512_mask_cmpeq_epi32_mask(work_todo_mask, zero, hash_map_value);


        // std::cout << "\t---FOUND---\n";
        if(foundPos != 0){//A
            // Now we have to gather the count. IMPORTANT! the count is a 32bit integer. 
                // FOR NOW THIS IS CORRECT BUT MIGHT CHANGE LATER!
            // For 64bit integers we would need to find a different solution!
            __m512i hash_map_value = _mm512_mask_i32gather_epi32(zero, foundPos, work_hash, countVec, 4);
            // on this count we can know add the pre calculated values. and scatter it back to their positions
            hash_map_value = _mm512_maskz_add_epi32(foundPos, hash_map_value, work_add);
            _mm512_mask_i32scatter_epi32(countVec, foundPos, work_hash, hash_map_value, 4);
            
            // finaly we remove the entries we just saved from the no_conflicts_mask such that the work to be done shrinkes.
            work_todo_mask = _mm512_kandn(foundPos, work_todo_mask);
        }

        // std::cout << "\twork_mask:\t"; printMask(work_todo_mask);
        // std::cout << "\twork_todo:\t"; print512i(work_todo);
        // std::cout << "\twork_hash:\t"; print512i(work_hash);
        // std::cout << "\twork_add:\t"; print512i(work_add);

        // std::cout << "\t---FOUND-EMPTY---\n";
        if(foundEmpty != 0){//B1
            // now we have to check for conflicts to prevent two different entries to write to the same position.
            __m512i saveConflicts = _mm512_maskz_conflict_epi32(foundEmpty, work_hash);
            __m512i empty = _mm512_set1_epi32(foundEmpty);
            saveConflicts = _mm512_and_epi32(saveConflicts, empty);

            __mmask16 to_save_data = _mm512_cmpeq_epi32_mask(zero, saveConflicts);
            to_save_data = _mm512_kand(to_save_data, foundEmpty);

            // with the cleaned mask we can now save the data.
            _mm512_mask_i32scatter_epi32(hashVec, to_save_data, work_hash, work_todo, 4);
            _mm512_mask_i32scatter_epi32(countVec, to_save_data, work_hash, work_add, 4);
            
            //and again we need to remove the data from the todo list
            work_todo_mask = _mm512_kandn(to_save_data, work_todo_mask);
        }
        
        // std::cout << "\twork_mask:\t"; printMask(work_todo_mask);
        // std::cout << "\twork_todo:\t"; print512i(work_todo);
        // std::cout << "\twork_hash:\t"; print512i(work_hash);
        // std::cout << "\twork_add:\t"; print512i(work_add);
        // std::cout << "\t-------------\n";

        // afterwards we add one on the current positions of the still to be handled values.
        work_hash = _mm512_maskz_add_epi32(work_todo_mask, work_hash, one);
        // Since there isn't a modulo operation we have to check if the values are bigger or equal the HSIZE AND IF we have to set them to zero
        __mmask16 tobig = _mm512_mask_cmp_epi32_mask(work_todo_mask, work_hash, _mm512_set1_epi32(HSIZE), _MM_CMPINT_NLT);
        work_hash = _mm512_mask_set1_epi32(work_hash, tobig, 0);
        entries_left = __builtin_popcount((uint32_t)(work_todo_mask));  
        // we repeat this for one vector as long as their is still a value to be saved.
        

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

template class AVX512_group_count_SoA_conflict_v2<uint32_t>;
template class AVX512_group_count_SoA_conflict_v2<uint64_t>;