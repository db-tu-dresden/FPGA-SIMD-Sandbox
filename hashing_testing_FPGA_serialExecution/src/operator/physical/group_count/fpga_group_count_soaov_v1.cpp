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
    
// @TODO create a stack allocated m512i array (=AVX version, here of type fpvec<uint32_t>) with implicit size. 
// it should be small enough for the heap.
// note we need 2 of these. with this we can go around 

    uint32_t one = 1;
    uint32_t zero = 0;
    fpvec<uint32_t> oneMask = set1(one);
    fpvec<uint32_t> zeroMask = set1(zero);
    fpvec<uint32_t> zeroM512iArray = set1(zero);
    fpvec<uint32_t> oneM512iArray = set1(one);


// @TODO    storing a fpvec<uint32_t> into a fpvec<uint32_t> at position i currently not working
//          Restructuring of the data structure fpvec<T> necessary?
// hash_map & count_map currently contains only one element of type fpvec<uint32_t> at position 0, because
// this element has already a size of 512 bit
    fpvec<fpvec<uint32_t>> hash_map;
    fpvec<fpvec<uint32_t>> count_map;
    for (int i=0; i<(64/sizeof(uint32_t)); i++) {
        hash_map.elements[i] = set1((uint32_t)(this->m_HSIZE_v));
        count_map.elements[i] = set1((uint32_t)(this->m_HSIZE_v));
	}


// due to error/todo from line 51, 
    // loading data. On the first exec this should result in only 0 vals.
    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        size_t h = i * this->m_elements_per_vector;

        hash_map.elements[i] = load_epi32(oneMask, this->m_hash_vec, h, this->m_HSIZE_v);
        count_map.elements[i] = load_epi32(oneMask, this->m_count_vec, h, this->m_HSIZE_v);

        // additional function to store a single element isn't needed anymore
        //storeSingleElement_vec(&hash_map, (uint32_t)i, data_hashMap);
        //storeSingleElement_vec(&count_map, (uint32_t)i, data_countMap);
    }



// @TODO    fpvec<uint32_t> with 17 elements - how?
//          define two data structures to handle all 17 elements?
//  old     __mmask16 masks[17];
/* 
    //creating writing masks
    fpvec<fpvec<uint32_t>> masks;
    masks.elements[0] = cvtu32_mask16(0);
    for(size_t i = 1; i <= 16; i++){
        masks.elements[i] = cvtu32_mask16(1 << (i-1));
    }
*/

/*      ==== following Intel Intrinsics need to be replaced by functions from primitives.hpp 
    int p = 0;
    while (p < data_size) {

        uint32_t inputValue = input[p];
        uint32_t hash_key = this->m_hash_function(inputValue, this->m_HSIZE_v);

        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);

        while(1) {
            // compare vector with broadcast value against vector with following elements for equality
            __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, hash_map[hash_key]);

            // found match
            if (compareRes > 0) {
                count_map[hash_key] = _mm512_mask_add_epi32(count_map[hash_key], compareRes, count_map[hash_key], oneM512iArray);

                p++;
                break;
            } else { // no match found
                // deterime free position within register
                __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), hash_map[hash_key]);
                if(checkForFreeSpace > 0) {                // CASE B1    
                    uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
                    
                    //store key
                    hash_map[hash_key] = _mm512_mask_set1_epi32(hash_map[hash_key], masks[pos], inputValue);
                    //set count to one
                    count_map[hash_key] = _mm512_mask_set1_epi32(count_map[hash_key], masks[pos], 1);
                    p++;
                    break;
                }   else    { // CASE B2
                    hash_key = (hash_key + 1) % this->m_HSIZE_v;
                }
            }
        }
    }

    //store data
    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        size_t h = i * this->m_elements_per_vector;
        _mm512_store_epi32(&this->m_hash_vec[h], hash_map[i]);
        _mm512_store_epi32(&this->m_count_vec[h], count_map[i]);
    }*/
}


template class FPGA_group_count_SoAoV_v1<uint32_t>;
// template class FPGA_group_count_SoAoV_v1<uint64_t>;