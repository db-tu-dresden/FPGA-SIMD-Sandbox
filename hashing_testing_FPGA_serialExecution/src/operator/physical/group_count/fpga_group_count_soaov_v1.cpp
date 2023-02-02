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
    this->m_HSIZE = HSIZE;
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

    fpvec<uint32_t>* hash_map;
    fpvec<uint32_t>* count_map;

    // use a vectore with elements of type <fpvec<uint32_t> as structure "around" the registers
    hash_map = new fpvec<uint32_t>[this->m_HSIZE_v];
    count_map = new fpvec<uint32_t>[this->m_HSIZE_v];

    // loading data. On the first exec this should result in only 0 vals.   
    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        size_t h = i * this->m_elements_per_vector;

        hash_map[i] = load_epi32(oneMask, this->m_hash_vec, h, this->m_HSIZE);
        count_map[i] = load_epi32(oneMask, this->m_count_vec, h, this->m_HSIZE);
    }

///    
// !!! ATTENTION - changed indizes !!!
// mask with only 0 => zero_cvtu32_mask
// masks = array of 16 masks respectively fpvec<uint32_t> with one 1 at unique positions 
///
    // creating writing masks
    fpvec<uint32_t> zero_cvtu32_mask = cvtu32_mask16((uint32_t)0);
    std::array<fpvec<uint32_t>, 16> masks {};
    for(uint32_t i = 1; i <= 16; i++){
        masks[i-1] = cvtu32_mask16((uint32_t)(1 << (i-1)));
    }

    int p = 0;
    while (p < data_size) {
        uint32_t inputValue = input[p];
        uint32_t hash_key = this->m_hash_function(inputValue, this->m_HSIZE_v);

        fpvec<uint32_t> broadcastCurrentValue = set1(inputValue);

        while(1) {
            // compare vector with broadcast value against vector with following elements for equality
            fpvec<uint32_t> compareRes = cmpeq_epi32_mask(broadcastCurrentValue, hash_map[hash_key]);

            // found match
            if (mask2int(compareRes) != 0) {
                count_map[hash_key] = mask_add_epi32(count_map[hash_key], compareRes, count_map[hash_key], oneM512iArray);

                p++;
                break;
            } else { // no match found
                // deterime free position within register
                fpvec<uint32_t> checkForFreeSpace = cmpeq_epi32_mask(zeroMask, hash_map[hash_key]);
                if(mask2int(checkForFreeSpace) != 0) {                // CASE B1    
// old : uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
// --> omit +1, because masks with only 0 at every position is outsourced to zero_cvtu32_mask                
                    uint32_t pos = ctz_onceBultin(checkForFreeSpace);

                    //store key
                    hash_map[hash_key] = mask_set1(hash_map[hash_key], masks[pos], inputValue);
                    //set count to one
                    count_map[hash_key] = mask_set1(count_map[hash_key], masks[pos], 1);
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
        store_epi32(this->m_hash_vec, h, hash_map[i]);
        store_epi32(this->m_count_vec, h, count_map[i]);
    }
}

template class FPGA_group_count_SoAoV_v1<uint32_t>;
// template class FPGA_group_count_SoAoV_v1<uint64_t>;