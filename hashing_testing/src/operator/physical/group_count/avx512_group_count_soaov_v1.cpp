#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "avx512_group_count_soaov_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_group_count_SoAoV_v1<T>::AVX512_group_count_SoAoV_v1(size_t HSIZE, T (*hash_function)(T, size_t))
    :  Group_count<T>(HSIZE, hash_function)
{
    m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector; 
    this->m_hash_vec[5];
    this->m_count_vec[5];
    std::cout << HSIZE << "\twowie\t" << this->m_HSIZE << std::endl;
    for(size_t i = 0; i < this->m_HSIZE; i++){
        this->m_hash_vec[i] = _mm512_setzero_si512();
        this->m_count_vec[i] = _mm512_setzero_si512();
        std::cout << "seg fault here!\n";
    }

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
    for(size_t i = 1; i < 16; i++){
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

        while(1) {
            
            // compare vector with broadcast value against vector with following elements for equality
            __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, this->m_hash_vec[arrayPos]);
    
            uint32_t matchPos = (32-__builtin_clz(compareRes)); 

        // cout <<p<<" "<<inputValue<<" + matchPos="<<matchPos<<endl;
        // cout <<inputValue<<" "<<hash_key<<" "<<arrayPos<<" "<<matchPos<<endl;

            // found match
            if (matchPos>0) {
                //cout <<"Update key"<<endl;
                this->m_count_vec[arrayPos] = _mm512_mask_add_epi32( this->m_count_vec[arrayPos], compareRes ,  this->m_count_vec[arrayPos], oneM512iArray);
            //  print512_num(countVecs[arrayPos]);
                p++;
                break;

            } else { // no match found
                // deterime free position within register
                __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), this->m_hash_vec[arrayPos]);
                if(checkForFreeSpace != 0) {                // CASE B1    
                //  cout <<"Input new key: "<<endl;
                    __mmask16 mask1 = _mm512_knot(checkForFreeSpace);   
                    uint32_t pos = (32-__builtin_clz(mask1))%16;
                    pos = pos+1;

                    //store key
                    this->m_hash_vec[arrayPos] = _mm512_mask_set1_epi32(this->m_hash_vec[arrayPos],masks[pos],inputValue);

                    //set count to one
                    this->m_count_vec[arrayPos] = _mm512_mask_set1_epi32(this->m_count_vec[arrayPos],masks[pos],1);
                    //print512_num(countVecs[arrayPos]);

                    p++;
                    break;
                }   else    { 
                    //cout<<"B2"<<endl;
                    arrayPos = (arrayPos++)%this->m_HSIZE;
                }
            }
        }
    }
}

template <typename T>
void AVX512_group_count_SoAoV_v1<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    std::cout << "print not yet supported!\n";
    // if(horizontal){
        
    //     for(size_t i = 0; i < HSIZE; i++){
    //             std::cout << "\t" << i;
    //     }
    //     std::cout << std::endl;

    //     for(size_t i = 0; i < HSIZE; i++){
    //             std::cout << "\t" << m_hash_vec[i];
    //     }
    //     std::cout << std::endl;

    //     for(size_t i = 0; i < HSIZE; i++){
    //         std::cout << "\t" << m_count_vec[i];
    //         count += m_count_vec[i];
    //     }
    //     std::cout << std::endl << "Total Count:\t" << count << std::endl;
    // }
    // else{
    //     for(size_t i = 0; i < HSIZE; i++){
    //         std::cout << i << "\t" << m_hash_vec[i] << "\t" << m_count_vec[i] << std::endl;
    //         count += m_count_vec[i];
    //     }
    //     std::cout << "Total Count:\t" << count << std::endl;
    // }
}


template <typename T>
T AVX512_group_count_SoAoV_v1<T>::get(T input){

    return 0;
}


template class AVX512_group_count_SoAoV_v1<uint32_t>;
// template class AVX512_group_count_SoAoV_v1<uint64_t>;