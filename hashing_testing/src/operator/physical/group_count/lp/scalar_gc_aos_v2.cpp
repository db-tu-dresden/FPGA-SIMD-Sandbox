#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "operator/physical/group_count/lp/scalar_gc_aos_v2.hpp"

#define EMPTY_SPOT 0

template <typename T>
Scalar_gc_AoS_V2<T>::Scalar_gc_AoS_V2(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend, int32_t bonus_scale)
    : Group_count<T>(HSIZE, hash_function, bonus_scale)
{   
    size_t elements = HSIZE;
    if(extend){
        size_t help = (512/8) / sizeof(T);
        elements = (elements + help - 1) / help; // how many vectors
        elements = elements * help;
    }
    elements *= bonus_scale;

    // we allocate twice the size because of the AoS 
    m_hash_table = (Entry<T>*) aligned_alloc(64, elements * sizeof(Entry<T>));
    for(size_t i = 0; i < elements; i++){
        m_hash_table[i] = {EMPTY_SPOT, 0};
    }
}

template <typename T>
Scalar_gc_AoS_V2<T>::~Scalar_gc_AoS_V2(){
    free(m_hash_table);
    m_hash_table = nullptr;
}

template <typename T>
void Scalar_gc_AoS_V2<T>::create_hash_table(T* input, size_t data_size){
    size_t HSIZE = this->m_HSIZE;

    // Iterate over input 
    for(size_t p = 0; p < data_size; p++){
        // get the possible possition of the element.
        T hash_key = this->m_hash_function(input[p], HSIZE);
        // std::cout << input[p] << "\t->\t" << hash_key << std::endl; 
        while(1){
            // get the value of this position
            T key = m_hash_table[hash_key].key;
            // std::cout << "\t" << hash_key <<":\t" << key << std::endl;

            if(key == input[p]){
                // Check if it is the correct spot
                m_hash_table[hash_key].value++;
                
                // std::cout << "=>\t" << hash_key <<":\t[" << m_hash_table[hash_key].key << ", "<< m_hash_table[hash_key].value << "]" << std::endl;
                break;
            }else if(key == EMPTY_SPOT){
                // Check if the spot is empty
                m_hash_table[hash_key].key = input[p];
                m_hash_table[hash_key].value = 1;
                break;
            }
            //go to the next spot
            hash_key = (hash_key + 1) % HSIZE;
        }
    }
}

template <typename T>
T Scalar_gc_AoS_V2<T>::get(T input){
    size_t rounds = 0;
    size_t HSIZE = this->m_HSIZE;
    
    size_t hash_key = this->m_hash_function(input, HSIZE);
    size_t index = hash_key << 1;
    for(size_t i = 0; i < HSIZE + 1; i++){
        T key = m_hash_table[hash_key].key;
        
        if(key == input){
            return m_hash_table[hash_key].value;
        }else if(key == EMPTY_SPOT){
            return 0;
        }

        hash_key = (hash_key + 1) % HSIZE;
    }
    return 0;
}

template <typename T>
void Scalar_gc_AoS_V2<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    if(horizontal){
        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << i;
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << m_hash_table[i].key;
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
            std::cout << "\t" << m_hash_table[i].value;
            count += m_hash_table[i].value;
        }
        std::cout << std::endl << "Total Count:\t" << count << std::endl;
    }
    else{
        for(size_t i = 0; i < HSIZE; i++){
            std::cout << i << "\t" << m_hash_table[i].key << "\t" << m_hash_table[i].value << std::endl;
            count += m_hash_table[i].value;
        }
        std::cout << "Total Count:\t" << count << std::endl;
    }
}

template <typename T>
void Scalar_gc_AoS_V2<T>::clear(){
    for(size_t i = 0; i < this->m_HSIZE; i++){
        m_hash_table[i] = {0,0};
    }
}

template <typename T>
size_t Scalar_gc_AoS_V2<T>::get_HSIZE(){
    return this->m_HSIZE;
}

template class Scalar_gc_AoS_V2<uint32_t>;
// template class Scalar_gc_AoS_V2<uint64_t>;
