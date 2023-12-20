#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "operator/physical/group_count/lp/scalar_gc_aos.hpp"

#define EMPTY_SPOT 0

template <typename T>
Scalar_gc_AoS<T>::Scalar_gc_AoS(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend, int32_t bonus_scale)
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
    m_hash_table = (T*) aligned_alloc(64, elements * 2 * sizeof(T));
    for(size_t i = 0; i < elements * 2; i++){
        m_hash_table[i] = 0;
    }
}

template <typename T>
Scalar_gc_AoS<T>::~Scalar_gc_AoS(){
    free(m_hash_table);
    m_hash_table = nullptr;
}

template <typename T>
void Scalar_gc_AoS<T>::create_hash_table(T* input, size_t data_size){

    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    for(size_t p = 0; p < data_size; p++){
        // get the possible possition of the element.
        T hash_key = this->m_hash_function(input[p], HSIZE);
        size_t index = 0;
        while(1){
            // get the value of this position
            index = hash_key << 1;
            T value = m_hash_table[index];
            
            if(input[p] == value){
                // Check if it is the correct spot
                m_hash_table[index + 1]++;
                break;
            }else if(value == EMPTY_SPOT){
                // Check if the spot is empty
                m_hash_table[index] = input[p];
                m_hash_table[index + 1] = 1;
                break;
            }
            //go to the next spot
            ++hash_key;
            if(hash_key >= this->m_HSIZE){
                hash_key = 0;
            }
            // hash_key = (hash_key + 1) % HSIZE;
        }
    }
}

template <typename T>
T Scalar_gc_AoS<T>::get(T input){
    size_t rounds = 0;
    size_t HSIZE = this->m_HSIZE;
    
    size_t hash_key = this->m_hash_function(input, HSIZE);
    size_t index = hash_key << 1;
    for(size_t i = 0; i < HSIZE + 1; i++){
        T value = m_hash_table[index];
        
        if(value == input){
            return m_hash_table[index + 1];
        }else if(value == EMPTY_SPOT){
            return 0;
        }

        hash_key = (hash_key + 1) % HSIZE;
        index = hash_key << 1;
    }
    return 0;
}

template <typename T>
void Scalar_gc_AoS<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    std::cout << "TODO: implement print Scalar Group COUNT AoS\n";
    if(horizontal){
        
        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << i;
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << m_hash_table[i * 2];
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
            T value = m_hash_table[i * 2 + 1];
            std::cout << "\t" << value;
            count += value;
        }
        std::cout << std::endl << "Total Count:\t" << count << std::endl;
    }
    else{
        for(size_t i = 0; i < HSIZE; i++){
            size_t id = i * 2;
            std::cout << i << "\t" << m_hash_table[id] << "\t" << m_hash_table[id + 1] << std::endl;
            count += m_hash_table[id + 1];
        }
        std::cout << "Total Count:\t" << count << std::endl;
    }
}

template <typename T>
void Scalar_gc_AoS<T>::clear(){
    for(size_t i = 0; i < this->m_HSIZE * 2; i++){
        m_hash_table[i] = 0;
    }
}

template <typename T>
size_t Scalar_gc_AoS<T>::get_HSIZE(){
    return this->m_HSIZE;
}

template class Scalar_gc_AoS<uint32_t>;
// template class Scalar_gc_AoS<uint64_t>;
