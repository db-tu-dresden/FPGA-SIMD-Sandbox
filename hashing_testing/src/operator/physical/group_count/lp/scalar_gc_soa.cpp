#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "operator/physical/group_count/lp/scalar_gc_soa.hpp"

#define EMPTY_SPOT 0

template <typename T>
Scalar_gc_SoA<T>::Scalar_gc_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend, int32_t bonus_scale)
    : Group_count<T>(HSIZE, hash_function, bonus_scale)
{   
    size_t elements = HSIZE;
    if(extend){
        size_t help = (512/8) / sizeof(T);
        elements = (elements + help - 1) / help; // how many vectors
        elements = elements * help;
    }
    elements *= bonus_scale;
    m_hash_vec = (T*) aligned_alloc(64, elements * sizeof(T));
    m_count_vec = (T*) aligned_alloc(64, elements * sizeof(T));
    for(size_t i = 0; i < elements; i++){
        m_hash_vec[i] = 0;
        m_count_vec[i] = 0;
    }
}

template <typename T>
Scalar_gc_SoA<T>::~Scalar_gc_SoA(){
    free(m_hash_vec);
    m_hash_vec = nullptr;
    free(m_count_vec);
    m_count_vec = nullptr;
}

template <typename T>
void Scalar_gc_SoA<T>::create_hash_table(T* input, size_t data_size){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    while(p < data_size){
        int error = 0;
        // get the possible possition of the element.
        T hash_key = this->m_hash_function(input[p], HSIZE);
        
        while(1){
            // get the value of this position
            T value = m_hash_vec[hash_key];
            
            // Check if it is the correct spot
            if(value == input[p]){
                m_count_vec[hash_key]++;
                break;
            
            // Check if the spot is empty
            }else if(value == EMPTY_SPOT){
                m_hash_vec[hash_key] = input[p];
                m_count_vec[hash_key] = 1;
                break;
            
            }
            // else{
                //go to the next spot
            ++hash_key;
            if(hash_key >= this->m_HSIZE_v){
                hash_key = 0;
            }
                // hash_key = (hash_key + 1) % HSIZE;
                //we assume that the hash_table is big enough
            // }
        }
        p++;
    }
}

template <typename T>
T Scalar_gc_SoA<T>::get(T input){
    size_t rounds = 0;
    size_t HSIZE = this->m_HSIZE;
    
    T hash_key = this->m_hash_function(input, HSIZE);

    while(rounds <= 1){
        T value = this->m_hash_vec[hash_key];
        
        if(value == input){
            return this->m_count_vec[hash_key];
        }else if(value == EMPTY_SPOT){
            return 0;
        }else{
            hash_key = (hash_key + 1) % HSIZE;
            rounds += (hash_key == 0);
        }
    }
    return 0;
}

template <typename T>
void Scalar_gc_SoA<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    if(horizontal){
        
        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << i;
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << m_hash_vec[i];
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
            std::cout << "\t" << m_count_vec[i];
            count += m_count_vec[i];
        }
        std::cout << std::endl << "Total Count:\t" << count << std::endl;
    }
    else{
        for(size_t i = 0; i < HSIZE; i++){
            std::cout << i << "\t" << m_hash_vec[i] << "\t" << m_count_vec[i] << std::endl;
            count += m_count_vec[i];
        }
        std::cout << "Total Count:\t" << count << std::endl;
    }
}

template <typename T>
void Scalar_gc_SoA<T>::clear(){
    for(size_t i = 0; i < this->m_HSIZE; i++){
        m_hash_vec[i] = 0;
        m_count_vec[i] = 0;
    }
}

template <typename T>
size_t Scalar_gc_SoA<T>::get_HSIZE(){
    return this->m_HSIZE;
}

template class Scalar_gc_SoA<uint32_t>;
// template class Scalar_gc_SoA<uint64_t>;
