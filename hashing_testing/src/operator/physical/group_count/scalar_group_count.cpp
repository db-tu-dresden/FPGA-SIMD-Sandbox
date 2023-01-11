#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "scalar_group_count.hpp"

#define EMPTY_SPOT 0

template <typename T>
Scalar_group_count<T>::Scalar_group_count(size_t HSIZE, T (*hash_function)(T, size_t))
    : Group_count<T>(HSIZE, hash_function)
{
    // m_HSIZE = HSIZE;
    // m_hash_function = hash_function;
    m_hash_vec = (T*) aligned_alloc(64, HSIZE * sizeof(T));
    m_count_vec = (T*) aligned_alloc(64, HSIZE * sizeof(T));

    for(size_t i = 0; i < HSIZE; i++){
        m_hash_vec[i] = 0;
        m_count_vec[i] = 0;
    }
    // initialize(m_hash_vec, HSIZE, EMPTY_SPOT);
    // initialize(m_count_vec, HSIZE, 0);
}

template <typename T>
Scalar_group_count<T>::~Scalar_group_count(){
    free(m_hash_vec);
    free(m_count_vec);
}

template <typename T>
void Scalar_group_count<T>::create_hash_table(T* input, size_t dataSize){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input 
    while(p < dataSize){
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
            else{
                //go to the next spot
                hash_key = (hash_key + 1) % HSIZE;
                //we assume that the hash_table is big enough
            }
        }
        p++;
    }
}

template <typename T>
T Scalar_group_count<T>::get(T input){
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
void Scalar_group_count<T>::print(){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

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

template <typename T>
void Scalar_group_count<T>::print2(){
    size_t count =0;
    size_t HSIZE = this->m_HSIZE;

    for(size_t i = 0; i < HSIZE; i++){
        std::cout << i << "\t" << m_hash_vec[i] << "\t" << m_count_vec[i] << std::endl;
        count += m_count_vec[i];
    }
    std::cout << "Total Count:\t" << count << std::endl;
}

template <typename T>
std::string Scalar_group_count<T>::identify(){
    return "Scalar Group Count";
}


template class Scalar_group_count<uint32_t>;
template class Scalar_group_count<uint64_t>;