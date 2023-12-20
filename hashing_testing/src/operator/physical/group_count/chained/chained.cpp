#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "operator/physical/group_count/chained/chained.hpp"

#define EMPTY_SPOT 0

template <typename T>
Chained<T>::Chained(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend, int32_t bonus_scale)
    : Group_count<T>(HSIZE, hash_function, bonus_scale)
{   
    this->map = new std::unordered_map<T,T>();
}

template <typename T>
Chained<T>::~Chained(){
    delete map;
}

template <typename T>
void Chained<T>::create_hash_table(T* input, size_t data_size){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input
    for(size_t p = 0; p < data_size; p++){
        (*map)[input[p]]++;
    }
}

template <typename T>
T Chained<T>::get(T input){
    return (*map)[input];
}

template <typename T>
void Chained<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    std::cout << "unprintable\n";
}

template <typename T>
void Chained<T>::clear(){
    map->clear();
}

template <typename T>
size_t Chained<T>::get_HSIZE(){
    return this->m_HSIZE;
}

template class Chained<uint32_t>;
template class Chained<uint64_t>;
