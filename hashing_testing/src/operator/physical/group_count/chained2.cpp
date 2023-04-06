#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "chained2.hpp"

#define EMPTY_SPOT 0

template <typename T>
Chained2<T>::Chained2(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend, int32_t bonus_scale)
    : Group_count<T>(HSIZE, hash_function, bonus_scale)
{   
    this->map = new std::unordered_map<Key, T, KeyHasher>();
}

template <typename T>
Chained2<T>::~Chained2(){
    delete map;
}

template <typename T>
std::string Chained2<T>::identify(){
    return "Chained2 Group Count AoS";
}

template <typename T>
void Chained2<T>::create_hash_table(T* input, size_t data_size){
    size_t p = 0;
    size_t HSIZE = this->m_HSIZE;
    // Iterate over input
    for(size_t p = 0; p < data_size; p++){
        (*map)[{input[p], this->m_HSIZE}]++;
    }
}

template <typename T>
T Chained2<T>::get(T input){
    return (*map)[{input, this->m_HSIZE}];
}

template <typename T>
void Chained2<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    std::cout << "unprintable\n";
}

template <typename T>
void Chained2<T>::clear(){
    map->clear();
}

template <typename T>
size_t Chained2<T>::get_HSIZE(){
    return this->m_HSIZE;
}

template class Chained2<uint32_t>;
// template class Chained2<uint64_t>;
