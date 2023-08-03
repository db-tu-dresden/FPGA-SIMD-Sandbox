#ifndef TUD_HASHING_TESTING_DATAGENERATOR
#define TUD_HASHING_TESTING_DATAGENERATOR

#include "main/datagenerator/datagen_help.hpp"
#include "main/hash_function.hpp"

template<typename T>
size_t generate_strided_data(
    T*& result, 
    size_t data_size,
    size_t distinct_values,
    size_t hsize,
    hash_fptr<T> hash_function,
    size_t collision_size,
    size_t seed,
    bool non_collisions_first = true,
    bool evenly_distributed = true
);

template<typename T>
void generate_strided_data_raw(
    std::vector<T> &collision_data,
    std::vector<T> &non_collision_data,
    size_t distinct_values,
    size_t hsize,
    hash_fptr<T> hash_function,
    size_t collision_size,
    size_t &seed
);


#endif