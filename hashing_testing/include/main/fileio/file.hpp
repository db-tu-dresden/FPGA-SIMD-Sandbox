#ifndef TUD_HASHING_TESTING_FILE_IO
#define TUD_HASHING_TESTING_FILE_IO

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstdint>

#include "main/hash_function.hpp"

/// @brief creates a new empty file with the given namen. important an already existing file might be overwritten
/// @param filename the name of the file that shall be created
void create_result_file(std::string filename, size_t test_number);

/// @brief Writes all the necessary information about one benchmark run into a file
/// @param filename the name of the file
/// @param alg_identification the name of the algorithm ( can be optained using identify() )
/// @param time how long the execution took in ns
/// @param data_size how much data was processed
/// @param bytes how many bytes the base data type had
/// @param distinct_value_count how many distinct values were used during the computation
/// @param scale the scaleing factor for the hash table
/// @param HSIZE the real hash table size
/// @param hash_function_enum the enum for the hash function
/// @param seed which datageneration seed was used
/// @param rsd the run id with the same configuration
/// @param config_collision_count how many collision groups where plant
/// @param config_collition_size how big these collisions were
/// @param conig_cluster_count how many cluster were plant
/// @param config_cluster_size how big theses cluster were
/// @param config_id a general id that combines all 4 values with loss
void write_to_file( 
    std::string filename, 
    std::string alg_identification, 
    uint64_t time, 
    size_t data_size,
    size_t bytes,
    size_t distinct_value_count, 
    float scale, 
    size_t HSIZE, 
    HashFunction hash_function_enum, 
    size_t seed, 
    size_t rsd, 
    size_t config_collision_count,
    size_t config_collition_size,
    size_t conig_cluster_count,
    size_t config_cluster_size,
    size_t config_id
);


void create_strided_benchmark_result_file(std::string filename);
void write_to_strided_benchmark_file(
    std::string filename,
    std::string alg_identification,
    uint64_t time, 
    size_t data_size,
    size_t bytes,
    size_t distinct_value_count,
    double scale,
    size_t HSIZE,
    HashFunction hash_function_enum, 
    size_t seed,  
    size_t run_id,
    size_t collision_count
);

void write_to_file(
    std::string filename,
    std::string content,
    bool override = false
);
#endif //TUD_HASHING_TESTING_FILE_IO