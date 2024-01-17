
#ifndef TUD_HASHING_TESTING_AVX512_MAIN_TSL_BENCHMARK
#define TUD_HASHING_TESTING_AVX512_MAIN_TSL_BENCHMARK

#include <stdlib.h>
#include <cstdint>
#include "main/utility.hpp"
#include "main/hash_function.hpp"
#include "operator/physical/group_count/group_count_handler/group_count_algorithms.hpp"
#include "main/datagenerator/datagenerator.hpp"

void build_benchmark(
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    size_t* hash_table_locations,           //where to create the hash table
    size_t num_hash_table_locations,
    size_t* build_data_locations,           // where to create data 
    size_t num_build_data_locations,
    size_t repeats_different_data,          // how often to repeat all the experiments with different data
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    size_t repeats_different_layout,        // how often to use a different layout for the data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    Base_Datatype* datatypes_undertest,     // which datatypes should be tested
    size_t num_datatypes_undertest,
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
    HashFunction* hashfunctions_undertest,  // which hashfunctions to use for the test
    size_t num_hashfunctions_undertest,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,               // division differance between tests
    size_t seed = 0
);

void build_benchmark_template_helper_base(
    std::string result_file_name, 
    std::string config_string,
    std::string result_string,
    Base_Datatype base,    
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    size_t* hash_table_locations,  //where to create the hash table
    size_t num_hash_table_locations,
    size_t* build_data_locations,  // where to create data 
    size_t num_build_data_locations,
    size_t repeats_different_data,          // how often to repeat all the experiments with different data
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    size_t repeats_different_layout,        // how often to use a different layout for the data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    HashFunction* hashfunctions_undertest,  // which hashfunctions to use for the test
    size_t num_hashfunctions_undertest,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,               // division differance between tests
    size_t& seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T, class Vec> 
void build_benchmark_datagen(
    std::string result_file_name,
    std::string config_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t* build_data_locations,  // where to create data 
    size_t num_build_data_locations,
        size_t repeats_different_data,          // how often to repeat all the experiments with different data
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
        size_t repeats_different_layout,        // how often to use a different layout for the data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
        HashFunction* hashfunctions_undertest,  // which hashfunctions to use for the test
        size_t num_hashfunctions_undertest,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,              // division differance between tests
    size_t& seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T, class Vec>
void build_benchmark_data(
    std::string result_file_name,
    std::string config_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t* build_data_locations,  // where to create data 
    size_t num_build_data_locations,
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    hash_fptr<T> function,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,              // division differance between tests
    Datagenerator<T> *datagen,    
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T, class Vec> 
void build_benchmark_final(
    std::string result_file_name,
    std::string config_string,
    T* data,
    size_t data_size,
    size_t hsize,
    hash_fptr<T> function,
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t repeats_same_data               // how often to repeat all the experiments with the same data

);

template <class T>
size_t run_test_build(Group_Count_TSL_SOA<T>*& group_count, T* data, size_t data_size, bool cleanup = false, bool reset = true);

template <typename T>
size_t run_test_build(Group_Count_TSL_SOA<T>*& group_count, T* data, T* result, size_t data_size);


void probe_benchmark(
    const size_t distinct_value_count,                  // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,                      // how many values should be probed for
    const size_t probe_data_count,
    size_t* hash_table_locations,                       // where to create the hash table
    size_t num_hash_table_locations,            
    size_t* probe_locations,
    size_t num_probe_locations,            
    size_t repeats_different_data,                      // how often to repeat all the experiments with different data
    size_t repeats_same_data,                           // how often to repeat all the experiments with the same data
    size_t repeats_different_layout,                    // how often to use a different layout for the data
    Group_Count_Algorithm_TSL* algorithms_undertest,    // which algorithms to test
    size_t num_algorithms_undertest,    
    Base_Datatype* datatypes_undertest,                 // which datatypes should be tested
    size_t num_datatypes_undertest,             
    Vector_Extention* extentions_undertest,             // which vector extentions should be tested
    size_t num_extentions_undertest,            
    HashFunction* hashfunctions_undertest,              // which hashfunctions to use for the test
    size_t num_hashfunctions_undertest,             
    double* scale_factors,                              // what different scale factors to use during testing
    size_t num_scale_factors,                       
    size_t max_collision_size,                          // maximum collisions
    size_t num_collision_test,                          // number different collision test to do
    size_t collision_diminish,                          // division differance between tests
    float *selectivities,
    size_t num_selectivities,
    size_t seed = 0    
);

void probe_benchmark_template_helper_base(
    std::string result_file_name, 
    std::string config_string,
    std::string result_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t probe_data_count,          // how many values should be included in the dataset 
    const size_t probe_size,
    Base_Datatype base,    
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
    size_t* hash_table_locations,  //where to create the hash table
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    size_t repeats_different_data,          // how often to repeat all the experiments with different data
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    size_t repeats_different_layout,        // how often to use a different layout for the data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    HashFunction* hashfunctions_undertest,  // which hashfunctions to use for the test
    size_t num_hashfunctions_undertest,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,               // division differance between tests
    float* selectivities,
    size_t num_selectivities,
    size_t& seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T> 
void probe_benchmark_datagen(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    const size_t probe_size,
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
    size_t*hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    size_t repeats_different_data,          // how often to repeat all the experiments with different data
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    size_t repeats_different_layout,        // how often to use a different layout for the data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    HashFunction* hashfunctions_undertest,  // which hashfunctions to use for the test
    size_t num_hashfunctions_undertest,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,              // division differance between tests
    float* selectivities,
    size_t num_selectivities,
    size_t& seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T>
void probe_benchmark_data(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset
    const size_t probe_size,
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
    T* data, 
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    hash_fptr<T> function,
    double* scale_factors,                  // what different scale factors to use during testing
    size_t num_scale_factors,           
    size_t max_collision_size,              // maximum collisions
    size_t num_collision_test,              // number different collision test to do
    size_t collision_diminish,              // division differance between tests
    float *selectivities,
    size_t num_selectivities,
    Datagenerator<T> *datagen,    
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T>
void probe_benchmark_vector_extention(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,    
    const size_t build_data_count,
    const size_t probe_data_count,
    Datagenerator<T> *datagen,
    T* data,
    size_t hsize,
    hash_fptr<T> function,
    Vector_Extention ve,
    Group_Count_Algorithm_TSL* algorithms_undertest,
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    float *selectivities,
    size_t num_selectivities,
    size_t repeats_same_data,
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T, class Vec>
void probe_benchmark_hash_table(
 std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t build_data_count,
    const size_t probe_data_count,
    Datagenerator<T> *datagen,
    T* data,
    size_t hsize,
    hash_fptr<T> function,
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    float *selectivities,
    size_t num_selectivities,
    size_t repeats_same_data,
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
);

template<typename T, class Vec> 
void probe_benchmark_final(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t probe_data_count,
    Datagenerator<T> *datagen,
    Group_Count_TSL_SOA<T> *alg,
    size_t* probe_locations,
    size_t num_probe_locations,
    float *selectivities,
    size_t num_selectivities,
    size_t repeats_same_data,
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
);

#endif //TUD_HASHING_TESTING_AVX512_MAIN_TSL_BENCHMARK