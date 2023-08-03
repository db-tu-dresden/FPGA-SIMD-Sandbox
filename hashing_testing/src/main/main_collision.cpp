#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <string>

#include "operator/physical/group_count/group_count_handler/group_count_algorithms.hpp"

#include "main/benchmark/table.hpp"
#include "main/datagenerator/datagen.hpp"
#include "main/fileio/file.hpp"

#include "main/hash_function.hpp"
#include "main/utility.hpp"


// TODO
// further refactor 
// cmake!
// run test!
// p2 data generaiton
// TVL integration!
// rethink how different collisions could be passed to the given test. ATM: hard coded in the testX functions.
// documentation


//--------------------------------------------
//validation function
//--------------------------------------------

/// @brief Checks the given execution of the algorithm against a baseline
/// @tparam T the data type on which the algorithm was executed
/// @param grouping the algorithm in question
/// @param validation_baseline the baseline execution
/// @param validation_size the size of the hash table of the validation_baseline
/// @return 
template <typename T> 
bool validation(Group_count<T>* grouping, Scalar_gc_SoA<T>* validation_baseline, size_t validation_size);

//--------------------------------------------
// Time and Run functions for one Benchmark
//--------------------------------------------

/// @brief Executes the hash function and collecting performance data and checks if the given algorithm has the same result as the Scalar Group Count algorithm 
/// @tparam T the data type on which the algorithm shall be executed
/// @param group_count The algorithm (group_count operation) that shall be executed.
/// @param data The data on which the operation shall be evaluated
/// @param data_size 
/// @param validation_baseline the Scalar Group Count execution. It must have been executed on the same data before hand.
/// @param validation_size the hash table size of the validation
/// @param validate if the given execution shall be checked for errors
/// @param cleanup true if group_count should be delete when the benchmark is finished. 
/// @param reset true if the hash table and count table shall be cleared again
/// @return the time the execution took in nano seconds
template <typename T>
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, Scalar_gc_SoA<T>*& validation_baseline, size_t validation_size, bool validate = true, bool cleanup = false, bool reset = true);

/// @brief Executes the hash function and collecting performance data. 
/// @tparam T the data type on which the algorithm shall be executed
/// @param group_count The algorithm (group_count operation) that shall be executed.
/// @param data The data on which the operation shall be evaluated
/// @param data_size 
/// @param cleanup true if group_count should be delete when the benchmark is finished. 
/// @param reset true if the hash table and count table shall be cleared again
/// @return the time the execution took in nano seconds
template <typename T>
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, bool cleanup = false, bool reset = true);

//--------------------------------------------
// output functions
//--------------------------------------------




/// @brief Prints to the console how long the benchmark already runs, and gives an estimation on how long it will keep on running
/// @param runs_done how many iterations are done already (will be changed by this function!)
/// @param total_runs how many iterations there are in total 
/// @param percentage_done how many percent of the benchmark are already done (will be changed by this function!)
/// @param percentage_print at what percentage intervals the msg should be displayed
/// @param time_begin the starting time of the benchmark
void status_output(size_t &runs_done, const size_t total_runs, double &percentage_done, const double percentage_print, std::chrono::high_resolution_clock::time_point time_begin);

//--------------------------------------------
// Benchmark Execution and Testing Functions
//--------------------------------------------

/// @brief Executes a the given algorithms with the given functions a number of times. Uses generate_data_p0 to generate the data and uses different scaleing factors
/// @tparam T  the datatype with which the test shall be executed
/// @param data_size how large the table that gets agregated shall be
/// @param distinct_value_count of how many distinct values the table should be composed of
/// @param algorithms_undertest the different algorithms that shall be benchmarked
/// @param algorithms_undertest_size 
/// @param functions_to_test the different hash functions that shall be used during the benchmark
/// @param all_hash_functions_size 
/// @return 0
template <typename T> 
int test0(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float scale_boost = 1.f);
/// @brief Executes a the given algorithms with the given functions a number of times. Uses generate_data_p1 to generate the data
/// @tparam T the datatype with which the test shall be executed
/// @param data_size how large the table that gets agregated shall be
/// @param distinct_value_count of how many distinct values the table should be composed of
/// @param algorithms_undertest the different algorithms that shall be benchmarked
/// @param algorithms_undertest_size 
/// @param functions_to_test the different hash functions that shall be used during the benchmark
/// @param all_hash_functions_size 
/// @return -1 if one of the given configuration can't be used. 0 otherwise
template <typename T> 
int test1(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float scale_boost = 1.f);
template <typename T> 
int test2(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float scale_boost = 1.f);
template <typename T> 
int test3(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float unused_float = 1.f);
template <typename T> 
int test4(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float unused_float = 1.f);

/// @brief Executes a the given algorithms with the given functions a number of times. Uses generate_data_p0 to generate the data and uses different scaleing factors
/// @tparam T  the datatype with which the test shall be executed
/// @param data_size how large the table that gets agregated shall be
/// @param distinct_value_count of how many distinct values the table should be composed of
/// @param algorithms_undertest the different algorithms that shall be benchmarked
/// @param algorithms_undertest_size 
/// @param functions_to_test the different hash functions that shall be used during the benchmark
/// @param all_hash_functions_size 
/// @return 0
template <typename T> 
int test0_benchdata(T* data, size_t data_size, std::string bench_filename, size_t column, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float scale_boost = 1.f);

/// @brief Test if the given Group_Count_Algorithm with the given HashFunction works correctly
/// @tparam T type for the execution
/// @param data_size how much data should be used to test the given Group_Count_Algorithm
/// @param distinct_value_count how many distinct values should be used
/// @param algorithms_undertest Group_Count_Algorithm to be tested
/// @param hash_function_enum HashFunction that shall be used
template <typename T> 
void alg_testing(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm algorithms_undertest, HashFunction hash_function_enum = HashFunction::MULITPLY_SHIFT);


template <typename T> 
bool all_algorithm_test();
//--------------------------------------------
// MAIN!
//--------------------------------------------

//meta benchmark info!
using ps_type = uint32_t; 
size_t repeats_same_data = 3;
size_t repeats_different_data = 3;

int main(int argc, char** argv){
    // Chained<uint32_t>->chained_hash_function = get_hash_function(HashFunction::NOISE);
    // Chained<uint32_t>->chained_HSIZE = 2048;
    fill_tab_table();


    // all_algorithm_test<ps_type>();
    // return 0;

    size_t distinct_value_count = 1024;
    size_t all_data_sizes = 1 * 1024 * 1024;// 1024*1024*1024;

    float scale_boost = 1.0f;

    Group_Count_Algorithm algorithms_undertest [] = {
        Group_Count_Algorithm::SCALAR_GROUP_COUNT_SOA 
        , Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS
        , Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS_V2

    //     , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V1
    //     , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V1 
        
    //     // // , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V2  // uninteresting 
    //     // // , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V3  // uninteresting

    //     , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V1 
    //     , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_CONFLICT_V1 
        
    //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V1 
    //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V1
    //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V3

    //     , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V2 
    //     , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V2 
    //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V4

    // //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2 // uninteresting
        
    // //     // , Group_Count_Algorithm::CHAINED 
    // //     // , Group_Count_Algorithm::CHAINED2 

    // //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V2 // not (yet) implemented 
    // //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V3 // not (yet) implemented
    // //     // , Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_CONFLICT_V2 // not (yet) implemented
    };
    
    // size_t (*all_hash_functions[])(ps_type, size_t) = {&hashx, &id_mod, &murmur, &tab};
    HashFunction functions_to_test[] = {
        // // HashFunction::MULTIPLY_PRIME,
        // HashFunction::MULITPLY_SHIFT, 
        // HashFunction::MULTIPLY_ADD_SHIFT, 
        // HashFunction::MODULO, 
        // // HashFunction::MURMUR, 
        // // HashFunction::TABULATION,
        // HashFunction::SIP_HASH,        
        HashFunction::NOISE
    };
    
    size_t number_algorithms_undertest = sizeof(algorithms_undertest) / sizeof(algorithms_undertest[0]);
    size_t number_hash_functions = sizeof(functions_to_test) / sizeof(functions_to_test[0]);

    //* //data generation Benchmarks
    // test0<ps_type>(all_data_sizes, distinct_value_count, algorithms_undertest, number_algorithms_undertest, functions_to_test, number_hash_functions, scale_boost);
    // test1<ps_type>(all_data_sizes, distinct_value_count, algorithms_undertest, number_algorithms_undertest, functions_to_test, number_hash_functions, scale_boost);
    test4<ps_type>(all_data_sizes, distinct_value_count, algorithms_undertest, number_algorithms_undertest, functions_to_test, number_hash_functions, scale_boost);
    
    //*/

    // alg_testing<ps_type>(128, 64, Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2, &id_mod);
    
    /* //Benchmarks with table data.
    std::string benchmark = "orders";
    size_t target = 4;

    std::stringstream file_name_builder;
    file_name_builder << benchmark << ".tbl";     
    std::string file_name = file_name_builder.str();

    Table<ps_type> tab(file_name);
    ps_type * column = tab.get_column(target);
    size_t row_count = tab.get_row_count();
    // ps_type * data = (ps_type*) aligned_alloc(64, row_count * sizeof(ps_type));

    // for(size_t i = 0; i < row_count; i++){
    //     data[i] = column[i];
    // }

    //count the distinct values:
    distinct_value_count = tab.get_distinct_values(target);
    std::cout << "DATA INFO:\t" << benchmark << " has " << row_count << " values with " << distinct_value_count << " distinct once\n"; 
    test0_benchdata<ps_type>(column, row_count, benchmark, target, distinct_value_count, algorithms_undertest, number_algorithms_undertest, functions_to_test, number_hash_functions);
    // free(data);
    //*/
}

//--------------------------------------------
// Benchmark Execution and Testing Functions
//--------------------------------------------

template <typename T> 
int test0(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float scale_boost){
    if(scale_boost < 1){
        return -1;
    }
    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();

    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();

    //Test Parameter Declaration!
    std::stringstream file_name_builder;
    file_name_builder << "benchmark_test0_" << data_size << "_" << distinct_value_count << ".csv";     
    std::string file_name = file_name_builder.str();
    create_result_file(file_name);


    size_t collision_count[] = {1,                    8,                      16,                      16,                      distinct_value_count/16, 0};
    size_t collision_size[] =  {distinct_value_count, distinct_value_count/8, distinct_value_count/32, distinct_value_count/64, 16,                      0};
    size_t configuration_count = sizeof(collision_count)/sizeof(collision_count[0]);

    float all_scales[] = {1.0f ,1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f}; //{1.0f ,1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f};
    size_t all_scales_size = sizeof(all_scales)/sizeof(all_scales[0]);

    const size_t elements = (512 / 8) / sizeof(T);


    size_t total_configs = configuration_count * all_hash_functions_size * repeats_different_data * all_scales_size * algorithms_undertest_size;
    size_t total_runs = repeats_same_data * total_configs; 
    
    double percentage_print = 1;
    double percentage_done = -percentage_print;
    size_t runs_done = 0;

    std::cout << "test0 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " times resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:\n";


    //run variables
    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T)); // alternative
    Group_count<T> *alg = nullptr;

    for(size_t conf = 0; conf < configuration_count; conf++){// iterate over all configurations of collisions and clusters
        for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
            size_t seed = noise(noise_id++, 0);
        
            for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data    
                HashFunction hash_function_enum = functions_to_test[hash_function_id];
                hash_fptr<T> function = get_hash_function<T>(hash_function_enum);
        
                generate_data_p0<T>( // the seed is for rdd the run id
                    data, data_size, distinct_value_count, function, 
                    collision_count[conf], collision_size[conf], seed
                );

                for(size_t ass = 0; ass < all_scales_size && data != nullptr; ass++){
                    float scale = all_scales[ass] * scale_boost;

                    size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);
                    HSIZE = (HSIZE + elements - 1);
                    HSIZE /= elements;
                    HSIZE *= elements;

                    for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                        Group_Count_Algorithm test = algorithms_undertest[aus];    
                        std::string alg_identification = "";
                        size_t internal_HSIZE;

                        getGroupCount(alg, test, HSIZE, function);
                        internal_HSIZE = alg->get_HSIZE();
                        alg_identification = alg->identify();
                        
                        for(size_t rsd = 0; rsd < repeats_same_data; rsd++){ // could be seen as a run id
                            alg->clear();
                            size_t time = 0;
                            
                            time = run_test<T>(alg, data, data_size, false);
                            
                            write_to_file(//TODO!!!
                                file_name, alg_identification, time, data_size, 
                                sizeof(T), distinct_value_count, scale, internal_HSIZE, 
                                hash_function_enum, seed, rsd, //hashfunction id, datagen seed, run id
                                collision_count[conf], collision_size[conf], 0, 0, // collision details
                                conf
                            );

                            //print some information about the progress
                            status_output(runs_done, total_runs, percentage_done, percentage_print, time_begin);
                        }
                    }
                }
            }
        }
    }

    if(data != nullptr){
        free(data);
        data = nullptr;
    }

    if(alg != nullptr){
        delete alg;
        alg = nullptr;
    }


    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);

    // std::cout << "\t100%\n"
    std::cout << "\n\nIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";
    return 0;
}


template <typename T> 
int test1(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float unused_float){
    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();
    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();

    std::stringstream file_name_builder;
    file_name_builder << "benchmark_test1_" << data_size << "_" << distinct_value_count << ".csv";     
    std::string file_name = file_name_builder.str();
    create_result_file(file_name);

    const size_t elements = (512 / 8) / sizeof(T);
    float scale = 1.5f;
    size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);
    HSIZE = (HSIZE + elements - 1);
    HSIZE /= elements;
    HSIZE *= elements;

    size_t collision_count[] =  {1,                    8,                      16,                      distinct_value_count/16, 0,                    0};
    size_t collision_size[] =   {distinct_value_count, distinct_value_count/8, distinct_value_count/64, 16,                      0,                    0};
    size_t cluster_count[] =    {0,                    0,                      16,                      0,                       1,                    128};
    size_t cluster_size[] =     {0,                    0,                      distinct_value_count/16, 0,                       distinct_value_count, distinct_value_count/128};
    size_t configuration_count = sizeof(collision_count) / sizeof(collision_count[0]);

    //verify and print configurations:
    bool config_problem = false;
    for(size_t i = 0; i < configuration_count; i++){
        size_t calc_distinct_value_count = p1_parameter_gen_distinct(collision_count[i], collision_size[i], cluster_count[i], cluster_size[i]);
        size_t calc_HSIZE_value = p1_parameter_gen_hsize(collision_count[i], collision_size[i], cluster_count[i], cluster_size[i]);
        bool DISTINCT_VALUE_COUNT_MATCH = calc_distinct_value_count == distinct_value_count;
        bool NEEDED_HSIZE_MATCH = calc_HSIZE_value <= HSIZE;
        if(!(DISTINCT_VALUE_COUNT_MATCH && NEEDED_HSIZE_MATCH)){
            std::cout << data_size << "\tProblem with Configuration " << i << std::endl;
            config_problem = true;     
        }
    }
    if(config_problem){
        return -1;
    }
    
    size_t total_configs = configuration_count * all_hash_functions_size  * algorithms_undertest_size;
    size_t total_runs = repeats_same_data * total_configs; 

    double percentage_print = 1;
    double percentage_done = -percentage_print;
    size_t runs_done = 0;

    std::cout << "test1 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " times resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:\n";


    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T));
    Group_count<T> *alg = nullptr;

    for(size_t conf = 0; conf < configuration_count && data != nullptr; conf++){// iterate over all configurations of collisions and clusters
        
        for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data        
            HashFunction hash_function_enum = functions_to_test[hash_function_id];
            hash_fptr<T> function = get_hash_function<T>(hash_function_enum);
        
            for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
                size_t seed = noise(noise_id++, 0);

                for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                    Group_Count_Algorithm test = algorithms_undertest[aus];    
                    
                    std::string alg_identification = "";
                    size_t internal_HSIZE;
                    size_t distinct_vals_generated;
                    size_t data_gen_try = 0;
                    do{
                        distinct_vals_generated = generate_data_p1<T>( // the seed is for rdd the run id
                            data, data_size, distinct_value_count, HSIZE, function,
                            collision_count[conf], collision_size[conf], cluster_count[conf], cluster_size[conf],
                            seed + data_gen_try, test == Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V1
                        );
                        data_gen_try++;
                    }while(distinct_vals_generated == 0 && data_gen_try < 5);
                    if(distinct_vals_generated == 0){
                        throw std::runtime_error("generate_data_p1 run into a problem with its data generation");
                    }

                    getGroupCount(alg, test, HSIZE, function);
                    internal_HSIZE = alg->get_HSIZE();
                    alg_identification = alg->identify();
                    
                    for(size_t rsd = 0; rsd < repeats_same_data; rsd++){ // could be seen as a run id
                        alg->clear();
                        size_t time = 0;

                        time = run_test<T>(alg, data, data_size, false);
                        
                        write_to_file(
                            file_name, alg_identification, time, data_size,  
                            sizeof(T), distinct_value_count, scale, internal_HSIZE, 
                            hash_function_enum, seed, rsd,    // run
                            collision_count[conf], collision_size[conf], cluster_count[conf], cluster_size[conf],
                            conf
                        );

                        status_output(runs_done, total_runs, percentage_done, percentage_print, time_begin);
                    }
                }
            }
        }
    }

    if(data != nullptr){
        free(data);
        data = nullptr;
    }

    if(alg != nullptr){
        delete alg;
        alg = nullptr;
    }


    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";

    return 0;
}

template <typename T> 
int test2(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float unused_float){
    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();

    /// Benchmark info and variable declaration    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();

    std::stringstream file_name_builder;
    file_name_builder << "benchmark_test2_" << data_size << "_" << distinct_value_count << ".csv";     
    std::string file_name = file_name_builder.str();
    create_result_file(file_name);

    const size_t elements = (512 / 8) / sizeof(T);
    float scale = 1.0f;
    size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);


    size_t collision_size[] = {48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}; // = {4080, 680, 240, 48, 24, 20, 17, 16, 15, 12, 8, 6, 4, 3, 2, 1};
    size_t configuration_count = sizeof(collision_size) / sizeof(collision_size[0]);

    size_t collision_count[configuration_count];

    for(size_t i = 0; i < configuration_count; i++){
        collision_count[i] = distinct_value_count/collision_size[i];
    }

    
    size_t total_configs = configuration_count * all_hash_functions_size * repeats_different_data * algorithms_undertest_size;
    size_t total_runs = repeats_same_data * total_configs; 



    /// run information and status output
    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T));
    Group_count<T> *alg = nullptr;

    double percentage_print = 1;
    double percentage_done = -percentage_print;
    size_t runs_done = 0;

    std::cout << "test2 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " times resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:\n";

    for(size_t conf = 0; conf < configuration_count && data != nullptr; conf++){// iterate over all configurations of collisions and clusters
        size_t current_collision_count = collision_count[conf];
        size_t current_collision_size = collision_size[conf];
        size_t current_cluster_size = distinct_value_count - current_collision_count * current_collision_size;
        size_t current_cluster_count = current_cluster_size != 0;


        for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
            size_t seed = noise(noise_id++, 0);

            for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data        
                HashFunction hash_function_enum = functions_to_test[hash_function_id];
                hash_fptr<T> function = get_hash_function<T>(hash_function_enum);
        
                size_t distinct_vals_generated = generate_data_p0<T>( // the seed is for rdd the run id
                    data, data_size, distinct_value_count, function,
                    current_collision_count, current_collision_size, seed
                );

                if(distinct_vals_generated == 0){
                    throw std::runtime_error("generate_data_p0 run into a problem with its data generation");
                }

                for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                    Group_Count_Algorithm test = algorithms_undertest[aus];    
                    getGroupCount(alg, test, HSIZE, function);
                    
                    size_t internal_HSIZE = alg->get_HSIZE();
                    std::string alg_identification = alg->identify();
                    
                    for(size_t rsd = 0; rsd < repeats_same_data; rsd++){ // could be seen as a run id
                        size_t time = 0;
                        alg->clear();

                        time = run_test<T>(alg, data, data_size, false);
                        write_to_file(
                            file_name, alg_identification, time, data_size,  
                            sizeof(T), distinct_value_count, scale, internal_HSIZE, 
                            hash_function_enum, seed, rsd,    // run
                            current_collision_count, current_collision_size, current_cluster_count, current_cluster_size,
                            conf
                        );

                        status_output(runs_done, total_runs, percentage_done, percentage_print, time_begin);
                    }
                }
            }
        }
    }

    if(data != nullptr){
        free(data);
        data = nullptr;
    }

    if(alg != nullptr){
        delete alg;
        alg = nullptr;
    }

    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";

    return 0;
}

//todo rework
template <typename T> 
int test3(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float unused_float){
    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();

    /// Benchmark info and variable declaration    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();

    std::stringstream file_name_builder;
    file_name_builder << "benchmark_test3_" << data_size << "_" << distinct_value_count << ".csv";     
    std::string file_name = file_name_builder.str();
    create_result_file(file_name, 3);

    const size_t elements_in_vector = (512 / 8) / sizeof(T);

    float all_scales[] = {1.0f ,1.1f, 1.5f, 2.0f}; //higher is better for performance
    size_t nr_scales = sizeof(all_scales) / sizeof(all_scales[0]);

    size_t all_collision_chain_length[] = {18, 16, 14, 12, 10, 8, 6, 4, 2, 1}; // lower is better for performance
    size_t nr_collision_chain_length = sizeof(all_collision_chain_length) / sizeof(all_collision_chain_length[0]);

    size_t all_collision_chain_count[]{32, 8}; //lower is better for performance
    size_t nr_collision_chain_count = sizeof(all_collision_chain_count)/sizeof(all_collision_chain_count[0]);

    size_t total_configs = nr_scales * nr_collision_chain_count * nr_collision_chain_length * all_hash_functions_size * repeats_different_data * algorithms_undertest_size * 2;
    size_t total_runs = repeats_same_data * total_configs; 

    /// run information and status output
    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T));
    Group_count<T> *alg = nullptr;

    double percentage_print = 0.1;
    double percentage_done = -percentage_print;
    size_t runs_done = 0;

    std::cout << "test3 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " times resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:\n";
    // std::cout << nr_scales << " " << nr_collision_chain_count << " " << nr_collision_chain_length << std::endl;
    for(size_t good = 0; good <= 1; good++){
        bool best_case_layout = good == 1;
    // std::cout << "d1\n"; 
        for(size_t collision_chain_count_id = 0; collision_chain_count_id < nr_collision_chain_count; collision_chain_count_id++){
            size_t current_collision_chain_count = all_collision_chain_count[collision_chain_count_id];
    // std::cout << "\td2\n"; 
            
            for(size_t conf = 0; conf < nr_collision_chain_length && data != nullptr; conf++){// iterate over all configurations of collisions and clusters
                size_t current_collision_chain_length = all_collision_chain_length[conf];
    // std::cout << "\t\td3\n"; 

                for(size_t scale_id = 0; scale_id < nr_scales; scale_id ++){
                    float current_scale = all_scales[scale_id];
    // std::cout << "\t\t\td4\n"; 

                    size_t HSIZE = (size_t)(current_scale * distinct_value_count + 0.5f);
                    HSIZE = (HSIZE + elements_in_vector - 1);
                    HSIZE /= elements_in_vector;
                    HSIZE *= elements_in_vector;

                    for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
                        size_t seed = noise(noise_id++, 0);
    // std::cout << "\t\t\t\td5\n"; 
    
                        for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){
                            HashFunction hash_function_enum = functions_to_test[hash_function_id];
                            hash_fptr<T> function = get_hash_function<T>(hash_function_enum);

                    
                            size_t distinct_vals_generated = generate_data_v3<T>(
                                data, data_size, distinct_value_count, HSIZE, function,
                                current_collision_chain_count, current_collision_chain_length, seed, best_case_layout
                            );

                            if(distinct_vals_generated == 0){
                                throw std::runtime_error("generate_data_v3 run into a problem with its data generation");
                            }

                            for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                                Group_Count_Algorithm test = algorithms_undertest[aus];    
                                getGroupCount(alg, test, HSIZE, function);

                                
                                size_t internal_HSIZE = alg->get_HSIZE();
                                std::string alg_identification = alg->identify();
                                
                                for(size_t rsd = 0; rsd < repeats_same_data; rsd++){
                                    size_t time = 0;
                                    alg->clear();

                                    time = run_test<T>(alg, data, data_size, false);
                                    write_to_file(
                                        file_name, alg_identification, time, data_size,  
                                        sizeof(T), distinct_value_count, current_scale, internal_HSIZE, 
                                        hash_function_enum, seed, rsd,    // run
                                        current_collision_chain_count, current_collision_chain_length, best_case_layout, good,
                                        conf
                                    );

                                    status_output(runs_done, total_runs, percentage_done, percentage_print, time_begin);
                                }
                            }
                        }
                    }
                }
            }        
        }
    }

    if(data != nullptr){
        free(data);
        data = nullptr;
    }

    if(alg != nullptr){
        delete alg;
        alg = nullptr;
    }

    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";

    return 0;
}

//todo rework
template <typename T> 
int test4(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float unused_float){
    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();

    /// Benchmark info and variable declaration    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();

    std::stringstream file_name_builder;
    file_name_builder << "benchmark_test4_" << data_size << "_" << distinct_value_count << ".csv";     
    std::string file_name = file_name_builder.str();
    create_result_file(file_name, 4);

    const size_t elements_in_vector = (512 / 8) / sizeof(T);

    // float all_scales[] = {1.0f ,1.1f, 1.5f, 2.0f}; //higher is better for performance
    // size_t nr_scales = sizeof(all_scales) / sizeof(all_scales[0]);
    const float current_scale = 1.5f;
    std::cout << "illegal?\n";
    size_t HSIZE = (size_t)(current_scale * distinct_value_count + 0.5f);
    std::cout << "illegal?\n";
    HSIZE = (HSIZE + elements_in_vector - 1);
    HSIZE /= elements_in_vector;
    HSIZE *= elements_in_vector;

    size_t all_collision_chain_length[] = {18, 16, 14, 12, 10, 8, 6, 4, 2, 1}; // lower is better for performance
    size_t nr_collision_chain_length = sizeof(all_collision_chain_length) / sizeof(all_collision_chain_length[0]);

    size_t all_collision_chain_count[]{32, 16, 8}; //lower is better for performance
    size_t nr_collision_chain_count = sizeof(all_collision_chain_count)/sizeof(all_collision_chain_count[0]);

    const size_t max_space = 20;
    const size_t space_step_size = 2;
    
    size_t total_configs = nr_collision_chain_count * nr_collision_chain_length * all_hash_functions_size * repeats_different_data * algorithms_undertest_size * (max_space/space_step_size + 1);
    size_t total_runs = repeats_same_data * total_configs; 

    /// run information and status output
    T* data = nullptr;    
    data = (T*) aligned_alloc(64, data_size * sizeof(T));
    Group_count<T> *alg = nullptr;

    double percentage_print = 0.1;
    double percentage_done = -percentage_print;
    size_t runs_done = 0;

    std::cout << "test4 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " times resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:\n";
    // std::cout << nr_scales << " " << nr_collision_chain_count << " " << nr_collision_chain_length << std::endl;
    size_t good = 0; //we want a bad layout!!!
    bool best_case_layout = good == 1;
    for(size_t collision_chain_count_id = 0; collision_chain_count_id < nr_collision_chain_count; collision_chain_count_id++){
        size_t current_collision_chain_count = all_collision_chain_count[collision_chain_count_id];
        
        for(size_t conf = 0; conf < nr_collision_chain_length && data != nullptr; conf++){// iterate over all configurations of collisions and clusters
            size_t current_collision_chain_length = all_collision_chain_length[conf];

            for(size_t space = 0; space <= max_space; space += space_step_size){
        
                for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
                    size_t seed = noise(noise_id++, 0);

                    for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){
                        HashFunction hash_function_enum = functions_to_test[hash_function_id];
                        hash_fptr<T> function = get_hash_function<T>(hash_function_enum);

                        size_t distinct_vals_generated = generate_data_v4<T>(
                            data, data_size, distinct_value_count, HSIZE, function,
                            current_collision_chain_count, current_collision_chain_length, seed, space
                        );

                        if(distinct_vals_generated == 0){
                            throw std::runtime_error("generate_data_v4 run into a problem with its data generation");
                        }

                        for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                            Group_Count_Algorithm test = algorithms_undertest[aus];    
                            getGroupCount(alg, test, HSIZE, function);

                            size_t internal_HSIZE = alg->get_HSIZE();
                            std::string alg_identification = alg->identify();
                            
                            for(size_t rsd = 0; rsd < repeats_same_data; rsd++){
                                size_t time = 0;
                                alg->clear();

                                time = run_test<T>(alg, data, data_size, false);
                                write_to_file(
                                    file_name, alg_identification, time, data_size,  
                                    sizeof(T), distinct_value_count, current_scale, internal_HSIZE, 
                                    hash_function_enum, seed, rsd,    // run
                                    current_collision_chain_count, current_collision_chain_length, good, space,
                                    conf
                                );

                                status_output(runs_done, total_runs, percentage_done, percentage_print, time_begin);
                            }
                        }
                    }
                }
            }        
        }
    }

    if(data != nullptr){
        free(data);
        data = nullptr;
    }
    if(alg != nullptr){
        delete alg;
        alg = nullptr;
    }
    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";
    return 0;
}



template <typename T> 
int test0_benchdata(T* data, size_t data_size, std::string bench_filename, size_t column, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size, float scale_boost){
    if(scale_boost < 1){
        return -1;
    }
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!


    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();

    //Test Parameter Declaration!
    std::stringstream file_name_builder;
    file_name_builder << "benchmark_test0_" << data_size << "_" << distinct_value_count << "_" << bench_filename << "_" << column << ".csv";     
    std::string file_name = file_name_builder.str();
    create_result_file(file_name);


    float all_scales[] = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f}; //{1.0f ,1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f};
    size_t all_scales_size = sizeof(all_scales)/sizeof(all_scales[0]);

    const size_t elements = (512 / 8) / sizeof(T);


    size_t total_configs = all_hash_functions_size * all_scales_size * algorithms_undertest_size;
    size_t total_runs = repeats_same_data * total_configs; 
    
    double percentage_print = 1;
    double percentage_done = -percentage_print;
    size_t runs_done = 0;

    std::cout << "test0_benchdata has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " times resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:\n";


    //run variable

    Group_count<T> *alg = nullptr;

    for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data    
        HashFunction hash_function_enum = functions_to_test[hash_function_id];
        hash_fptr<T> function = get_hash_function<T>(hash_function_enum);

        for(size_t ass = 0; ass < all_scales_size && data != nullptr; ass++){
            float scale = all_scales[ass] * scale_boost;

            size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);
            HSIZE = (HSIZE + elements - 1);
            HSIZE /= elements;
            HSIZE *= elements;
            // std::cout << "Current HSIZE:\t" << HSIZE << std::endl;
            for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                Group_Count_Algorithm test = algorithms_undertest[aus];    
                std::string alg_identification = "";
                size_t internal_HSIZE;

                getGroupCount(alg, test, HSIZE, function);
                internal_HSIZE = alg->get_HSIZE();
                alg_identification = alg->identify();
                
                for(size_t rsd = 0; rsd < repeats_same_data; rsd++){ // could be seen as a run id
                    
                    alg->clear();
                    size_t time = 0;
                    
                    time = run_test<T>(alg, data, data_size, false);
                
                    write_to_file(
                        file_name, alg_identification, time, data_size, 
                        sizeof(T), distinct_value_count, scale, internal_HSIZE, 
                        hash_function_enum, 0, rsd, //hashfunction id, datagen seed, run id
                        0, 0, 0, 0, // collision details
                        column
                    );
                    

                    //print some information about the progress
                    status_output(runs_done, total_runs, percentage_done, percentage_print, time_begin);
                }
            }
        }
    }

    if(alg != nullptr){
        delete alg;
        alg = nullptr;
    }



    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);

    // std::cout << "\t100%\n"
    std::cout << "\n\nIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";
    return 0;
}

template <typename T> 
void alg_testing(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm algorithms_undertest, HashFunction hash_function_enum){

    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();

    size_t seed = noise(noise_id++, 0);
    size_t HSIZE = distinct_value_count * 1.5;
    //run variables
    T* data = (T*) aligned_alloc(64, data_size * sizeof(T)); // alternative
    Group_count<T> *alg = nullptr;

    hash_fptr<T> hash_function = get_hash_function<T>(hash_function_enum);
    
    getGroupCount(alg, algorithms_undertest, HSIZE, hash_function);
    Scalar_gc_SoA<T> *val = new Scalar_gc_SoA<T>(HSIZE, hash_function);

    generate_data_p0<T>( // the seed is for rdd the run id
        data, data_size, distinct_value_count, hash_function, 
        2, (distinct_value_count/16)+2, seed
    );
    alg->create_hash_table(data, data_size);
    std::cout << alg->identify() << std::endl;
    val->create_hash_table(data, data_size);

    validation(alg, val, HSIZE);
}

//--------------------------------------------
// Time and Run functions for one Benchmark
//--------------------------------------------


template <typename T>
size_t run_test(Group_count<T>*& group_count, T*& data, size_t data_size, Scalar_gc_SoA<T>* validation_baseline, size_t validation_size, bool validate, bool cleanup, bool reset){
    uint64_t duration = run_test(group_count, data, data_size, false, false); 

    if(validate && validation_baseline != nullptr){
        validation<ps_type>(group_count, validation_baseline, validation_size);
    }

    if(cleanup){
        delete group_count;
        group_count = nullptr;
    }

    return duration;
}

template <typename T>
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, bool cleanup, bool reset){
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    
    time_begin = time_now();
    group_count->create_hash_table(data, data_size);;
    time_end = time_now();
    
    size_t duration = duration_time(time_begin, time_end);
    
    if(!cleanup && reset){
        group_count->clear();
    }
    
    if(cleanup){
        delete group_count;
        group_count = nullptr;
    }
    return duration;
}

//--------------------------------------------
//validation function
//--------------------------------------------

template <typename T> 
bool validation(Group_count<T>* grouping, Scalar_gc_SoA<T> *validation_baseline, size_t validation_size){
    std::cout << "Start Validation";
    size_t nr_of_errors = 0;
    for(size_t i = 0; i < validation_size; i++){
        T value = validation_baseline->getval(i);
        if(value != 0){
            T expected_count = validation_baseline->get(value);
            T result_count = grouping->get(value);
            if(expected_count != result_count){
                if(nr_of_errors == 0){
                    std::cout << std::endl;
                }

                nr_of_errors++;
                std::cout << "\tERROR Count\t" << value << " has a count of\t" 
                    << result_count << " but expected to have\t" << expected_count << std::endl;
            }
        }
    }
    if(nr_of_errors == 1){
        std::cout << "\tFound one Error\n";
    }else if(nr_of_errors > 1){
        std::cout << "\tFound " << nr_of_errors << " Errors\n";
    }else{
        std::cout << "\t---\t";
    }
    std::cout << "End of Validation\n";
    return nr_of_errors != 0;
}

//--------------------------------------------
// output functions
//--------------------------------------------

void status_output(size_t &runs_done, const size_t total_runs, double &percentage_done, const double percentage_print, std::chrono::high_resolution_clock::time_point time_begin){
    runs_done++;
    std::chrono::high_resolution_clock::time_point time_end;
    double current_percentage = (runs_done * 100.) / total_runs;
    if(current_percentage > percentage_done + percentage_print){
        while(current_percentage > percentage_done + percentage_print){
            percentage_done += percentage_print;
        }

        time_end = time_now();
        size_t meta_time = duration_time(time_begin, time_end);
        size_t meta_time_sec = (size_t)(meta_time / 1000000000.0 + 0.5);
        double work_done = (runs_done * 1. / total_runs);

        size_t meta_time_min = (size_t)(meta_time_sec / 60.0 + 0.5);
        size_t meta_time_left = (size_t)(meta_time_sec / work_done * (1 - work_done));
        if(meta_time_sec < 60){
            std::cout << "\t" <<((int32_t)(1000 * percentage_done))/1000. << "%\tit took ~" << meta_time_sec << " sec. Approx time left:\t" ;
        }else{
            std::cout << "\t" << ((int32_t)(1000 * percentage_done))/1000. << "%\tit took ~" << meta_time_min << " min. Approx time left:\t" ;
        }
        if(meta_time_left < 60){
            std::cout << meta_time_left << " sec" << std::endl;
        }else{
            meta_time_left = (size_t)(meta_time_left / 60.0 + 0.5);
            std::cout << meta_time_left << " min" << std::endl;
        }
    }
}

//--------------------------------------------
// other functions
//--------------------------------------------

template <typename T> 
bool all_algorithm_test(){
    srand(std::time(nullptr));

    std::cout << "\nAlgorithm Test\n";

    Group_Count_Algorithm all_algorithms_undertest [] = {
        Group_Count_Algorithm::SCALAR_GROUP_COUNT_SOA, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V2, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V3, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V2, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2, 
        Group_Count_Algorithm::CHAINED, 
        Group_Count_Algorithm::CHAINED2,
        
        Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS, 
        Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS_V2, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V2, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V3, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V1,
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V2,
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_CONFLICT_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V3,
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V4,
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_CONFLICT_V2
    };

    const size_t algorithm_count = sizeof(all_algorithms_undertest) / sizeof(all_algorithms_undertest[0]);
    const size_t distinct = 128;
    const size_t data_size = distinct * 2 + 1;

    bool error = false;
    hash_fptr<T> function = get_hash_function<T>(HashFunction::MODULO);

    Group_count<T> *base_line = nullptr;
    getGroupCount(base_line, Group_Count_Algorithm::SCALAR_GROUP_COUNT_SOA, distinct * 2.1, function);

    T* data = (T*) aligned_alloc(64, data_size * sizeof(T));


    size_t d = generate_data_p0<T>(
        data, 
        data_size, 
        distinct, 
        function,
        2,  // 2 collisions groups
        24,  //  of length 24
        std::rand()
    );
    
    base_line->create_hash_table(data, data_size);

    T * all_values = (T*) aligned_alloc(64, distinct * sizeof(T));

    size_t nr_placed = 0;
    for(size_t i = 0; i < data_size && nr_placed < d; i++){
        bool placed = false;

        for(size_t e = 0; e < nr_placed && !placed; e++){
            placed = all_values[e] == data[i];
        }
        
        if(!placed){
            all_values[nr_placed] = data[i];
            nr_placed++;
        }
    }

    Group_count<T> *alg = nullptr;
    for(size_t alg_id = 0; alg_id < algorithm_count; alg_id++){
        Group_Count_Algorithm check = all_algorithms_undertest[alg_id];
        bool executable = true;    
        try{
            getGroupCount(alg, check, distinct, function);
        }catch(std::exception& e){
            executable = false;
            std::cout << "|\tUNKOWN ALGORITHM. Enum entry: " << check << std::endl;
        }
        if(executable){
            bool promted = false;
            alg->create_hash_table(data, data_size);
            std::cout << "|\t" << alg->identify();
            for(size_t i = 0; i < nr_placed; i++){
                
                size_t value = all_values[i];
                size_t a_count = alg->get(value);
                size_t b_count = base_line->get(value);

                if(a_count != b_count){
                    if(!promted){
                        std::cout << "\tERROR\n";
                        promted = true;
                    }
                    std::cout << "|\t|\t" << value << "\tis: " << a_count << "\tshould be: " << b_count<<std::endl;
                    error = true;
                }
            }
            if(promted && error){
                std::cout << "|\t+------------------\n";
            }
            if(!promted){
                std::cout << " \tOK\n";
            }
        }
    }
    std::cout << "+-------------\n\n";
    return error;
}