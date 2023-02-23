#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>

#include "../operator/physical/group_count/scalar_group_count.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v1.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v2.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v3.hpp"
#include "../operator/physical/group_count/avx512_group_count_soaov_v1.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_conflict_v1.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_conflict_v2.hpp"

#include "datagen.hpp"
#include "hash_function.hpp"

// TODO
// p2 data generaiton
// TVL integration!
// rethink how different collisions could be passed to the given test. ATM: hard coded in the testX functions.
// documentation
// und hsize * 16

enum Algorithm{
    SCALAR_GROUP_COUNT, 
    AVX512_GROUP_COUNT_SOA_V1, 
    AVX512_GROUP_COUNT_SOA_V2, 
    AVX512_GROUP_COUNT_SOA_V3, 
    AVX512_GROUP_COUNT_SOAOV_V1, 
    AVX512_GROUP_COUNT_SOA_CONFLICT_V1, 
    AVX512_GROUP_COUNT_SOA_CONFLICT_V2
};

/// @brief creates a new instance of Group_count.
/// @tparam T the data type for the execution
/// @param run the instance of the Group_count. If not nullptr then it gets deleted
/// @param test the algorithm that shall be created
/// @param HSIZE the hash table size for the algorithm
/// @param function the hash function that shall be used
template <typename T>
void getGroupCount(Group_count<T> *& run, Algorithm test, size_t HSIZE, size_t (*function)(T, size_t));

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
bool validation(Group_count<T>* grouping, Scalar_group_count<T>* validation_baseline, size_t validation_size);

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
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, Scalar_group_count<T>*& validation_baseline, size_t validation_size, bool validate = true, bool cleanup = false, bool reset = true);

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

/// @brief creates a time point now
/// @return current time point
std::chrono::high_resolution_clock::time_point time_now();

/// @brief calculates the time differeance between two time points
/// @param begin starting time point
/// @param end end time point
/// @return time between the given time points in nano seconds
uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end);

//--------------------------------------------
// output functions
//--------------------------------------------

/// @brief creates a new empty file with the given namen. important an already existing file might be overwritten
/// @param filename the name of the file that shall be created
void create_result_file(std::string filename);

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
int test0(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size);
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
int test1(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size);
/// @brief Test if the given Algorithm with the given HashFunction works correctly
/// @tparam T type for the execution
/// @param data_size how much data should be used to test the given Algorithm
/// @param distinct_value_count how many distinct values should be used
/// @param algorithms_undertest Algorithm to be tested
/// @param hash_function_enum HashFunction that shall be used
template <typename T> 
void alg_testing(size_t data_size, size_t distinct_value_count, Algorithm algorithms_undertest, HashFunction hash_function_enum = HashFunction::MULITPLY_SHIFT);


//--------------------------------------------
// MAIN!
//--------------------------------------------

//meta benchmark info!
using ps_type = uint32_t; 
size_t repeats_same_data = 5;
size_t repeats_different_data = 1;

int main(int argc, char** argv){
    fill_tab_table();

    size_t distinct_value_count = 2048;
    size_t all_data_sizes = 32 * 1024 * 1024;// 1024*1024*1024;
    
    Algorithm algorithms_undertest [] = {
        Algorithm::SCALAR_GROUP_COUNT
        , Algorithm::AVX512_GROUP_COUNT_SOA_V1
        , Algorithm::AVX512_GROUP_COUNT_SOA_V2
        , Algorithm::AVX512_GROUP_COUNT_SOA_V3
        , Algorithm::AVX512_GROUP_COUNT_SOAOV_V1 
        , Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V1
        , Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2
    };
    
    // size_t (*all_hash_functions[])(ps_type, size_t) = {&hashx, &id_mod, &murmur, &tab};
    HashFunction functions_to_test[] = {
        HashFunction::MULITPLY_SHIFT, 
        HashFunction::MULTIPLY_ADD_SHIFT, 
        HashFunction::MODULO, 
        HashFunction::MURMUR, 
        HashFunction::TABULATION
    };
    
    size_t number_algorithms_undertest = sizeof(algorithms_undertest) / sizeof(algorithms_undertest[0]);
    size_t number_hash_functions = sizeof(functions_to_test) / sizeof(functions_to_test[0]);

    test0<ps_type>(all_data_sizes, distinct_value_count, algorithms_undertest, number_algorithms_undertest, functions_to_test, number_hash_functions);
    // alg_testing<ps_type>(128, 64, Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2, &id_mod);

}

//--------------------------------------------
// Benchmark Execution and Testing Functions
//--------------------------------------------

template <typename T> 
int test0(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size){
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


    size_t collision_count[] = { 8, 8, 128, 0};
    size_t collision_size[] = {distinct_value_count/8, distinct_value_count/16, distinct_value_count/128, 0};
    size_t configuration_count = sizeof(collision_count)/sizeof(collision_count[0]);

    float all_scales[] = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f};
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
        for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data    
            HashFunction hash_function_enum = functions_to_test[hash_function_id];
            hash_fptr<T> function = get_hash_function<T>(hash_function_enum);
        
            for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
                size_t seed = noise(noise_id++, 0);
                generate_data_p0<T>( // the seed is for rdd the run id
                    data, data_size, distinct_value_count, function, 
                    collision_count[conf], collision_size[conf], seed
                );

                for(size_t ass = 0; ass < all_scales_size && data != nullptr; ass++){
                    float scale = all_scales[ass];

                    size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);
                    HSIZE = (HSIZE + elements - 1);
                    HSIZE /= elements;
                    HSIZE *= elements;

                    for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                        Algorithm test = algorithms_undertest[aus];    
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

    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);

    // std::cout << "\t100%\n"
    std::cout << "\n\nIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";
    return 0;
}


template <typename T> 
int test1(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, HashFunction* functions_to_test, size_t all_hash_functions_size){
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

    size_t collision_count[] = {8, 8, 128, 0, 0};
    size_t cluster_count[] = {0, 8, 0, 1, 128};
    size_t collision_size[] = {distinct_value_count/8, distinct_value_count/16, distinct_value_count/128, 0, 0};
    size_t cluster_size[] = {0, distinct_value_count/8, 0, distinct_value_count, distinct_value_count/128};
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
    
    size_t total_configs = configuration_count * all_hash_functions_size * repeats_different_data * algorithms_undertest_size;
    size_t total_runs = repeats_same_data * total_configs; 

    double percentage_print = 2.5;
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
                    Algorithm test = algorithms_undertest[aus];    
                    
                    std::string alg_identification = "";
                    size_t internal_HSIZE;
                    size_t distinct_vals_generated;
                    size_t data_gen_try = 0;
                    do{
                        distinct_vals_generated = generate_data_p1<T>( // the seed is for rdd the run id
                            data, data_size, distinct_value_count, HSIZE, function,
                            collision_count[conf], collision_size[conf], cluster_count[conf], cluster_size[conf],
                            seed + data_gen_try, test == Algorithm::AVX512_GROUP_COUNT_SOAOV_V1
                        );
                        data_gen_try++;
                    }while(distinct_vals_generated == 0 && data_gen_try < 5);
                    if(distinct_vals_generated == 0){
                        throw std::runtime_error("generate_data_p1 run into a problem with it's data generation");
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

    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";

    return 0;
}


template <typename T> 
void alg_testing(size_t data_size, size_t distinct_value_count, Algorithm algorithms_undertest, HashFunction hash_function_enum){

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
    Scalar_group_count<T> *val = new Scalar_group_count<T>(HSIZE, hash_function);

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
size_t run_test(Group_count<T>*& group_count, T*& data, size_t data_size, Scalar_group_count<T>* validation_baseline, size_t validation_size, bool validate, bool cleanup, bool reset){
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

std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}


//--------------------------------------------
//validation function
//--------------------------------------------

template <typename T> 
bool validation(Group_count<T>* grouping, Scalar_group_count<T> *validation_baseline, size_t validation_size){
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

void create_result_file(std::string filename){
    std::ofstream myfile;
    myfile.open (filename);
    if(myfile.is_open()){
        myfile << "Algorithm,time,data_size,bytes,distinct_value_count,scale,hash_table_size,hash_function_ID,seed,run_ID,collision_count,collision_length,cluster_count,cluster_length,config_ID\n";
        myfile.close();
    } else {
        throw std::runtime_error("Could not open file to write results!");
    }
}

void write_to_file( std::string filename, //string
    std::string alg_identification, //string
    // benchmark time
    uint64_t time, //size_t or uint64_t
    // config
    size_t data_size,   // size_t 
    size_t bytes,
    size_t distinct_value_count, // size_t 
    float scale, // Scaleing factor for Hash Table double/float
    size_t HSIZE, // HASH Table Size size_t 
    HashFunction hash_function_enum, // hash function index size_t 
    size_t seed, // Datageneration seed COULD BE REPLACED BY ANNOTHER ID BUT!  size_t 
    size_t rsd, // run id (same config with same runs) size_t
    size_t config_collision_count,
    size_t config_collition_size,
    size_t config_cluster_count,
    size_t config_cluster_size, 
    size_t config_id
){
    std::ofstream myfile;
    myfile.open (filename, std::ios::app);
    if(myfile.is_open()){
        // "Algorithm,time,data size,bytes,distinct value count,scale,hash table size,hash function ID,seed,run ID";
        myfile << alg_identification << "," << time << "," 
            << data_size << "," << bytes 
            << "," << distinct_value_count  << "," << scale << "," 
            << HSIZE << "," << get_hash_function_name(hash_function_enum) << "," 
            << seed << "," << rsd << "," 
            << config_collision_count << "," << config_collition_size << "," 
            << config_cluster_count << "," << config_cluster_size << ","
            << config_id << "\n"; 
        myfile.close();
    } else {
        throw std::runtime_error("Could not open file to write results!");
    }
}

void status_output(size_t &runs_done, const size_t total_runs, double &percentage_done, const double percentage_print, std::chrono::high_resolution_clock::time_point time_begin){
    runs_done++;
    std::chrono::high_resolution_clock::time_point time_end;
    uint32_t current_percentage = (runs_done * 100) / total_runs;
    if(current_percentage >= percentage_done + percentage_print){
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
            std::cout << "\t" << percentage_done << "%\tit took ~" << meta_time_sec << " sec. Approx time left:\t" ;
        }else{
            std::cout << "\t" << percentage_done << "%\tit took ~" << meta_time_min << " min. Approx time left:\t" ;
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
void getGroupCount(Group_count<T> *& run, Algorithm test, size_t HSIZE, size_t (*function)(T, size_t)){
    if(run != nullptr){
        delete run;
        run = nullptr;
    }

    switch(test){
        case Algorithm::SCALAR_GROUP_COUNT:
            run = new Scalar_group_count<T>(HSIZE, function);
            break;
        case Algorithm::AVX512_GROUP_COUNT_SOA_V1:
            run = new AVX512_group_count_SoA_v1<T>(HSIZE, function);
            break;
        case Algorithm::AVX512_GROUP_COUNT_SOA_V2:
            run = new AVX512_group_count_SoA_v2<T>(HSIZE, function);
            break;
        case Algorithm::AVX512_GROUP_COUNT_SOA_V3:
            run = new AVX512_group_count_SoA_v3<T>(HSIZE, function);
            break;
        case Algorithm::AVX512_GROUP_COUNT_SOAOV_V1:
            run = new AVX512_group_count_SoAoV_v1<T>(HSIZE, function);
            break;
        case Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V1:
            run = new AVX512_group_count_SoA_conflict_v1<T>(HSIZE, function);
            break;
        case Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2:
            run = new AVX512_group_count_SoA_conflict_v2<T>(HSIZE, function);
            break;
        default:
            throw std::runtime_error("One of the Algorithms isn't supported yet!");
    }
}