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
#include "../operator/physical/group_count/avx512_group_count_soa_collision_v1.hpp"

#include "datagen.hpp"


// soaov p1 data generation
// p0 data generation
// p2 data generaiton
// more hashing functions
// collisions with 16 x 32bit elements or 8 x 64bit

enum Algorithm{
    SCALAR_GROUP_COUNT, 
    AVX512_GROUP_COUNT_SOA_V1, 
    AVX512_GROUP_COUNT_SOA_V2, 
    AVX512_GROUP_COUNT_SOA_V3, 
    AVX512_GROUP_COUNT_SOAOV_V1, 
    AVX512_GROUP_COUNT_SOA_COLLISION_V1
};


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
        case Algorithm::AVX512_GROUP_COUNT_SOA_COLLISION_V1:
            run = new AVX512_group_count_SoA_collision_v1<T>(HSIZE, function);
            break;
        default:
            throw std::runtime_error("One of the Algorithms isn't supported yet!");
    }
}

//---------------------------------------
// hash function
//---------------------------------------
size_t table[8][256];
void fill_tab_table();

template <typename T>
size_t force_collision(T, size_t HSIZE);

// simple multiplicative hashing function
size_t hashx(uint32_t key, size_t HSIZE);

template <typename T>
size_t id_mod(T key, size_t HSIZE);


//NEEDS INITIALISATION FOR TABLE!
template<typename T>
size_t tab(T key, size_t HSIZE);

//based on a seven dimensional ... murmur3_64_finalizer
template<typename T>
size_t murmur(T key, size_t HSIZE);

template<typename T>
size_t multiply_shift(T key, size_t HSIZE);

template<typename T>
size_t multiply_shift_add(T key, size_t HSIZE);

//---------------------------------------
//validation functions
//---------------------------------------

template <typename T> 
size_t createCountValidationTable(T** res_table, T** res_count, T* data, size_t data_size, size_t HSIZE);

template <typename T> 
bool validation(Group_count<T>* grouping, T* table_value, T* table_count, size_t data_size);
template <typename T> 
bool validation(Group_count<T>* grouping, Scalar_group_count<T>* validation_baseline, size_t validation_size);

//---------------------------------------
// benchmark functions
//---------------------------------------

template <typename T>
size_t run_test(Group_count<T>* group_count, T* data, size_t data_size, T* validation_value, T* validation_count, size_t validation_size, bool cleanup = true);
template <typename T>
size_t run_test(Group_count<T>* group_count, T* data, size_t data_size, Scalar_group_count<T>* validation_baseline, size_t validation_size, bool validate = true, bool cleanup = true);
template <typename T>
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, bool cleanup = true); // without validation

std::chrono::high_resolution_clock::time_point time_now();

uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end);

//---------------------------------------
// output functions
//---------------------------------------

void create_result_file(std::string filename);

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
    size_t hash_function_id, // hash function index size_t 
    size_t seed, // Datageneration seed COULD BE REPLACED BY ANNOTHER ID BUT!  size_t 
    size_t rsd, // run id (same config with same runs) size_t 
    size_t config_collision_count,
    size_t config_collition_size,
    size_t conig_cluster_count,
    size_t config_cluster_size,
    size_t config_id
);


//---------------------------------------
// MAIN!
//---------------------------------------
template <typename T> 
int test0(size_t data_size, size_t distinct_value_count = 2048);
template <typename T> 
int test1(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, size_t (**all_hash_functions)(T, size_t), size_t all_hash_functions_size);


//meta benchmark info!
using ps_type = uint32_t; 
size_t repeats_same_data = 5;
size_t repeats_different_data = 3;

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
    };
    
    size_t (*all_hash_functions[])(ps_type, size_t) = {&hashx, &id_mod};
    
    size_t number_algorithms_undertest = sizeof(algorithms_undertest) / sizeof(algorithms_undertest[0]);
    size_t number_hash_functions = sizeof(all_hash_functions) / sizeof(all_hash_functions[0]);

    test1<ps_type>(all_data_sizes, distinct_value_count, algorithms_undertest, number_algorithms_undertest, all_hash_functions, number_hash_functions);
}

//---------------------------------------
// benchmark functions
//---------------------------------------

//should use p0 for data gen since we just place the data into memory. important note is that cluster is not a meaningfull metric for that placement since p0 generates for distinct data.
template <typename T> 
int test0(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, size_t (**all_hash_functions)(T, size_t), size_t all_hash_functions_size){
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


    size_t collision_count[] = {0, 0, 1, 8, 8, 128, 128};
    size_t collision_size[] = {0, 0, distinct_value_count, distinct_value_count/8, distinct_value_count/16, distinct_value_count/128, 8};
    size_t configuration_count = sizeof(collision_count)/sizeof(collision_count[0]);

    float all_scales[] = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f};
    size_t all_scales_size = sizeof(all_scales)/sizeof(all_scales[0]);

    const size_t elements = (512 / 8) / sizeof(T);


    size_t total_configs = configuration_count * all_hash_functions_size * repeats_different_data * all_scales_size * algorithms_undertest_size;
    size_t total_runs = repeats_same_data * total_configs; 
    
    double percentage_print = 5;
    double percentage_done = 0;
    size_t runs_done = 0;

    std::cout << "test0 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:";


    //run variables
    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T)); // alternative
    Group_count<T> *alg = nullptr;

    for(size_t conf = 0; conf < configuration_count; conf++){// iterate over all configurations of collisions and clusters
        for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data    
            size_t (*function)(T, size_t) = all_hash_functions[hash_function_id];
        
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
                                hash_function_id, seed, rsd, //hashfunction id, datagen seed, run id
                                collision_count[conf], collision_size[conf], 0, 0, // collision details
                                conf
                            );

                            runs_done++;
                            if((runs_done * 100) / total_runs > percentage_done + percentage_print){
                                percentage_done += percentage_print;
                                std::cout << "\t" << percentage_done << "%";
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

    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);

    // std::cout << "\t100%\n"
    std::cout << "\n\nIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";
    return 0;
}


//test different "perfect" layouts against each other. "SHORT TEST"
template <typename T> 
int test1(size_t data_size, size_t distinct_value_count, Algorithm *algorithms_undertest, size_t algorithms_undertest_size, size_t (**all_hash_functions)(T, size_t), size_t all_hash_functions_size){
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


    size_t collision_count[] = {0, 0, 1, 8, 8, 128};
    size_t cluster_count[] = {1, 128, 0, 0, 8, 0};
    size_t collision_size[] = {0, 0, distinct_value_count, distinct_value_count/8, distinct_value_count/16, distinct_value_count/128};
    size_t cluster_size[] = {distinct_value_count, distinct_value_count/128, 0, 0, distinct_value_count/8, 0};
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


    double percentage_print = 5;
    double percentage_done = 0;
    size_t runs_done = 0;

    std::cout << "test1 has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " resulting in " << total_runs << " total runs\n";
    std::cout << "percentage done:";


    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T));
    Group_count<T> *alg = nullptr;

    for(size_t conf = 0; conf < configuration_count && data != nullptr; conf++){// iterate over all configurations of collisions and clusters
        
        for(size_t hash_function_id = 0; hash_function_id < all_hash_functions_size; hash_function_id++){ // sets the hashfunction. different functions lead to different data        
            size_t (*function)(T, size_t) = all_hash_functions[hash_function_id];
        
            for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
                size_t seed = noise(noise_id++, 0);

                for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                    Algorithm test = algorithms_undertest[aus];    
                    
                    std::string alg_identification = "";
                    size_t internal_HSIZE;

                    generate_data_p1<T>( // the seed is for rdd the run id
                        data, data_size, distinct_value_count, HSIZE, function,
                        collision_count[conf], collision_size[conf], cluster_count[conf], cluster_size[conf],
                        seed, test == Algorithm::AVX512_GROUP_COUNT_SOAOV_V1
                    );

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
                            hash_function_id, seed, rsd,    // run
                            collision_count[conf], collision_size[conf], cluster_count[conf], cluster_size[conf],
                            conf
                        );

                        runs_done++;
                        if((runs_done * 100) / total_runs > percentage_done + percentage_print){
                            percentage_done += percentage_print;
                            std::cout << "\t" << percentage_done << "%";
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
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min for " << data_size << "\n\n";

    return 0;
}





/// @brief Executes the hash function and collecting performance Data. 
/// @tparam T 
/// @param group_count The group_count operation that shall be executed.
/// @param data The data on which the operation shall be evaluated
/// @param data_size 
/// @param validation_value //Information for validation Key column
/// @param validation_count //Information for validation count column
/// @param validation_size 
/// @param cleanup true if group_count should be delete when the benchmark is finished. 
template <typename T>
size_t run_test(Group_count<T>* group_count, T* data, size_t data_size, T* validation_value, T* validation_count, size_t validation_size, bool cleanup){
    // prepare for the testing of the function    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    uint64_t duration;
    double duration_s;
    double data_amount = (data_size * sizeof(ps_type) * 8)/1000000000.0; // Gbit
    double data_count = data_size / 1000000000.0; // Million Values

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    std::cout << group_count->identify() << std::endl;
    // run the test and time it
    time_begin = time_now();
    group_count->create_hash_table(data, data_size);;
    time_end = time_now();
    
    std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    // out put result
    duration = duration_time(time_begin, time_end);
    duration_s = duration / 1000000000.0;

    std::cout << "\tTime:\t" << duration << " ns\tOR\t" << duration_s << " s\n";
    std::cout << "\tData:\t" << data_amount << " Gbit\tOR\t" << data_size << " Values\n"; 
    std::cout << "\ttput:\t" << (data_amount)/(duration_s) << " Gbit/s\tor\tperf:\t" << (data_count)/(duration_s) << " Gval/s\n";
    std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    // validate run
    bool errors = validation<ps_type>(group_count, validation_value, validation_count, validation_size);

    if(cleanup){
        delete group_count;
        group_count = nullptr;
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";

    if(errors){
        return 0;
    }
    return duration;
}


/// @brief Executes the hash function and collecting performance Data. 
/// @tparam T 
/// @param group_count The group_count operation that shall be executed.
/// @param data The data on which the operation shall be evaluated
/// @param data_size 
/// @param validation_baseline annother run of the scalar algorithm to compare the results.
/// @param validation_size the hash table size of the validation_baseline
/// @param cleanup true if group_count should be delete when the benchmark is finished. 
template <typename T>
size_t run_test(Group_count<T>* group_count, T* data, size_t data_size, Scalar_group_count<T>* validation_baseline, size_t validation_size, bool validate, bool cleanup){
    // prepare for the testing of the function    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    uint64_t duration;

    time_begin = time_now();
    group_count->create_hash_table(data, data_size);;
    time_end = time_now();
    
    duration = duration_time(time_begin, time_end);

    // validate run
    bool errors = false;
    if(validate && validation_baseline != nullptr){
        errors = validation<ps_type>(group_count, validation_baseline, validation_size);
    }

    if(cleanup){
        delete group_count;
        group_count = nullptr;
    }

    if(errors){
        throw std::runtime_error("Problem during Validation!");
    }
    return duration;
}


/// @brief Executes the hash function and collecting performance Data. 
/// @tparam T 
/// @param group_count The group_count operation that shall be executed.
/// @param data The data on which the operation shall be evaluated
/// @param data_size 
/// @param cleanup true if group_count should be delete when the benchmark is finished. 
template <typename T>
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, bool cleanup){
    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    
    time_begin = time_now();
    group_count->create_hash_table(data, data_size);;
    time_end = time_now();
    
    size_t duration = duration_time(time_begin, time_end);

    if(cleanup){
        delete group_count;
        group_count = nullptr;
    }
    return duration;
}


/*
    creates a time point
*/
std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

/*
    gives the time between begin and end in nanoseconds
*/
uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

}

/// @brief Creates a arrays with the expected result values, that can be used to evaluate
/// @tparam T 
/// @param res_table pointer to the result array containing all values
/// @param res_count pointer to the result array containing the counts
/// @param data array contaning all the values
/// @param data_size number of entries in the data array
/// @param HSIZE the size of the Hash table 
/// @return the number of slots used in the result arrays
template <typename T> 
size_t createCountValidationTable(T** res_table, T** res_count, T* data, size_t data_size, size_t HSIZE){
    *res_table = new T[HSIZE];
    *res_count = new T[HSIZE];
    size_t m_id = 0;
    
    for(size_t p = 0; p < data_size; p++){
        T value = data[p];
        bool found = false;
        for(size_t i = 0; i < m_id; i++){
            if((*res_table)[i] == value){
                (*res_count)[i]++;
                found = true;
                break;
            }
        }
        if(!found){
            (*res_table)[m_id] = value;
            (*res_count)[m_id] = 1;
            m_id++;
        }
    }
    return m_id;
}


template <typename T> 
bool validation(Group_count<T>* grouping, T* table_value, T* table_count, size_t data_size){
    std::cout << "Start Validation";
    size_t nr_of_errors = 0;
    for(size_t i = 0; i < data_size; i++){
        T value = table_value[i];
        size_t count = grouping->get(value);

        if(count != table_count[i]){
            if(nr_of_errors == 0){
                std::cout << std::endl;
            }

            nr_of_errors++;
            std::cout << "\tERROR Count\t" << value << " has a count of\t" 
                << count << " but expected to have\t" << table_count[i] << std::endl;
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

//---------------------------------------
// output functions
//---------------------------------------

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
    size_t hash_function_id, // hash function index size_t 
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
            << HSIZE << "," << hash_function_id << "," 
            << seed << "," << rsd << "," 
            << config_collision_count << "," << config_collition_size << "," 
            << config_cluster_count << "," << config_cluster_size << ","
            << config_id << "\n"; 
        myfile.close();
    } else {
        throw std::runtime_error("Could not open file to write results!");
    }
}

//---------------------------------------
// hash function help function
//---------------------------------------

void fill_tab_table(){
    size_t seed = 0xfa342;
    for(size_t i = 0; i < 8; i++){
        for(size_t e = 0; e < 256; e++){
            table[i][e] = noise(i * 256 + e, seed);
        }
    }
}

//---------------------------------------
// hash function
//---------------------------------------

/// @brief This "Hash" functions results always in a collition
/// @param key  Key gets ignored but is needed for the function template
/// @param HSIZE Hash Table size. 
/// @return value range in [0:HSIZE-1]
template <typename T>
size_t force_collision(T key, size_t HSIZE){
    return HSIZE - 1;
}

// simple multiplicative hashing function
size_t hashx(uint32_t key, size_t HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

template<typename T>
size_t id_mod(T key, size_t HSIZE) {
    size_t k = key;
    return k % HSIZE;
}

template<typename T>
size_t tab(T key, size_t HSIZE){
    size_t result = 0;
    for(size_t i = 0; i < sizeof(T); i++){
        result ^= table[i][(char)(key >> 8*i)];
    }
    return result % HSIZE;
}


// BASED ON: "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing"'s 
// murmur3_64_finalizer
template<typename T>
size_t murmur(T key, size_t HSIZE){
    size_t k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    return k % HSIZE;
}

// BASED ON: "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing"'s 
size_t multiply_shift(uint32_t key, size_t HSIZE){
    size_t k = key;
    return 0;
}