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

#include "datagen.hpp"



enum Algorithm{SCALAR_GROUP_COUNT, AVX512_GROUP_COUNT_SOA_V1, 
    AVX512_GROUP_COUNT_SOA_V2, AVX512_GROUP_COUNT_SOA_V3, 
    AVX512_GROUP_COUNT_SOAOV_V1};


//---------------------------------------
// hash function
//---------------------------------------
template <typename T>
size_t force_collision(T, size_t HSIZE);

// simple multiplicative hashing function
size_t hashx(uint32_t key, size_t HSIZE);

template <typename T>
size_t id_mod(T key, size_t HSIZE);

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
    size_t ahfs, // hash function index size_t 
    size_t seed, // Datageneration seed COULD BE REPLACED BY ANNOTHER ID BUT!  size_t 
    size_t rsd, // run id (same config with same runs) size_t 
    size_t config_collision_count,
    size_t config_collition_size,
    size_t conig_cluster_count,
    size_t config_cluster_size 
);




template <typename T> 
int test1(){
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    time_begin = time_now();
//Test Parameter Declaration!
    std::string file_name;

    size_t distinct_value_count = 2048;
    size_t min_HSIZE = distinct_value_count * 1.1f + 0.5f;

    size_t collision_count[] = {0, 0, 1, 8, 8, 128, 128};
    size_t cluster_count[] = {1, 128, 0, 0, 8, 0, 128};
    size_t collision_size[] = {0, 0, distinct_value_count, distinct_value_count/8, distinct_value_count/16, distinct_value_count/128, 8};
    size_t cluster_size[] = {distinct_value_count, distinct_value_count/128, 0, 0, distinct_value_count/8, 0, distinct_value_count/128};
    size_t configuration_count = 7;

    //verify and print configurations:
    for(size_t i = 0; i < configuration_count; i++){
        size_t calc_distinct_value_count = p1_parameter_gen_distinct(collision_count[i], collision_size[i], cluster_count[i], cluster_size[i]);
        size_t calc_HSIZE_value = p1_parameter_gen_hsize(collision_count[i], collision_size[i], cluster_count[i], cluster_size[i]);
        std::string* config = p1_stringify(min_HSIZE, collision_count[i], collision_size[i], cluster_count[i], cluster_size[i]);
        bool DISTINCT_VALUE_COUNT_MATCH = calc_distinct_value_count == distinct_value_count;
        bool NEEDED_HSIZE_MATCH = calc_HSIZE_value <= min_HSIZE;
        std::cout << "Configurations\n";
        if(DISTINCT_VALUE_COUNT_MATCH && NEEDED_HSIZE_MATCH){
            std::cout << "\t" << i << ":\t" << *config << std::endl;
        }else{
            std::cout << "\tERROR with " << i << "\t(soll, ist)\tdistinct_value_count (" << distinct_value_count
                << ", " << calc_distinct_value_count << ")\tmin HSIZE(" << min_HSIZE << ", " << calc_HSIZE_value << ")\n";   
        }
        delete config;
    }
    



    float all_scales[] = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f};
    size_t all_data_sizes[] = {32 * 1024 * 1024, 1024*1024*1024};
    size_t (*all_hash_functions[])(T, size_t) = {&hashx, &id_mod}; // {&hashx, &force_collision}; 

    Algorithm algorithms_undertest [] = {
        Algorithm::SCALAR_GROUP_COUNT
        , Algorithm::AVX512_GROUP_COUNT_SOA_V1
        , Algorithm::AVX512_GROUP_COUNT_SOA_V2
        , Algorithm::AVX512_GROUP_COUNT_SOA_V3
        , Algorithm::AVX512_GROUP_COUNT_SOAOV_V1
    };

    size_t repeats_same_data = 5;
    size_t repeats_different_data = 3;

    size_t all_scales_size = sizeof(all_scales)/sizeof(all_scales[0]);
    size_t all_data_sizes_size = sizeof(all_data_sizes)/sizeof(all_data_sizes[0]);
    size_t algorithms_undertest_size = sizeof(algorithms_undertest)/sizeof(algorithms_undertest[0]);
    size_t all_hash_functions_size = sizeof(all_hash_functions)/sizeof(all_hash_functions[0]);

    size_t total_configs = repeats_different_data * all_scales_size * configuration_count
                    * all_data_sizes_size * algorithms_undertest_size * all_hash_functions_size;
    size_t total_runs = repeats_same_data * total_configs; 

    std::cout << "This Benchmark has " << total_configs << " different configurations. It will run each config " 
        << repeats_same_data << " resulting in " << total_runs << " total runs\n";

    file_name = "benchmark_result.csv";

    create_result_file(file_name);


    //run variables
    T* data = nullptr;
    Scalar_group_count<T> *validation_baseline = nullptr;

    size_t noise_id = 1;
    //FOR REPRODUCIBLE DATA REMOVE THE FOLLOWING TWO LINES OF CODE!
    srand(std::time(nullptr));
    noise_id = std::rand();


    for(size_t adss = 0; adss < all_data_sizes_size; adss++){
        if(data != nullptr){
            free(data);
            data = nullptr;
        }

        size_t data_size = all_data_sizes[adss];
        data = (T*) aligned_alloc(64, data_size * sizeof(T)); // alternative
        
        for(size_t conf = 0; conf < configuration_count; conf++){// iterate over all configurations of collisions and clusters

            for(size_t ahfs = 0; ahfs < all_hash_functions_size; ahfs++){ // sets the hashfunction. different functions lead to different data    
                size_t (*function)(T, size_t) = all_hash_functions[ahfs];
            
                // for(size_t ass = 0; ass < all_scales_size; ass++){
                    float scale = 1.1f;//all_scales[ass];
                    size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);
                    std::cout << "data size: " << data_size <<"\tcluster config: " << conf 
                        << "\tfunction id: " << ahfs << "\tscale: " << scale << "\tHSIZE: " << HSIZE;
                
                    for(size_t rdd = 0; rdd < repeats_different_data; rdd++){
                        size_t seed = noise(noise_id++, 0);
                        generate_data_p1<T>( // the seed is for rdd the run id
                            data, 
                            data_size, 
                            distinct_value_count, 
                            HSIZE,
                            function,
                            collision_count[conf],
                            collision_size[conf],
                            cluster_count[conf],
                            cluster_size[conf],
                            seed
                        );

                        if(validation_baseline != nullptr){
                            delete validation_baseline;
                            validation_baseline = nullptr;
                        }

                        validation_baseline = new Scalar_group_count<T>(distinct_value_count * 2, &id_mod);
                        validation_baseline->create_hash_table(data, data_size);
                        std::cout << "\t\t\t\t" << rdd << "\tseed: " << seed << std::endl;

                        for(size_t aus = 0; aus < algorithms_undertest_size; aus++){
                            Algorithm test = algorithms_undertest[aus];    
                            std::string alg_identification = "";
                            size_t internal_HSIZE = HSIZE;

                            Group_count<T> *run;
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
                                default:
                                    std::cout << "One of the Algorithms isn't supported yet!\n";
                                    return -1;
                            }
                            
                            internal_HSIZE = run->get_HSIZE();
                            alg_identification = run->identify();
                            std::cout << "\t\t\t\t\t" << alg_identification << std::endl;
                            
                            for(size_t rsd = 0; rsd < repeats_same_data; rsd++){ // could be seen as a run id
                                std::cout << "\t\t\t\t\t\t" << rsd << "\ttime: " ;
                                
                                run->clear();

                                size_t time = 0;
                                
                                time = run_test<T>(
                                    run, 
                                    data, 
                                    data_size, 
                                    validation_baseline,
                                    distinct_value_count * 2,
                                    false,
                                    false
                                );
                                std::cout << time << "ns\n";
                                
                                write_to_file(//TODO!!!
                                    file_name, //string
                                    alg_identification, //string
                                // benchmark time
                                    time, //size_t or uint64_t
                                // config
                                    data_size,   // size_t 
                                    sizeof(T),
                                    distinct_value_count, // size_t 
                                    scale, // Scaleing factor for Hash Table double/float
                                    internal_HSIZE, // HASH Table Size size_t 
                                    ahfs, // hash function index size_t 
                                    seed, // Datageneration seed COULD BE REPLACED BY ANNOTHER ID BUT!  size_t 
                                    rsd, // run id (same config with same runs) size_t 
                                    collision_count[conf],
                                    collision_size[conf],
                                    cluster_count[conf],
                                    cluster_size[conf]
                                );
                            }
                            delete run;
                        }
                    }
                // }
            }
        }
    }

    time_end = time_now();
    size_t duration = duration_time(time_begin, time_end);
    std::cout << "\n\n\tIT TOOK\t" << duration << " ns OR\t" << (uint32_t)(duration / 1000000000.0) << " s OR\t" << (uint32_t)(duration / 60000000000.0)  << " min\n\n";

    return 0;
}

//---------------------------------------
// MAIN!
//---------------------------------------
// using ps_type = uint64_t;
using ps_type = uint32_t; 

int main(int argc, char** argv){
    test1<ps_type>();
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
    double duration_s;
    double data_amount = (data_size * sizeof(ps_type) * 8)/1000000000.0; // Gbit
    double data_count = data_size / 1000000000.0; // Million Values

    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    // std::cout << group_count->identify() << std::endl;
// run the test and time it
    time_begin = time_now();
    group_count->create_hash_table(data, data_size);;
    time_end = time_now();
    
    // std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
// out put result
    duration = duration_time(time_begin, time_end);
    duration_s = duration / 1000000000.0;

    // std::cout << "\tTime:\t" << duration << " ns\tOR\t" << duration_s << " s\n";
    // std::cout << "\tData:\t" << data_amount << " Gbit\tOR\t" << data_size << " Values\n"; 
    // std::cout << "\ttput:\t" << (data_amount)/(duration_s) << " Gbit/s\tor\tperf:\t" << (data_count)/(duration_s) << " Gval/s\n";
    // std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

// validate run
    bool errors = false;
    if(validate){
        errors = validation<ps_type>(group_count, validation_baseline, validation_size);
    }

    if(cleanup){
        delete group_count;
    }
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";

    if(errors){
        throw std::runtime_error("Problem during Validation!");
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
        myfile << "Algorithm,time,data_size,bytes,distinct_value_count,scale,hash_table_size,hash_function_ID,seed,run_ID,collision_count,collision_lenght,cluster_count,cluster_length\n";
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
    size_t ahfs, // hash function index size_t 
    size_t seed, // Datageneration seed COULD BE REPLACED BY ANNOTHER ID BUT!  size_t 
    size_t rsd, // run id (same config with same runs) size_t
    size_t config_collision_count,
    size_t config_collition_size,
    size_t config_cluster_count,
    size_t config_cluster_size 
){
    std::ofstream myfile;
    myfile.open (filename, std::ios::app);
    if(myfile.is_open()){
        // "Algorithm,time,data size,bytes,distinct value count,scale,hash table size,hash function ID,seed,run ID";
        myfile << alg_identification << "," << time << "," 
            << data_size << "," << bytes 
            << "," << distinct_value_count  << "," << scale << "," 
            << HSIZE << "," << ahfs << "," 
            << seed << "," << rsd << "," 
            << config_collision_count << "," << config_collition_size << "," 
            << config_cluster_count << "," << config_cluster_size << "\n"; 
        myfile.close();
    } else {
        throw std::runtime_error("Could not open file to write results!");
    }
//std::cout << alg_identification << "\t" << time << "\t" << rsd << "\t" << scale << "\t" << distinct_value_count << "\t" << seed << "\t" << data_size << std::endl;
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
    return key % HSIZE;
}