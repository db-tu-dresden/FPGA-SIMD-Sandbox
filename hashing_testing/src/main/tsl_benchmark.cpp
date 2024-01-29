#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cmath>


#include <numa.h>

#include "main/utility.hpp"

#include "main/hash_function.hpp"
#include "operator/physical/group_count/group_count_handler/group_count_algorithms.hpp"
#include "main/fileio/file.hpp"
#include "main/datagenerator/datagenerator.hpp"

#include <tslintrin.hpp>
#include "main/tsl_benchmark.hpp"

#include <thread>



#include <omp.h>

using ps_type = uint32_t;

/*
* Generation types:
*   0: strided
*   1: bad
*/
const uint8_t MAX_GENERATION_TYPES = 2;

double percentage_print = 0.25;

template<typename T>
void safe_numa_free(T*& data, size_t count){
    if(data != nullptr){
        numa_free(data, count * sizeof(T));
        data = nullptr;
    }
}

template<typename T>
void safe_numa_alloc(T*& data, size_t count, size_t numa_node){
    if(data != nullptr){
        safe_numa_free<T>(data, count);
    }
    data = (T*) numa_alloc_onnode(count * sizeof(T), numa_node);
}

template<typename T>
void move_data(T*& data, size_t count, size_t numa_node){    
    std::cout << "moveing data" << std::flush;
    T* data_new = nullptr;
    safe_numa_alloc<T>(data_new, count, numa_node);
    
    if(data != nullptr){    //just copy data if data not nullptr
        std::memcpy(data_new, data, count * sizeof(T));
    }
    
    safe_numa_free(data, count);
    data = data_new;
    std::cout << " - done\n";
}

template<typename T>
void create_Datagenerator(
    Datagenerator<T> *& datagen,
    size_t hsize, 
    size_t (*hash_function)(T, size_t),
    size_t max_collision_size, 
    size_t number_seed)
{
    if(datagen != nullptr){
        delete datagen;
    }
    datagen = new Datagenerator<T>(hsize, hash_function, max_collision_size, number_seed);
}

size_t seq(size_t*&result, size_t min, size_t max, size_t step){
    
    if(min == max){
        max++;
    }
    if(min > max){
        min ^= max;
        max ^= min;
        min ^= max;
    }
    if(step < 1){
        step = 1;
    }

    size_t count = (max - min) / step;

    result = (size_t*) malloc(sizeof(size_t) * count);
    for(size_t i = 0; i < count; i++){
        result[i] = i * step + min;
    }

    return count;
}

std::chrono::high_resolution_clock::time_point time_begin;
const size_t exec_node = 0;
const size_t max_planed_collisions = 256;

int main(int argc, char** argv){
    fill_tab_table();

    //TODO user input so we don't need to recompile all the time!
    size_t distinct_value_count = 8 * 1024 * 1024;
    size_t build_data_amount = distinct_value_count * 16;
    size_t probe_data_amount = distinct_value_count * 16;

    size_t repeats_same_data = 3;
    size_t repeats_different_data = 1;
    size_t repeats_different_layout = 1;

    Group_Count_Algorithm_TSL algorithms_undertest[] = {
        Group_Count_Algorithm_TSL::LCP_SOA,
        Group_Count_Algorithm_TSL::LP_H_SOA
    };
    size_t num_alg_undertest = sizeof(algorithms_undertest) / sizeof(algorithms_undertest[0]);


    Base_Datatype datatypes_undertest[] = {
        // Base_Datatype::UI8,
        // Base_Datatype::UI16,
        // Base_Datatype::UI32,
        Base_Datatype::UI64
    };
    size_t num_datatypes_undertest = sizeof(datatypes_undertest)/ sizeof(datatypes_undertest[0]);

    Vector_Extention extentions_undertest[] = {
        // Vector_Extention::SCALAR,
        Vector_Extention::SSE,
        Vector_Extention::AVX2,
        Vector_Extention::AVX512
    };
    size_t num_extentions_undertest = sizeof(extentions_undertest)/ sizeof(extentions_undertest[0]);

    HashFunction hashfunctions_undertest[] = {
        HashFunction::MODULO
    };
    size_t num_hashfunc_undertest = sizeof(hashfunctions_undertest) / sizeof(hashfunctions_undertest[0]);

    // double scale_factors[] = {1., 2., 4., 8., 16.};
    double scale_factors[] = {2., 4., 8., 16.};
    // double scale_factors[] = {8., 16.};
    size_t num_scale_factors = sizeof(scale_factors)/sizeof(scale_factors[0]);


    size_t max_collision = distinct_value_count / 64;
    size_t num_collision_tests = 3;
    size_t collision_diminish = max_collision + 1;
    
    if(num_collision_tests > 1){
        collision_diminish = std::ceil(std::pow(10, std::log10(max_collision)/(num_collision_tests - 1)));
        if(collision_diminish <= 1){
            collision_diminish = 2;
        }
        size_t max_collision_count = std::ceil(std::log(max_collision)/std::log(collision_diminish)) + 1;
        if(num_collision_tests > max_collision_count){
            std::cout << "WARNING: The Number of distinct values is to small to run " 
                << num_collision_tests 
                << " well spaced collision tests.\n\tThe number of collision test gets set to: "
                << max_collision_count << std::endl; 
            num_collision_tests = max_collision_count;
        }
    }
//TODO: 
// Nice to have for easier switching of data generation types: Functionptr for memberfunction for different layout generation methods.
    float selectivities[] = {1, 0.75};
    size_t num_select = sizeof(selectivities) / sizeof(selectivities[0]);

    size_t min_mem_numa = 0;
    size_t max_mem_numa = 1;
    size_t step_size_numa = 1;

    size_t *hash_table_locations;
    size_t num_hash_table_locations = seq(hash_table_locations, min_mem_numa, max_mem_numa, step_size_numa);
    size_t *build_data_locations;
    size_t num_build_data_locations = seq(build_data_locations, min_mem_numa, max_mem_numa, step_size_numa);
    size_t *probe_data_locations;
    size_t num_probe_data_locations = seq(probe_data_locations, min_mem_numa, max_mem_numa, step_size_numa);


    const size_t num_concurrent_build_tests = 1;
    // if(false){
    if(true){
        build_benchmark(
            distinct_value_count,
            build_data_amount,
            hash_table_locations,
            num_hash_table_locations,
            build_data_locations,
            num_build_data_locations, 
            repeats_different_data,
            repeats_same_data,
            repeats_different_layout,

            algorithms_undertest, 
            num_alg_undertest,
            datatypes_undertest,
            num_datatypes_undertest,

            extentions_undertest,
            num_extentions_undertest,
            hashfunctions_undertest,
            num_hashfunc_undertest,

            scale_factors,
            num_scale_factors,

            max_collision,
            num_collision_tests,
            collision_diminish
        );
    }else{
        probe_benchmark(
            distinct_value_count,
            build_data_amount,
            probe_data_amount,
            hash_table_locations,
            num_hash_table_locations,
            probe_data_locations,
            num_probe_data_locations, 
            repeats_different_data,
            repeats_same_data,
            repeats_different_layout,

            algorithms_undertest, 
            num_alg_undertest,
            datatypes_undertest,
            num_datatypes_undertest,

            extentions_undertest,
            num_extentions_undertest,
            hashfunctions_undertest,
            num_hashfunc_undertest,

            scale_factors,
            num_scale_factors,

            max_collision,
            num_collision_tests,
            collision_diminish,
            selectivities,
            num_select
        );
    }

}



template<typename T, class Vec> 
void build_benchmark_final(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    T* data,
    size_t data_size,
    size_t hsize,
    size_t hash_table_loc,
    hash_fptr<T> function,
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    size_t& run_count,
    const size_t max_run_count
){
    // std::cout << "config_string build_benchmark_final:\t" << config_string << std::endl;
    Group_Count_TSL_SOA<T> *alg = nullptr;


    for(size_t algorithms_undertest_i = 0; algorithms_undertest_i < num_algorithms_undertest; algorithms_undertest_i++){
        getTSLGroupCount<Vec, T>(alg, algorithms_undertest[algorithms_undertest_i], hsize, function, hash_table_loc);

        for(size_t run = 0; run < repeats_same_data; run++){
            // std::cout << "\t\t" << algorithms_undertest_i << "\t" << run << std::endl;
            alg->clear();
            size_t time = 0;

            // std::thread thread([alg, data, data_size, &time](){
            time += run_test_build<T>(alg, data, data_size);
            time += run_test_build<T>(alg, data, data_size);
            // });
            // cpu_set_t cpuset;
            // CPU_ZERO(&cpuset);
            // CPU_SET(0, &cpuset);
            // int rc = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
            // thread.join();
            

            std::stringstream config_ss;
            config_ss << config_string << ",algorithm,reported_hsize,run,time";
            std::stringstream result_ss;
            result_ss << result_string << "," << alg->identify() << "," << alg->get_HSIZE() << "," << run << "," << time;
            if(run_count == 0){
                write_to_file(result_file_name, config_ss.str(), true);
            }
            write_to_file(result_file_name, result_ss.str());
            bool force = run_count == 0;
            status_output(++run_count, max_run_count, percentage_print, time_begin, force);
        }
    }

    if(alg != nullptr){
        delete alg;
    }
}

template<typename T>
void build_benchmark_vector_extention(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    T* data,
    size_t data_size,
    size_t hsize,
    size_t hash_table_loc,
    hash_fptr<T> function,
    Vector_Extention ve,
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t repeats_same_data,               // how often to repeat all the experiments with the same data
    size_t& run_count,
    const size_t max_run_count
){
    switch (ve)
    {
    case Vector_Extention::SCALAR:
        build_benchmark_final<T, tsl::scalar>(
            result_file_name, config_string, result_string, 
            data, data_size, hsize, hash_table_loc, 
            function, algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations, repeats_same_data, 
            run_count, max_run_count);
        break;

    case Vector_Extention::SSE:
        build_benchmark_final<T, tsl::sse>(
            result_file_name, config_string, result_string, 
            data, data_size, hsize, hash_table_loc, 
            function, algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations, repeats_same_data, 
            run_count, max_run_count);
        break;

    case Vector_Extention::AVX2:
        build_benchmark_final<T, tsl::avx2>(
            result_file_name, config_string, result_string, 
            data, data_size, hsize, hash_table_loc, 
            function, algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations, repeats_same_data, 
            run_count, max_run_count);
        break;

    case Vector_Extention::AVX512:
        build_benchmark_final<T, tsl::avx512>(
            result_file_name, config_string, result_string, 
            data, data_size, hsize, hash_table_loc, 
            function, algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations, repeats_same_data, 
            run_count, max_run_count);
        break;
    default:
        std::cout << "unknown Vector Extention" << std::endl;
        break;
    }
}


template<typename T>
void build_benchmark_data(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset
    T* data, 
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
){
    // std::cout << "config_string build_benchmark_data:\t" << config_string << std::endl;
    
    for(size_t scale_i = 0; scale_i < num_scale_factors; scale_i++){
        double scale = scale_factors[scale_i];

        size_t hsize = distinct_value_count * scale;
        hsize += (scale <= 1);
        
        datagen->transform_hsize(hsize);
        for(size_t hash_location_i = 0; hash_location_i < num_hash_table_locations; hash_location_i++){
            size_t loc = hash_table_locations[hash_location_i];
            
            size_t collisions = max_collision_size;
            for(size_t i = 0; i < num_collision_test; i++){
                std::cout << "data gen:\t" << std::flush;
                std::chrono::high_resolution_clock::time_point tb = time_now();
                datagen->get_data_strided(data, build_data_count, distinct_value_count, collisions, seed);
                std::chrono::high_resolution_clock::time_point te = time_now();
                std::cout << "it took ";
                print_time(tb, te, false);
                std::cout << " seconds to generate the build data\n";

                for(size_t ve_id = 0; ve_id < num_extentions_undertest; ve_id++){
                    Vector_Extention ve = extentions_undertest[ve_id];

                    std::stringstream config_ss;
                    config_ss <<"vector_extention,"<< config_string << ",scale,hsize,table_location,collision_count";
                    std::stringstream result_ss;
                    result_ss << vector_extention_to_string(ve) << "," << result_string << "," << scale << "," << hsize << "," << loc << "," << collisions;
                    build_benchmark_vector_extention<T>(
                        result_file_name, config_ss.str(), result_ss.str(), 
                        data, build_data_count, hsize, loc, 
                        function, ve,  algorithms_undertest, 
                        num_algorithms_undertest, hash_table_locations, num_hash_table_locations, 
                        repeats_same_data, run_count, max_run_count);
                }
                collisions /= collision_diminish;
            }
        }
        
    }
}

template<typename T> 
void build_benchmark_datagen(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    Vector_Extention* extentions_undertest, // which vector extentions should be tested
    size_t num_extentions_undertest,
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
){
    // std::cout << "config_string build_benchmark_datagen:\t" << config_string << std::endl;
    //datagenerator type
    Datagenerator<T> *datagen = nullptr;
    double max_scale_factor = 1;
    for(size_t scale_id = 0; scale_id < num_scale_factors; scale_id++){
        if(scale_factors[scale_id] > max_scale_factor){
            max_scale_factor = scale_factors[scale_id];
        }
    }

    T* data = nullptr;

    for(size_t hash_function_id = 0; hash_function_id < num_hashfunctions_undertest; hash_function_id++){
        HashFunction function_id = hashfunctions_undertest[hash_function_id];
        hash_fptr<T> function = get_hash_function<T>(function_id);
        
        size_t max_collisions_to_generate = max_collision_size;
        size_t max_planed_collisions = (512 * 4) / (8 * sizeof(T));
        if(max_collisions_to_generate > max_planed_collisions){
            max_collisions_to_generate = max_planed_collisions;
        }
        std::chrono::high_resolution_clock::time_point tb = time_now();
        create_Datagenerator(
            datagen,
            distinct_value_count * max_scale_factor,
            function,
            max_collisions_to_generate,
            seed
        );
        std::chrono::high_resolution_clock::time_point te = time_now();
        std::cout << "it took ";
        print_time(tb, te, false);
        std::cout << " seconds to generate the raw data\n";
        for(size_t ht_loc_i = 0; ht_loc_i < num_build_data_locations; ht_loc_i++){
            
            size_t loc = build_data_locations[ht_loc_i];
            safe_numa_free<T>(data, build_data_count);
            safe_numa_alloc<T>(data, build_data_count, loc);

            for(size_t different_layout_i = 0; different_layout_i < repeats_different_layout; different_layout_i++){
                size_t layout_seed = noise(different_layout_i, noise(seed, seed));

                std::stringstream config_ss;
                config_ss << config_string << ",build_data_location,hash_function,dataseed,layoutseed";
                std::stringstream result_ss;
                result_ss << result_string << "," << loc  << "," << get_hash_function_name(function_id) << "," << seed << "," << layout_seed;

                build_benchmark_data<T>(
                    result_file_name,
                    config_ss.str(),
                    result_ss.str(),
                    extentions_undertest, 
                    num_extentions_undertest,
                    distinct_value_count,
                    build_data_count,
                    data,
                    hash_table_locations,
                    num_hash_table_locations,
                    build_data_locations,
                    num_build_data_locations,
                    repeats_same_data,
                    algorithms_undertest,
                    num_algorithms_undertest,
                    function,
                    scale_factors,
                    num_scale_factors,
                    max_collision_size,
                    num_collision_test,
                    collision_diminish,
                    datagen,
                    layout_seed,
                    run_count,
                    max_run_count
                );
            }
        }
    }
    if(datagen != nullptr){
        delete datagen;
    }
    seed++;
    
    safe_numa_free<T>(data, build_data_count);
}

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
){
    // std::cout << "config_string build_benchmark_template_helper_ps:\t" << config_string << std::endl;
    switch(base){
        case Base_Datatype::UI8:
            build_benchmark_datagen<uint8_t>(
                result_file_name,
                config_string,
                result_string,
                extentions_undertest, 
                num_extentions_undertest,
                distinct_value_count,
                build_data_count,
                hash_table_locations,
                num_hash_table_locations,
                build_data_locations,
                num_build_data_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                seed,
                run_count,
                max_run_count
            );
            break;
        case Base_Datatype::UI16:
            build_benchmark_datagen<uint16_t>(
                result_file_name,
                config_string,
                result_string,
                extentions_undertest, 
                num_extentions_undertest,
                distinct_value_count,
                build_data_count,
                hash_table_locations,
                num_hash_table_locations,
                build_data_locations,
                num_build_data_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                seed,
                run_count,
                max_run_count
            );
            break;
        case Base_Datatype::UI32:
            build_benchmark_datagen<uint32_t>(
                result_file_name,
                config_string,
                result_string,
                extentions_undertest, 
                num_extentions_undertest,
                distinct_value_count,
                build_data_count,
                hash_table_locations,
                num_hash_table_locations,
                build_data_locations,
                num_build_data_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                seed,
                run_count,
                max_run_count
            );
            break;
        case Base_Datatype::UI64:
            build_benchmark_datagen<uint64_t>(
                result_file_name,
                config_string,
                result_string,
                extentions_undertest, 
                num_extentions_undertest,
                distinct_value_count,
                build_data_count,
                hash_table_locations,
                num_hash_table_locations,
                build_data_locations,
                num_build_data_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                seed,
                run_count,
                max_run_count
            );
            break;
        default:
            std::cout << "Unknown Basetype\n";
    }
}

/*
*   Benchmarking the build phase of the hashalgorithm
*/
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
    size_t seed
){
    
    time_begin = time_now();
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    }
    // std::cout << "configs" << num_hash_table_locations  << "\t" << num_build_data_locations  << "\t" << num_algorithms_undertest << "\t" << num_datatypes_undertest << "\t" 
    //     << num_extentions_undertest  << "\t" << num_hashfunctions_undertest << "\t" << num_scale_factors << "\t" << num_collision_test;
    size_t configs = num_hash_table_locations 
                    * num_build_data_locations 
                    * num_algorithms_undertest 
                    * num_datatypes_undertest 
                    * num_extentions_undertest 
                    * num_hashfunctions_undertest
                    * num_scale_factors
                    * num_collision_test;
                    
    size_t repeats = repeats_different_data
                    * repeats_same_data
                    * repeats_different_layout;
    
    size_t total_tests = configs * repeats;

    std::cout << "TSL build benchmark is going to evaluate: " << configs << ". Every config will be rerun " << repeats << " different times.\n";
    std::cout << "\tIn total " << total_tests << " will be run" << std::endl;

    std::stringstream file_name_builder;
    file_name_builder << "tsl_build_benchmark_" << build_data_count << "_" << distinct_value_count << ".csv";
    std::string file_name = file_name_builder.str();
    
    size_t run_count = 0;

    for(size_t seed_id = 0; seed_id < repeats_different_data; seed_id++){
        size_t data_seed = noise(seed_id, seed);

        for(size_t bt_id = 0; bt_id < num_datatypes_undertest; bt_id++){
            Base_Datatype bt = datatypes_undertest[bt_id];
            
            std::stringstream config_ss;
            config_ss << "datatype";
            std::stringstream result_ss;
            result_ss << base_datatype_to_string(bt);

            build_benchmark_template_helper_base(
                file_name,
                config_ss.str(),
                result_ss.str(),
                bt,
                extentions_undertest,
                num_extentions_undertest,
                distinct_value_count,
                build_data_count, 
                hash_table_locations,
                num_hash_table_locations,
                build_data_locations,
                num_build_data_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                data_seed,
                run_count,
                total_tests
            );
        }
    }

    status_output(run_count, total_tests, percentage_print, time_begin, true);
}

// runs the given algorithm with the given data. Afterwards it might clear the hash_table (reset = true) and or deletes the operator (clean up)
template <class T>
size_t run_test_build(Group_Count_TSL_SOA<T>*& group_count, T* data, size_t data_size, bool cleanup, bool reset){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
    size_t seed    
){
    time_begin = time_now();
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    }
   
    size_t configs = num_hash_table_locations 
                    * num_probe_locations 
                    * num_algorithms_undertest 
                    * num_datatypes_undertest 
                    * num_extentions_undertest 
                    * num_hashfunctions_undertest
                    * num_scale_factors
                    * num_collision_test
                    * num_selectivities;
                    
    size_t repeats = repeats_different_data
                    * repeats_same_data
                    * repeats_different_layout;
    
    size_t total_tests = configs * repeats;

    std::cout << "TSL probe benchmark is going to evaluate: " << configs << ". Every config will be rerun " << repeats << " different times.\n";
    std::cout << "\tIn total " << total_tests << " will be run" << std::endl;

    std::stringstream file_name_builder;
    file_name_builder << "tsl_probe_benchmark_" << probe_data_count << "_" << distinct_value_count << ".csv";
    std::string file_name = file_name_builder.str();
    
    size_t run_count = 0;

    for(size_t seed_id = 0; seed_id < repeats_different_data; seed_id++){
        size_t data_seed = noise(seed_id, seed);

        for(size_t bt_id = 0; bt_id < num_datatypes_undertest; bt_id++){
            Base_Datatype bt = datatypes_undertest[bt_id];
            
            std::stringstream config_ss;
            config_ss << "datatype";
            std::stringstream result_ss;
            result_ss << base_datatype_to_string(bt);

            probe_benchmark_template_helper_base(
                file_name,
                config_ss.str(),
                result_ss.str(),
                distinct_value_count,
                build_data_count,
                probe_data_count, 
                bt,
                extentions_undertest,
                num_extentions_undertest,
                hash_table_locations,
                num_hash_table_locations,
                probe_locations,
                num_probe_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                selectivities,
                num_selectivities,
                data_seed,
                run_count,
                total_tests
            );
        }
    }
    status_output(run_count, total_tests, percentage_print, time_begin, true);
}

void probe_benchmark_template_helper_base(
    std::string result_file_name, 
    std::string config_string,
    std::string result_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    const size_t probe_data_count,
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
){
    // std::cout << "config_string build_benchmark_template_helper_ps:\t" << config_string << std::endl;
    switch(base){
        case Base_Datatype::UI8:
            probe_benchmark_datagen<uint8_t>(
                result_file_name,
                config_string,
                result_string,
                distinct_value_count,
                build_data_count,
                probe_data_count,
                extentions_undertest, 
                num_extentions_undertest,
                hash_table_locations,
                num_hash_table_locations,
                probe_locations,
                num_probe_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                selectivities,
                num_selectivities,
                seed,
                run_count,
                max_run_count
            );
            break;
        case Base_Datatype::UI16:
            probe_benchmark_datagen<uint16_t>(
                result_file_name,
                config_string,
                result_string,
                distinct_value_count,
                build_data_count,
                probe_data_count,
                extentions_undertest, 
                num_extentions_undertest,
                hash_table_locations,
                num_hash_table_locations,
                probe_locations,
                num_probe_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                selectivities,
                num_selectivities,
                seed,
                run_count,
                max_run_count
            );
            break;
        case Base_Datatype::UI32:
            probe_benchmark_datagen<uint32_t>(
                result_file_name,
                config_string,
                result_string,
                distinct_value_count,
                build_data_count,
                probe_data_count,
                extentions_undertest, 
                num_extentions_undertest,
                hash_table_locations,
                num_hash_table_locations,
                probe_locations,
                num_probe_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                selectivities,
                num_selectivities,
                seed,
                run_count,
                max_run_count
            );
            break;
        case Base_Datatype::UI64:
            probe_benchmark_datagen<uint64_t>(
                result_file_name,
                config_string,
                result_string,
                distinct_value_count,
                build_data_count,
                probe_data_count,
                extentions_undertest, 
                num_extentions_undertest,
                hash_table_locations,
                num_hash_table_locations,
                probe_locations,
                num_probe_locations,
                repeats_different_data,
                repeats_same_data,
                repeats_different_layout,
                algorithms_undertest,
                num_algorithms_undertest,
                hashfunctions_undertest,
                num_hashfunctions_undertest,
                scale_factors,
                num_scale_factors,
                max_collision_size,
                num_collision_test,
                collision_diminish,
                selectivities,
                num_selectivities,
                seed,
                run_count,
                max_run_count
            );
            break;
        default:
            std::cout << "Unknown Basetype\n";
    }
}

template<typename T> 
void probe_benchmark_datagen(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    const size_t probe_data_count,
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
){
    // std::cout << "config_string build_benchmark_datagen:\t" << config_string << std::endl;
    //datagenerator type
    Datagenerator<T> *datagen = nullptr;
    double max_scale_factor = 1;
    for(size_t scale_id = 0; scale_id < num_scale_factors; scale_id++){
        if(scale_factors[scale_id] > max_scale_factor){
            max_scale_factor = scale_factors[scale_id];
        }
    }

    T* data = nullptr;

    for(size_t hash_function_id = 0; hash_function_id < num_hashfunctions_undertest; hash_function_id++){
        HashFunction function_id = hashfunctions_undertest[hash_function_id];
        hash_fptr<T> function = get_hash_function<T>(function_id);
        size_t max_collisions_to_generate = max_collision_size;
        size_t max_planed_collisions = (512 * 4) / (8 * sizeof(T));
        if(max_collisions_to_generate > max_planed_collisions){
            max_collisions_to_generate = max_planed_collisions;
        }
        std::cout << "max collision:\t" << max_collision_size << "\t" << std::flush;
        std::chrono::high_resolution_clock::time_point tb = time_now();
        create_Datagenerator(
            datagen,
            distinct_value_count * max_scale_factor,
            function,
            max_collisions_to_generate,
            seed
        );
        std::chrono::high_resolution_clock::time_point te = time_now();
        std::cout << "it took ";
        print_time(tb, te, false);
        std::cout << " seconds to generate the raw data\n";


        size_t loc = 0;
        
        safe_numa_free<T>(data, build_data_count);
        safe_numa_alloc<T>(data, build_data_count, loc);

        size_t layout_seed = noise(seed, noise(seed, seed));

        std::stringstream config_ss;
        config_ss << config_string << ",hash_function,dataseed,build_layoutseed";
        std::stringstream result_ss;
        result_ss << result_string << "," << get_hash_function_name(function_id) << "," << seed << "," << layout_seed;

        probe_benchmark_data<T>(
            result_file_name,
            config_ss.str(),
            result_ss.str(),
            distinct_value_count,
            build_data_count,
            probe_data_count,
            extentions_undertest, 
            num_extentions_undertest,
            data,
            hash_table_locations,
            num_hash_table_locations,
            probe_locations,
            num_probe_locations,
            repeats_same_data,
            algorithms_undertest,
            num_algorithms_undertest,
            function,
            scale_factors,
            num_scale_factors,
            max_collision_size,
            num_collision_test,
            collision_diminish,
            selectivities,
            num_selectivities,
            datagen,
            layout_seed,
            run_count,
            max_run_count
        );
    }
    if(datagen != nullptr){
        delete datagen;
    }
    seed++;
    
    safe_numa_free<T>(data, build_data_count);
}


template<typename T>
void probe_benchmark_data(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t distinct_value_count,      // how many different distinct values should be inserted into the hash table
    const size_t build_data_count,          // how many values should be included in the dataset 
    const size_t probe_data_count,
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
){
    size_t original_numa_node = 0;
    T* probe_data = nullptr;
    T* result_data = nullptr;
    safe_numa_alloc(probe_data, probe_data_count, original_numa_node);
    safe_numa_alloc(result_data, probe_data_count, original_numa_node);
    for(size_t scale_i = 0; scale_i < num_scale_factors; scale_i++){
        double scale = scale_factors[scale_i];

        size_t hsize = distinct_value_count * scale;
        hsize += scale <= 1;
        
        datagen->transform_hsize(hsize);
    
        size_t collisions = max_collision_size;
        for(size_t i = 0; i < num_collision_test; i++){
            std::cout << "data gen:\t" << std::flush;
            std::chrono::high_resolution_clock::time_point tb = time_now();
            datagen->get_data_strided(data, build_data_count, distinct_value_count, collisions, seed);
            std::chrono::high_resolution_clock::time_point te = time_now();
            std::cout << "it took ";
            print_time(tb, te, false);
            std::cout << " seconds to generate the build data\n";

            
            for(size_t sel_i = 0; sel_i < num_selectivities; sel_i++){
                float selectivity = selectivities[sel_i];

                std::cout << "data gen:\t" << std::flush;
                std::chrono::high_resolution_clock::time_point tb2 = time_now();
                datagen->get_probe_strided(probe_data, probe_data_count, selectivity, seed);
                std::chrono::high_resolution_clock::time_point te2 = time_now();
                std::cout << "it took ";
                print_time(tb2, te2, false);
                std::cout << " seconds to generate probe data\n";
                for(size_t ve_id = 0; ve_id < num_extentions_undertest; ve_id++){
                    Vector_Extention ve = extentions_undertest[ve_id];

                    std::stringstream config_ss;
                    config_ss <<"vector_extention,"<< config_string << ",scale,hsize,collision_count,selectivity";
                    std::stringstream result_ss;
                    result_ss << vector_extention_to_string(ve) << "," << result_string << "," << scale << "," << hsize << "," << collisions << "," << selectivity;
                    probe_benchmark_vector_extention<T>(
                        result_file_name, config_ss.str(), result_ss.str(), 
                        build_data_count, probe_data_count, datagen, 
                        data, probe_data, result_data, hsize, function, 
                        ve, algorithms_undertest, num_algorithms_undertest, 
                        hash_table_locations, num_hash_table_locations, 
                        probe_locations, num_probe_locations,
                        repeats_same_data, seed, run_count, max_run_count);
                }
                collisions /= collision_diminish;
            }
        }
    }
    safe_numa_free(probe_data, probe_data_count);
    safe_numa_free(result_data, probe_data_count);
}

template<typename T>
void probe_benchmark_vector_extention(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t build_data_count,
    const size_t probe_data_count,
    Datagenerator<T> *datagen,
    T* data,
    T* probe_data,
    T* result_data,
    size_t hsize,
    hash_fptr<T> function,
    Vector_Extention ve,
    Group_Count_Algorithm_TSL* algorithms_undertest,
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    size_t repeats_same_data,
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
){
    switch (ve)
    {
    case Vector_Extention::SCALAR:
        probe_benchmark_hash_table<T, tsl::scalar>(
            result_file_name, config_string, result_string, 
            build_data_count, probe_data_count, datagen, 
            data, probe_data, result_data, hsize, function, 
            algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations,
            probe_locations, num_probe_locations,
            repeats_same_data, seed, run_count, max_run_count);
        break;

    case Vector_Extention::SSE:
        probe_benchmark_hash_table<T, tsl::sse>(
            result_file_name, config_string, result_string, 
            build_data_count, probe_data_count, datagen, 
            data, probe_data, result_data, hsize, function, 
            algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations,
            probe_locations, num_probe_locations,
            repeats_same_data, seed, run_count, max_run_count);
        break;

    case Vector_Extention::AVX2:
        probe_benchmark_hash_table<T, tsl::avx2>(
            result_file_name, config_string, result_string, 
            build_data_count, probe_data_count, datagen, 
            data, probe_data, result_data, hsize, function, 
            algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations,
            probe_locations, num_probe_locations,
            repeats_same_data, seed, run_count, max_run_count);
        break;

    case Vector_Extention::AVX512:
        probe_benchmark_hash_table<T, tsl::avx512>(
            result_file_name, config_string, result_string, 
            build_data_count, probe_data_count, datagen, 
            data, probe_data, result_data, hsize, function, 
            algorithms_undertest, num_algorithms_undertest, 
            hash_table_locations, num_hash_table_locations,
            probe_locations, num_probe_locations,
            repeats_same_data, seed, run_count, max_run_count);
        break;

    default:
        std::cout << "unknown Vector Extention" << std::endl;
        break;
    }
}

template<typename T, class Vec>
void probe_benchmark_hash_table(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t build_data_count,
    const size_t probe_data_count,
    Datagenerator<T> *datagen,
    T* data,
    T* probe_data,
    T* result_data,
    size_t hsize,
    hash_fptr<T> function,
    Group_Count_Algorithm_TSL* algorithms_undertest, // which algorithms to test
    size_t num_algorithms_undertest,
    size_t* hash_table_locations,  // where to create the hash table
    size_t num_hash_table_locations,
    size_t* probe_locations,
    size_t num_probe_locations,
    size_t repeats_same_data,
    size_t seed,
    size_t& run_count,
    const size_t max_run_count
){
    Group_Count_TSL_SOA<T> *alg = nullptr;
    size_t ori_ht_loc = 0;
    for(size_t algorithms_undertest_i = 0; algorithms_undertest_i < num_algorithms_undertest; algorithms_undertest_i++){
        getTSLGroupCount<Vec, T>(alg, algorithms_undertest[algorithms_undertest_i], hsize, function, ori_ht_loc);
        std::cout << "table gen:\t" << std::flush;
        std::chrono::high_resolution_clock::time_point tb = time_now();
        alg->create_hash_table(data, build_data_count);
        std::chrono::high_resolution_clock::time_point te = time_now();
        std::cout << "it took ";
        print_time(tb, te, false);
        std::cout << " seconds to build the hash_table\n";

        for(size_t ht_loc_i = 0; ht_loc_i < num_hash_table_locations; ht_loc_i++){
            size_t ht_loc = hash_table_locations[ht_loc_i];
            alg->move_numa(ht_loc);

            std::stringstream config_ss;
            config_ss << config_string << ",algorithm,reported_hsize,hash_table_location";
            std::stringstream result_ss;
            result_ss << result_string << "," << alg->identify() << "," << alg->get_HSIZE() << "," << ht_loc;
            
            probe_benchmark_final<T, Vec>(
                result_file_name, config_ss.str(), result_ss.str(), 
                probe_data_count, datagen, alg,
                probe_data, result_data,
                probe_locations, num_probe_locations,
                repeats_same_data, seed, run_count, max_run_count
            );
        }
    }
    if(alg != nullptr){
        delete alg;
    }
}

template<typename T, class Vec> 
void probe_benchmark_final(
    std::string result_file_name,
    std::string config_string,
    std::string result_string,
    const size_t probe_data_count,
    Datagenerator<T> *datagen,
    Group_Count_TSL_SOA<T> *alg,
    T* probe_data,
    T* result_data,
    size_t* probe_locations,
    size_t num_probe_locations,
    size_t repeats_same_data,           
    size_t seed, 
    size_t& run_count,
    const size_t max_run_count
){
  
    for(size_t numa_node_i = 0; numa_node_i < num_probe_locations; numa_node_i++){
        size_t probe_loc = probe_locations[numa_node_i];
        
        move_data(probe_data, probe_data_count, probe_loc);
        move_data(result_data, probe_data_count, probe_loc);
        
        for(size_t run = 0; run < repeats_same_data; run++){
            
            size_t time = 0;
            time += run_test_probe<T>(alg, probe_data, result_data, probe_data_count); 
            
            std::stringstream config_ss;
            config_ss << config_string << ",probe_location,run,time";
            std::stringstream result_ss;
            result_ss << result_string << "," << probe_loc << "," << run << "," << time;
            if(run_count == 0){
                write_to_file(result_file_name, config_ss.str(), true);
            }
            write_to_file(result_file_name, result_ss.str());
            bool force = run_count == 0;
            status_output(++run_count, max_run_count, 1, time_begin, force);
        }
    }
}

// runs the given algorithm with the given data. Afterwards it might clear the hash_table (reset = true) and or deletes the operator (clean up)
template <typename T>
size_t run_test_probe(Group_Count_TSL_SOA<T>*& group_count, T* data, T * result, size_t data_size){
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    
    time_begin = time_now();
    group_count->probe(result, data, data_size);;
    time_end = time_now();
    
    size_t duration = duration_time(time_begin, time_end);
    return duration;
}