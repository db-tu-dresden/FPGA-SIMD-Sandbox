#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cmath>


#include "main/utility.hpp"

#include "main/hash_function.hpp"
#include "operator/physical/group_count/group_count_handler/group_count_algorithms.hpp"
#include "main/fileio/file.hpp"
#include "main/datagenerator/datagenerator.hpp"

using ps_type = uint32_t;
size_t repeats_same_data = 2;
size_t repeats_different_data = 1;
size_t repeats_different_layout = 2;

/*
* Generation types:
*   0: strided
*   1: bad
*/
const uint8_t MAX_GENERATION_TYPES = 2;


double percentage_print = 0.5;

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

template<typename T> 
size_t benchmark(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t count_alg_ut, HashFunction *hashfunctions_undertest, size_t count_hf_ut, size_t data_gen_typ_mask = 1);


int main(int argc, char** argv){
    fill_tab_table();

    //TODO user input so we don't need to recompile all the time!

    size_t distinct_value_count = 12 * 1024;
    size_t data_amount = 4 * 1024 * 1024;

    float scale_boost = 1.0f;


    Group_Count_Algorithm algorithms_undertest[] = {
        Group_Count_Algorithm::SCALAR_GROUP_COUNT_SOA,
        Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS,
        // Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS_V2,
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_CONFLICT_V1, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V2, 
        Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V2 
    };

    HashFunction hashfunctions_undertest[] = {
        HashFunction::MODULO
    };

    size_t num_alg_undertest = sizeof(algorithms_undertest) / sizeof(algorithms_undertest[0]);
    size_t num_hashfunc_undertest = sizeof(hashfunctions_undertest) / sizeof(hashfunctions_undertest[0]);

    benchmark<ps_type>(data_amount, distinct_value_count,algorithms_undertest, num_alg_undertest, hashfunctions_undertest, num_hashfunc_undertest);
    // distinct_value_count *= 2;
    // distinct_value_count = 2048;
    // strided_benchmark<ps_type>(data_amount, distinct_value_count,algorithms_undertest, num_alg_undertest, hashfunctions_undertest, num_hashfunc_undertest);
    // distinct_value_count *= 2;
    // strided_benchmark<ps_type>(data_amount, distinct_value_count,algorithms_undertest, num_alg_undertest, hashfunctions_undertest, num_hashfunc_undertest);
    // distinct_value_count *= 2;
    // strided_benchmark<ps_type>(data_amount, distinct_value_count,algorithms_undertest, num_alg_undertest, hashfunctions_undertest, num_hashfunc_undertest);
    // distinct_value_count *= 2;
    // strided_benchmark<ps_type>(data_amount, distinct_value_count,algorithms_undertest, num_alg_undertest, hashfunctions_undertest, num_hashfunc_undertest);

}


// runs the given algorithm with the given data. Afterwards it might clear the hash_table (reset = true) and or deletes the operator (clean up)
template <typename T>
size_t run_test(Group_count<T>*& group_count, T* data, size_t data_size, bool cleanup = false, bool reset = true){
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



// benchmarking the algorithms with the strided data generator. The data generator creates planned collision data.
// with higher scaling factors gabs are introduced in a orderly and uniform fashion. The values are "slotted" and the gabs come after every slot. 
template<typename T> 
size_t benchmark(size_t data_size, size_t distinct_value_count, Group_Count_Algorithm *algorithms_undertest, size_t count_alg_ut, HashFunction *hashfunctions_undertest, size_t count_hf_ut, size_t data_gen_typ_mask){
    size_t seed = 0;
    srand(std::time(nullptr));
    seed = std::rand();
    std::chrono::high_resolution_clock::time_point time_begin;
    time_begin = time_now();

    std::stringstream file_name_builder;
    file_name_builder << "strided_benchmark_" << data_size << "_" << distinct_value_count << ".csv";
    std::string file_name = file_name_builder.str();
    create_strided_benchmark_result_file(file_name);

    size_t collisions_steps = distinct_value_count;
    size_t diminish = 4;
    size_t max_count_collisions = std::ceil(std::log(collisions_steps)/std::log(diminish)) + 1;
    size_t count_collisions = 100;
    if(max_count_collisions < count_collisions){
        count_collisions = max_count_collisions;
    }

    double scale_factors[] = {1, 2, 4, 8, 16};
    size_t count_sf = sizeof(scale_factors) / sizeof(scale_factors[0]);
    double max_scale_factor = 1;
    for(size_t i = 0; i < count_sf; i++){
        if(max_scale_factor < scale_factors[i]){
            max_scale_factor = scale_factors[i];
        }
    }

    
    // double percentage_done  = -percentage_print;
    size_t runs_done = 0;

    size_t total_configs = count_alg_ut * count_hf_ut * count_sf * count_collisions;
    size_t total_runs = total_configs * repeats_same_data * repeats_different_data;

    std::cout << "stided benchmark has " << total_configs << " different Configs.\nThis results in " << total_runs << " total runs\n";
    std::cout << "percentage done:" << std::endl;
    
    T* data = nullptr;
    data = (T*) aligned_alloc(64, data_size * sizeof(T));
    Group_count<T> *alg = nullptr;
    Datagenerator<T> *datagen = nullptr;

    for(size_t hash_function_id = 0; hash_function_id < count_hf_ut; hash_function_id++){
        HashFunction function_id = hashfunctions_undertest[hash_function_id];
        hash_fptr<T> function = get_hash_function<T>(function_id);

        for(size_t random_data = 0; random_data < repeats_different_data; random_data++){
            
            size_t data_seed = noise(random_data, seed);

            create_Datagenerator(
                datagen,
                distinct_value_count * 1024 * 4,
                function,
                max_scale_factor * 2,
                data_seed
            );

            for(size_t scale_factors_id = 0; scale_factors_id < count_sf; scale_factors_id++){
                double current_scale = scale_factors[scale_factors_id];
                size_t hsize = distinct_value_count * current_scale;
                
                datagen->transform_hsize(hsize);
                bool success_datagen = datagen->transform_finalise();
                if(!success_datagen){
                    std::stringstream error_stream;
                    error_stream << "Datagenerator transformation not successful: "<< data_seed << " " << hsize;
                    throw std::runtime_error(error_stream.str());
                }

                for(size_t random_layout = 0; success_datagen &&  random_layout < repeats_different_layout; random_layout++){
                    size_t layout_seed = noise(random_layout, data_seed);

                    size_t local_steps = collisions_steps;
                    for(size_t collisions_id = count_collisions; collisions_id > 0; collisions_id--){
                        size_t collisions = local_steps;
                        local_steps /= diminish;

                        for(uint8_t data_generation_count = 0; data_generation_count < MAX_GENERATION_TYPES; data_generation_count++){
                            bool generated = (data_gen_typ_mask >> data_generation_count) & 0b1;
                            if(generated){
                                switch (data_generation_count)
                                {
                                case 0:
                                    datagen->get_data_strided(data, data_size, distinct_value_count, collisions, layout_seed);
                                    break;
                                case 1:
                                    datagen->get_data_bad(data, data_size, distinct_value_count, collisions, layout_seed);
                                    break;
                                default:
                                    std::stringstream error_stream;
                                    error_stream << "Unknown Datageneration function. Wanted Generation: " << data_generation_count;
                                    throw std::runtime_error(error_stream.str());
                                    break;
                                }
                            }

                            for(size_t alg_id = 0; generated && alg_id < count_alg_ut; alg_id++){
                                getGroupCount(alg, algorithms_undertest[alg_id], hsize, function);
                                std::string alg_identification = alg->identify();
                                size_t internal_HSIZE = alg->get_HSIZE();

                                for(size_t run_id = 0; run_id < repeats_same_data; run_id++){
                                    alg->clear();
                                    size_t time = 0;
                                    time = run_test<T>(alg, data, data_size);
                                    write_to_strided_benchmark_file(
                                        file_name, alg_identification, time, data_size,
                                        sizeof(T), distinct_value_count, current_scale, internal_HSIZE,
                                        function_id, data_seed, run_id, collisions
                                    );

                                    runs_done++;

                                    status_output(runs_done, total_runs, percentage_print, time_begin);
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

    status_output(runs_done, total_runs, percentage_print, time_begin, true);
    return 0;
}