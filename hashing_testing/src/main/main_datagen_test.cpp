#include <iostream>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>


#include <unordered_map>
#include "datagenerator/datagen.hpp"
#include "hash_function.hpp"
#include "benchmark/table.hpp"

std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}


template <typename T>
void analyse1(T* data, size_t data_size){
    uint64_t bits = sizeof(T) * 8;
    size_t total_bits = 0;
    size_t *bit_at_position = new size_t[bits];
    
    for(size_t i = 0; i < bits; i++){
        bit_at_position[i] = 0;
    }
    
    for(size_t i = 0; i < data_size; i++){
        size_t b_count = 0;
        T val = data[i];
        for(size_t b = 0; b < bits; b++){
            bit_at_position[b] += (val >> b) & 0x1;
            b_count += (val >> b) & 0x1;
        }
        total_bits += b_count;
    }
    
    std::cout << "Average Nr of Bits set:\t" << (total_bits * 1.) / data_size << "\t of " << bits << " Bits\n";

    std::cout << "Nr of Bits per position\n";
    for(size_t i = 0; i < bits; i++){
        std::cout << "\t" << (int)((bit_at_position[i]* 1.) / data_size * 1000)/1000.;
        if((i+1)%16 == 0){
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

template <typename T>
void analyse2(T* data, size_t data_size){
    uint64_t bits = sizeof(T) * 8;

    size_t bit_at_position[bits][bits][4];
    for(size_t a = 0; a < bits; a++){
        for(size_t b = 0; b < bits; b++){
            for(size_t i = 0; i < 4; i++){
                bit_at_position[a][b][i] = 0;
            }
        }
    }

    for(size_t i = 0; i < data_size; i++){
        T val = data[i];
        for(size_t a = 0; a < bits; a++){
            for(size_t b = 0; b < bits; b++){
                size_t pos = 0;
                pos += ((val >> a) & 0x1) << 1;
                pos += (val >> b) & 0x1;

                bit_at_position[a][b][pos]++;
            }
        }
    }

    for(size_t a = 0; a < bits; a++){
        for(size_t b = 0; b < bits; b++){
            std::cout << a << "\t" << b << ":\t";
            double x = 0;
            for(size_t i = 0; i < 4; i++){
                x += (i + 1) * (bit_at_position[a][b][i] * 1.) / data_size;
                std::cout << "\t" << (bit_at_position[a][b][i] * 1.) / data_size;
            }
            std::cout << "\t" << (int)(x * 1000)/1000.0;
            if(b%2 == 1){
                std::cout << std::endl;
            }else{
                std::cout << "\t\t\t";
            }
        }
    }
    



}



using ps_type = uint8_t;

int main(int argc, char** argv){
    size_t distinct = 9;
    size_t hsize = 1.7 * distinct + 1;
    size_t data_size = distinct * 3;

    ps_type* data = new ps_type[data_size];

    size_t generated = generate_data_v5<ps_type>(data, data_size, distinct, hsize, &modulo<ps_type>, 0, 0);

    std::cout << "Generated Numbers:\t" << generated << std::endl; 

    std::cout << (uint32_t)data[0];
    for(size_t i =1; i < data_size; i++){
        std::cout << ", " << (uint32_t) data[i];
    }
    std::cout << std::endl;

/*

    size_t data_size = 2048 * 3;
    uint32_t *data = new uint32_t[data_size];

    size_t distinct = 1024;
    size_t elements = 16;

    float s = 1.1f;
    float all_scales[] = {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f};
    size_t all_scales_size = sizeof(all_scales)/sizeof(all_scales[0]);

    size_t HSIZE = distinct * s;
    double spielraum = 0.00;


    size_t p1_max_col_group = p1_parameter_gen_max_collision_group(distinct, HSIZE);
    size_t valid_combinations = 0;
    std::unordered_map<std::string, int64_t> configuarations = {{"", 1}};

    fill_tab_table();
    HashFunction functions_to_test[] = {
        // HashFunction::MULTIPLY_PRIME,
        // HashFunction::MULITPLY_SHIFT, 
        // HashFunction::MULTIPLY_ADD_SHIFT, 
        HashFunction::MODULO, 
        // HashFunction::MURMUR, 
        // HashFunction::TABULATION,
        HashFunction::NOISE
    };
    size_t number_hash_functions = sizeof(functions_to_test) / sizeof(functions_to_test[0]);

    size_t collision_count;
    size_t collision_size;

    collision_count = 1;
    // collision_count = 8;
    // collision_count = 16;
    collision_size = distinct/collision_count;


    size_t ok;
    for(size_t a = 0; a < number_hash_functions; a++){
        hash_fptr<uint32_t> function = get_hash_function<uint32_t>(functions_to_test[a]);
        std::cout << "\n\n" << get_hash_function_name(functions_to_test[a]);

        auto time_start = time_now();
        size_t HSIZE = distinct * s + 0.5f;

        size_t test_size = 0xFF00000;

        for(size_t i = 0; i < test_size; i++){
            function(i, HSIZE);
        }

        auto time_end = time_now();
        std::cout<< "\tnano seconds per operation:\t" <<((duration_time(time_start, time_end)*100)/test_size)/100. << std::endl;

        ok = generate_data_p0<uint32_t>(data, data_size, distinct, function, collision_count, collision_size, 1794768573511499073, true);
        double score = 0;
        for(size_t b = 0; b < all_scales_size; b++){
            size_t min = distinct;
            size_t max = 0;

            s = all_scales[b];
            float bonus_scale = 1;
            HSIZE = distinct * s * bonus_scale + 0.5f;
            HSIZE = (HSIZE + elements - 1);
            HSIZE /= elements;
            HSIZE *= elements;


            size_t control[HSIZE];
            size_t bucket = 0;
            for(size_t i = 0; i < HSIZE; i++){
                control[i] = 0;
            }
            for(size_t i = 0; i < distinct; i++){
                control[function(data[i], HSIZE)]++;
            }
            for(size_t i = 0; i < HSIZE; i++){
                if(control[i] > max){
                    max = control[i];
                }
                if(control[i] < min && control[i] != 0){
                    min = control[i];
                }

                bucket += control[i] != 0;
            }
            score += (bucket * 1.0 / (distinct/16));
            std::cout << "\tscale: "<< s << "\tHSIZE: " << HSIZE << "\tbuckets used: " << bucket << "  \t";
            std::cout << "min: " << min << "  \tmax: " << max << std::endl;
            
            // if(bucket <= 32 && bucket > 1){
            //     std::cout << "\t\tbucket-info:";
            //     for(size_t i = 0; i < HSIZE; i++){
            //         if(control[i] != 0){
            //             std::cout << "\t" << i << ":" << control[i];
            //         }
            //     }
            //     std::cout << std::endl;
            // }
        
        
        }
        score = (int32_t)(score / all_scales_size * 1000)/10.;
        std::cout<< "\tscore: " << score << std::endl;
    }
    
    size_t N = 4;
    size_t i = 0;
    size_t l = range_bit_based2(i, N);
    size_t _s = 1;
    std::cout << __builtin_clz(N) << std::endl;

    // size_t max = 4294967295;
    size_t max = N << 5;

    bool end_it = false;
    std::cout << l << "\t" << i << "\t-\t";
    while( i <= max){
        i++;
        size_t l_n = range_bit_based2(i, N);
        if(l_n != l){
            end_it = l_n == 0;
            l = l_n;
            std::cout << i - 1 << "\tsize: " << _s << std::endl;
            _s = 0;
            if(i < max)
                std::cout << l << "\t" << i << "\t-\t";
        }
        _s++;
    }
    std::cout << std::endl;



    // for(size_t col_group = 0; col_group <= p1_max_col_group && p1_parameter_gen_hsize(col_group) <= HSIZE; col_group++){

    //     size_t p1_max_cols = p1_parameter_gen_max_collisions(distinct, HSIZE, col_group);
    //     for(size_t cols = 1; cols <= p1_max_cols && p1_parameter_gen_hsize(col_group, cols) <= HSIZE; cols++){//might not be correct ! cols might have to start at 0

    //         size_t p1_max_clust = p1_parameter_gen_max_cluster(distinct, HSIZE, col_group, cols);
    //         for(size_t clust = 0; clust <= p1_max_clust && p1_parameter_gen_hsize(col_group, cols, clust) <= HSIZE; clust++){
                
    //             size_t p1_max_clust_len = p1_parameter_gen_max_cluster_length(distinct, HSIZE, col_group, cols, clust);
    //             for(size_t clust_len = 0; clust_len <= p1_max_clust_len && p1_parameter_gen_hsize(col_group,cols, clust, clust_len) <= HSIZE; clust_len++){
                    
                    
    //                 size_t r_dis1 = p1_parameter_gen_distinct(col_group, cols, clust, clust_len);
    //                 size_t r_hsize = p1_parameter_gen_hsize(col_group, cols, clust, clust_len);

    //                 std::string *trans = p1_stringify( HSIZE, col_group, cols, clust, clust_len);
    //                 std::string config = *trans;
    //                 delete(trans);
    //                 bool IS_DISTINCT_VALUE = r_dis1 == distinct;
    //                 bool DISTINCT_IS_IN_RANGE = (r_dis1 <= HSIZE/(s-spielraum) && r_dis1 >= HSIZE/(s+spielraum));
    //                 bool NEEDED_HSIZE_FITS = r_hsize <= HSIZE;

    //                 bool NOT_USED_CONFIG = configuarations.count(config) == 0;


    //                 if((IS_DISTINCT_VALUE || DISTINCT_IS_IN_RANGE) && NEEDED_HSIZE_FITS && NOT_USED_CONFIG){
    //                     // std::cout << " ( " << IS_DISTINCT_VALUE << " || " << DISTINCT_IS_IN_RANGE << " ) && " << NEEDED_HSIZE_FITS << " && " << NOT_USED_CONFIG << std::endl; 
    //                     configuarations.insert(std::make_pair(config, 1));

    //                     // std::cout << distinct << " -> " << HSIZE<< "\t" << p1_max_col_group << " -> " << col_group <<"\t" << p1_max_cols << " -> " << cols << "\t" << p1_max_clust  << " -> " << clust 
    //                     //     << "\t" << p1_max_clust_len  << " -> " << clust_len << "\n";
    //                     std::cout << "config:\t" << col_group <<"\t" << cols << "\t" << clust  << "\t" << clust_len << "\t\t" << config << std::endl;
    //                     size_t r_dis2 = generate_data_p1(data, data_size, distinct, HSIZE, &mod, col_group, cols, clust, clust_len, 1);
    //                     // std::cout << "RESULT:\t" << r_dis1 << " = " <<  r_dis2<< std::endl ;//<< std::endl; 
    //                     valid_combinations++;
    //                 }
    //             }
    //         }
    //     }
    // }

    // std::cout << "debugging\n";
    // std::string *k = p1_stringify(6, 1, 3, 2, 1);
    // std::cout << "results in: " << *k << std::endl;
    // delete k;

    // std::cout << "VALID COMBINATIONS\t" << valid_combinations << std::endl;


    // for(size_t i = 0; i < data_size &&  i < 10; i++){
    //     std::cout << data[i] << "\t" << mod(data[i], HSIZE) <<std::endl;
    // }
    delete []data;

*/
}