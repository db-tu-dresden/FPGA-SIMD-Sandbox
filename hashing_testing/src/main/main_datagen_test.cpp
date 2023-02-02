#include <iostream>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>


#include <unordered_map>
#include "datagen.hpp"




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


template<typename T>
size_t mod(T nr, size_t HSIZE){
    return nr % HSIZE;
}

// simple multiplicative hashing function
size_t hashx(uint32_t key, size_t HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}


using ps_type = uint64_t;

int main(int argc, char** argv){

    size_t data_size = 2048;
    uint32_t *data = new uint32_t[data_size];

    size_t distinct = 2048;
    double s = 1.1;
    size_t HSIZE = distinct * s;
    double spielraum = 0.00;


    size_t p1_max_col_group = p1_parameter_gen_max_collision_group(distinct, HSIZE);
    size_t valid_combinations = 0;
    std::unordered_map<std::string, int64_t> configuarations = {{"", 1}};

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

    std::cout << "VALID COMBINATIONS\t" << valid_combinations << std::endl;
    size_t ok;
    ok = generate_data_p1(data, data_size, distinct, HSIZE, &mod, 1, distinct, 0, 0, 1);

    for(size_t i = 0; i < data_size &&  i < 10; i++){
        std::cout << data[i] << "\t" << mod(data[i], HSIZE) <<std::endl;
    }
    delete []data;

}