#ifndef TUD_HASHING_TESTING_DATAGEN
#define TUD_HASHING_TESTING_DATAGEN

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <array>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <vector>
#include <time.h>
#include <tuple>
#include <utility>

#include <string>

#include "datagen_help.hpp"

using namespace std;


// todo with one big cluster where we just place the data in a hashmap the same size of distinct data (so we just create collisions on the smallest size)
// todo cluster leaks into collisions!

// delete functions p1_stringify_number(), p1_stringify() and #include <sstream>
// reason: <sstream> requires GLIBCXX_3.4.26 , which leads to further incompatibilities in the DevCloud

/*
    Density sets the information about the data layout between the values.
    In the DENSE Case this should mean that the numbers are in an interval from [x:y]
        with a step size of 1.
    In the SPARSE case the numbers are taken at random from the full range of values.    
*/
enum class Density{DENSE, SPARSE};
std::string density_to_string(Density x){
    switch(x){
        case Density::DENSE:
            return "dense";
        case Density::SPARSE:
            return "sparse";
    }
    return "unknown";
}

/*
    The Distribution gives us another tuning factor.
    For this we disregard the order of the keys.
    With NORMAL we try to generate the data such that the different keys follow a
        normal distributuion.
    With UNIFORM we try to achieve a uniform distribution. This includes some variations
        in the relative frequencies.
*/
enum class Distribution{NORMAL, UNIFORM};
std::string distribution_to_string(Distribution x){
    switch(x){
        case Distribution::NORMAL:
            return "normal";
        case Distribution::UNIFORM:
            return "uniform";
    }
    return "unknown";
}

/*
    With Generation we try to achieve the same affect the paper: "A Seven-Dimensional 
        Analysis of Hashing Methods and its Implications on Query Processing"
    With the FLAT generation we have no prerequirements of how the data should look like.
    GRID on the other hand has the prerequrement that every 
*/
enum class Generation{FLAT, GRID};
std::string generation_to_string(Generation z){
    switch(z){
        case Generation::FLAT:
            return "flat";
        case Generation::GRID:
            return "grid";
    }
    return "unknown";
}



/*
    How collisions groups should be aligned. BAD and GOOD come with another parameter on which it should be alignt 
*/
enum class Alignment{UNALIGNED, BAD, GOOD};
std::string alignment_to_string(Alignment x){
    switch(x){
        case Alignment::UNALIGNED:
            return "unaligned";
        case Alignment::BAD:
            return "bad";
        case Alignment::GOOD:
            return "good";
    }
    return "unknown";
}



//Creates a GRID number described by the paper: A Seven Dimensional Analysis of Hashing Methods and its Implications on Query Processing
template<typename T>
T make_grid(size_t x){
    size_t halfword[] ={0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE};
    size_t nr_half = 14;
    T result = 0;

    for(size_t i = 0; i < sizeof(T); i++){
        size_t k = x % nr_half;
        x /= nr_half;
     
        result ^= halfword[k] << (8 * i);
    }
    
    return result;
}


/*
    helping function that turns the index into random numbers
*/
template<typename T>
void flat_number_gen(std::vector<T>& numbers, std::vector<size_t> index, size_t distinct_values){
    for(size_t i = 0; numbers.size() < distinct_values && i < index.size(); i++){
        T num = index[i];
        bool in = (num == 0);

        for(T x: numbers){
            if(x == num){
                in = true;
                break;
            }
        }

        if(!in){
            numbers.push_back((T)num);
        }
    }
}

/*
    helping function that turns the index into random numbers
*/
template<typename T>
void grid_number_gen(std::vector<T>& numbers, std::vector<size_t> index, size_t distinct_values){
    for(size_t i = 0; numbers.size() < distinct_values && i < index.size(); i++){
        // size_t id = index[i];
        T num = make_grid<T>(i);
        bool in = (num == 0);

        for(T x: numbers){
            if(x == num){
                in = true;
                break;
            }
        }

        if(!in){
            numbers.push_back((T)num);
        }
    }
}

/*
    random number generator for different layouts dense and sparse
*/
void index_dense(std::vector<size_t>& index, size_t distinct_values, size_t start){
    for(size_t i = start; i < distinct_values+start; i++){
        index.push_back(i);
    }
}

void index_sparse(std::vector<size_t>&index, size_t distinct_values, size_t seed){   
    for(size_t i = 1; i <= distinct_values; i++){
        index.push_back(noise(i, seed));
    }
}



size_t p1_parameter_gen_max_collision_group(size_t distinct_value, size_t HSIZE_DATAGEN){
    // std::cout << "col_group\n";
    return HSIZE_DATAGEN/2;
}

size_t p1_parameter_gen_max_collisions(size_t distinct_value, size_t HSIZE_DATAGEN, size_t collision_count){
    // std::cout << "cols\n";

    if(collision_count == 0){
        return 0;
    }

    size_t r =  (HSIZE_DATAGEN/collision_count) - 1;
    
    if(r > HSIZE_DATAGEN){
        return 0;
    }
    return r;
}

size_t p1_parameter_gen_max_cluster(size_t distinct_value, size_t HSIZE_DATAGEN, size_t collision_count, size_t collisions){
    // std::cout << "clust\n";
    if(collisions == 0){
        collision_count = 0;
    }

    if(collision_count == 0){
        return HSIZE_DATAGEN/2;
    }
    int64_t x = distinct_value - collision_count * collisions;
    if(x <= 0){
        return 0;
    }

    size_t left = HSIZE_DATAGEN - collision_count * (collisions + 1);
    return left/2 + collision_count;
}

size_t p1_parameter_gen_max_cluster_length(size_t distinct_value, size_t HSIZE_DATAGEN, size_t collision_count, size_t collisions, size_t cluster){
    // std::cout << "clust_len\n";
    // std::cout << distinct_value << "\t" << HSIZE_DATAGEN << "\t" << collision_count << "\t" << collisions << "\t" << cluster << "\n";
    // 5 6 cc 1 cs 5 clu 1
    if(cluster == 0){
        return 0;
    }
    if(collisions == 0){
        collision_count = 0;
    }
    if(collision_count == 0){
        size_t r =  (HSIZE_DATAGEN / cluster) - 1;
        if(r > distinct_value){
            // std::cout << "r0\n";
            return 0;
        }
        // std::cout << "r1\n";
        return r;
    }

    size_t x = (HSIZE_DATAGEN / cluster) - 1;
    if(x > HSIZE_DATAGEN){ 
        x = 0;
    }

    if(x > collisions){
        // for p1 the cluster can extend a collision groups size so that it achieves the necessary cluster length.
        // std::cout << "r2\n";
        return x;
    }

    size_t remaining_clusters = cluster - collision_count;
    if(remaining_clusters > cluster || remaining_clusters == 0){
        // std::cout << "r3\n";
        return 0;
    }

    size_t left = HSIZE_DATAGEN - collision_count * (collisions + 1);
    // std::cout << "\t" << left << "\t" << remaining_clusters << "\t" << left/remaining_clusters - 1 << std::endl;
    // std::cout << "r4\n";
    return left/remaining_clusters - 1;
}

size_t p1_parameter_gen_distinct(size_t collision_count, size_t collisions, size_t cluster, size_t cluster_lenght){
    // std::cout << "distinct_calc\n";
    size_t a = collision_count;
    size_t a_l = collisions;
    size_t b = cluster;
    size_t b_l = cluster_lenght;

    if(a == 0){
        a_l = 0;
    }
    if(b == 0){
        b_l = 0;
    }

    size_t c = a < b ? a : b;

    a = a - c;
    b = b - c;

    size_t m = a_l > b_l ? a_l : b_l;

    size_t res = c * m + a * a_l + b * b_l;
    return res;
}

size_t p1_parameter_gen_hsize(size_t collision_count, size_t collisions = 0, size_t cluster = 0, size_t cluster_lenght = 0){
    // std::cout << "hsize_calc\n";
    size_t a = collision_count;
    size_t a_l = collisions;
    size_t b = cluster;
    size_t b_l = cluster_lenght;

    size_t c = a < b ? a : b;

    if(a == 0){
        a_l = 0;
    }
    if(b == 0){
        b_l = 0;
    }

    a = a - c;
    b = b - c;

    a_l += 1;
    b_l += 1;

    size_t m = a_l > b_l ? a_l : b_l;

    size_t res = c * m + a * a_l + b * b_l;
    return res;
}

size_t p0_parameter_gen_hsize(size_t collision_count, size_t collisions = 0){
    return collision_count * collisions;
}

template<typename T>
size_t generate_data_p1(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t HSIZE_DATAGEN,
    size_t (*hash_function)(T, size_t),
    size_t collision_count,
    size_t collision_size,
    size_t cluster_count,
    size_t cluster_size,
    size_t seed
){
    std::multimap<size_t, T> all_numbers;
    std::vector<T> numbers;

    size_t expected_hsize = p1_parameter_gen_hsize(collision_count, collision_size, cluster_count, cluster_size);
    
    if(expected_hsize > HSIZE_DATAGEN || expected_hsize == 0){
        return 0; // HSIZE_DATAGEN is to small for the given configuration to fit.
    }
    if(seed == 0){
        return 0; // invalid seed
    }

    size_t pos = noise(HSIZE_DATAGEN * distinct_values, seed) % HSIZE_DATAGEN;
    size_t total_free = HSIZE_DATAGEN - distinct_values;
    size_t reserved_free = cluster_count> collision_count ? cluster_count: collision_count;
    reserved_free += 1;
    size_t distributed_free = total_free <= reserved_free ? 1 : total_free - reserved_free ;
    size_t cluster_after_collition_length = cluster_size > collision_size ? cluster_size - collision_size : 0;

    all_number_gen<T>(all_numbers, hash_function, HSIZE_DATAGEN, collision_size, seed+3);

    size_t i = 0;
    bool CREATE_CLUSTER = i < cluster_count;
    bool CREATE_COLLISION = i < collision_count;
    size_t remaining_cluster_length;

    while(CREATE_CLUSTER || CREATE_COLLISION){
        remaining_cluster_length = cluster_size;
        if(CREATE_COLLISION){
            generate_collision<T>(&numbers, &all_numbers, HSIZE_DATAGEN, pos, collision_size);
            pos = (pos + collision_size) % HSIZE_DATAGEN;
            remaining_cluster_length = cluster_after_collition_length;
        }
        
        if(CREATE_CLUSTER){
            generate_cluster<T>(&numbers, &all_numbers, HSIZE_DATAGEN, pos, remaining_cluster_length);
            pos = (pos + remaining_cluster_length) %HSIZE_DATAGEN;
        }

        next_position(pos, distributed_free, HSIZE_DATAGEN, seed + 2);

        i++;
        CREATE_COLLISION = i < collision_count;
        CREATE_CLUSTER = i < cluster_count;
    }
    
    if(numbers.size() == 0){
        return 0;
    }
    
    generate_benchmark_data<T>(result, data_size, &numbers, seed+3);    
    return numbers.size();
}


template<typename T>
size_t generate_data_p0(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t (*hash_function)(T, size_t),
    size_t collision_count,
    size_t collision_size,
    size_t seed,
    bool just_distinct_values = false // NOTE: this should only be used for datageneration testing
){
    std::multimap<size_t, T> all_numbers;   
    std::vector<T> numbers;

    size_t expected_hsize = p0_parameter_gen_hsize(collision_count, collision_size);

    if(expected_hsize > distinct_values){
        return 0; // HSIZE_DATAGEN is to small for the given configuration to fit.
    }
    if(seed == 0){
        return 0; // invalid seed
    }
    
    size_t pos = noise(distinct_values * distinct_values, seed) % distinct_values;
    size_t free_space = distinct_values - expected_hsize;

    all_number_gen<T>(all_numbers, hash_function, distinct_values, collision_size, seed +3);

    size_t i = 0;
    bool CREATE_COLLISION = i < collision_count;
    while(CREATE_COLLISION){
        generate_collision<T>(&numbers, &all_numbers, distinct_values, pos, collision_size);
        pos = (pos + collision_size) % distinct_values;
        
        i++;
        CREATE_COLLISION = i < collision_count;
    }

    generate_cluster<T>(&numbers, &all_numbers, distinct_values, pos, free_space);
    pos = (pos + free_space) % distinct_values;

    if(numbers.size() == 0 || numbers.size() > distinct_values){
        return 0;
    }

    if(just_distinct_values){
        for(size_t i = 0; i < numbers.size(); i++){
            result[i] = numbers[i];
        }
    }

    generate_benchmark_data<T>(result, data_size, &numbers, seed+1 );    
    return data_size;
}


/*
    Data generator with different options for data layout. 
    DOES NOT ALLOCATE THE MEMORY JUST FILLS IT!
*/
template<typename T>
size_t generate_data(
    T*& result, 
    size_t data_size,   // number of values to be generated
    size_t distinct_values, // number of distinct values
    Density den = Density::DENSE,
    Generation gen = Generation::FLAT,
    Distribution dis = Distribution::UNIFORM,
    size_t start = 0,   // starting offset for consecutive numbers (dense)
    size_t seed = 0     // for sparse number generation 0 true random, 1.. reproducible
){
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    } 
    
    double mul = 1.5;
    size_t retries = 0;
    retry:
    mul++;
    retries++;
    if(retries < 10)
    {
        std::vector<size_t> index;
        std::vector<T> numbers;
        switch(den){
        case Density::DENSE:
            index_dense(index, distinct_values*mul, start);
            break;
        case Density::SPARSE:
            index_sparse(index, distinct_values*mul, seed);
            break;    
        default:
            throw std::runtime_error("Unknown Density input");
        }
        
        switch (gen){
        case Generation::FLAT:
            flat_number_gen<T>(numbers, index, distinct_values);
            break;
        case Generation::GRID:
            grid_number_gen<T>(numbers, index, distinct_values);
            break;
        default:
            throw std::runtime_error("Unknown Generation methoed input");    
        }
        
        if(numbers.size() < distinct_values){
            goto retry;
        }
        switch(dis){
        case Distribution::NORMAL:
            throw std::runtime_error("Normal Distribution not yet implemented");    
            break;
        case Distribution::UNIFORM:
            for(size_t i = 0; i < data_size; i++){
                size_t ran = noise(i, seed + start + 1) % distinct_values;
                result[i] = numbers[ran];
            }
            break;
        default:
            throw std::runtime_error("Unknown Distribution input");    
        }
    }else{
        throw std::runtime_error("To many retries during data generation.");
    }
    return seed;
}

#endif //TUD_HASHING_TESTING_DATAGEN