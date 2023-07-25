#ifndef TUD_HASHING_TESTING_DATAGEN
#define TUD_HASHING_TESTING_DATAGEN

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>

#include <sstream>
#include <string>

#include "datagen_help.hpp"


// todo with one big cluster where we just place the data in a hashmap the same size of distinct data (so we just create collisions on the smallest size)
// todo cluster leaks into collisions!


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
        size_t id = index[i];
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



size_t p1_parameter_gen_max_collision_group(size_t distinct_value, size_t HSIZE){
    // std::cout << "col_group\n";
    return HSIZE/2;
}

size_t p1_parameter_gen_max_collisions(size_t distinct_value, size_t HSIZE, size_t collision_count){
    // std::cout << "cols\n";

    if(collision_count == 0){
        return 0;
    }

    size_t r =  (HSIZE/collision_count) - 1;
    
    if(r > HSIZE){
        return 0;
    }
    return r;
}

size_t p1_parameter_gen_max_cluster(size_t distinct_value, size_t HSIZE, size_t collision_count, size_t collisions){
    // std::cout << "clust\n";
    if(collisions == 0){
        collision_count = 0;
    }

    if(collision_count == 0){
        return HSIZE/2;
    }
    int64_t x = distinct_value - collision_count * collisions;
    if(x <= 0){
        return 0;
    }

    size_t left = HSIZE - collision_count * (collisions + 1);
    return left/2 + collision_count;
}

size_t p1_parameter_gen_max_cluster_length(size_t distinct_value, size_t HSIZE, size_t collision_count, size_t collisions, size_t cluster){
    // std::cout << "clust_len\n";
    // std::cout << distinct_value << "\t" << HSIZE << "\t" << collision_count << "\t" << collisions << "\t" << cluster << "\n";
    // 5 6 cc 1 cs 5 clu 1
    if(cluster == 0){
        return 0;
    }
    if(collisions == 0){
        collision_count = 0;
    }
    if(collision_count == 0){
        size_t r =  (HSIZE / cluster) - 1;
        if(r > distinct_value){
            // std::cout << "r0\n";
            return 0;
        }
        // std::cout << "r1\n";
        return r;
    }

    size_t x = (HSIZE / cluster) - 1;
    if(x > HSIZE){ 
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

    size_t left = HSIZE - collision_count * (collisions + 1);
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

std::string* p1_stringify_number(size_t max_val, size_t val){
    val--;
    max_val--;
    size_t m = max_val;
    size_t c = 0;
    while(m > 0){
        m /= 26;
        c++;
    }
    std::stringstream result;
    bool first = true;
    for(size_t i = 0; i < c; i++){
        if(i==0){
            result << (char)('A' + (val%26));
        }else{
            result << (char)('a' + (val%26));
        }
            val /= 26;
    }
    return new std::string(result.str());
}


std::string* p1_stringify( size_t HSIZE, size_t collision_count, size_t collisions, size_t cluster, size_t cluster_lenght){
    int64_t table [HSIZE];
    int64_t col_a = 0;
    int64_t clu_a = 0;
    size_t h_val = 1;
    size_t pos = 0;


    if(p1_parameter_gen_hsize(collision_count, collisions, cluster, cluster_lenght) > HSIZE){
        return new std::string("");
    }

    for(size_t i = 0; i < HSIZE; i++){
        table[i] = 0;
    }

    for(; col_a < collision_count; col_a++, clu_a++){
        int64_t i = 0;
        for( ; i < collisions; i++){
            table[i + pos] = h_val;
        }
        pos += collisions;
        h_val++;
        for(; i < cluster_lenght && clu_a < cluster;  i++){
            table[pos] = h_val;
            pos++;
            h_val++;
        }
        pos++;
    }

    for(; clu_a < cluster; clu_a++){
        for(int64_t i = 0; i < cluster_lenght && clu_a < cluster;  i++){
            table[pos] = h_val;
            pos++;
            h_val++;
        }
        pos++;
    }

    std::stringstream result;
    result << h_val << ":";
    for(size_t i = 0; i < HSIZE; i++){
        if(table[i] == 0){
            if((i+1) < HSIZE && table[i+1] != 0){
                result << "_";
            }else{
                break;
            }
        }else{
            std::string *helper;
            helper = p1_stringify_number(h_val, table[i]) ;
            result << *helper;
            delete helper; 
        }
    }
    return new std::string(result.str());
}



template<typename T>
size_t generate_data_p1(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t HSIZE,
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
    
    if(expected_hsize > HSIZE || expected_hsize == 0){
        return 0; // HSIZE is to small for the given configuration to fit.
    }
    if(seed == 0){
        return 0; // invalid seed
    }

    size_t pos = noise(HSIZE * distinct_values, seed) % HSIZE;
    size_t total_free = HSIZE - distinct_values;
    size_t reserved_free = cluster_count> collision_count ? cluster_count: collision_count;
    reserved_free += 1;
    size_t distributed_free = total_free <= reserved_free ? 1 : total_free - reserved_free ;
    size_t cluster_after_collition_length = cluster_size > collision_size ? cluster_size - collision_size : 0;

    all_number_gen<T>(all_numbers, hash_function, HSIZE, collision_size, seed+3);

    size_t i = 0;
    bool CREATE_CLUSTER = i < cluster_count;
    bool CREATE_COLLISION = i < collision_count;
    size_t remaining_cluster_length;

    while(CREATE_CLUSTER || CREATE_COLLISION){
        remaining_cluster_length = cluster_size;
        if(CREATE_COLLISION){
            generate_collision<T>(&numbers, &all_numbers, HSIZE, pos, collision_size);
            pos = (pos + collision_size) % HSIZE;
            remaining_cluster_length = cluster_after_collition_length;
        }
        
        if(CREATE_CLUSTER){
            generate_cluster<T>(&numbers, &all_numbers, HSIZE, pos, remaining_cluster_length);
            pos = (pos + remaining_cluster_length) %HSIZE;
        }

        next_position(pos, distributed_free, HSIZE, seed + 2);

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
        throw std::runtime_error("The configuration asks for to many numbers! increase distinct_values or decrease the collision parameters.");    
        return 0; // HSIZE is to small for the given configuration to fit.
    }
    if(seed == 0){
        throw std::runtime_error("The seed may not be zero!");
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
    return numbers.size();
}




// 
template<typename T>
size_t generate_data_v3(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t hsize,
    size_t (*hash_function)(T, size_t),
    size_t collision_chain_count,
    size_t collision_chain_length,
    size_t seed,
    bool best_case,
    bool just_distinct_values = false // NOTE: this should only be used for datageneration testing
){

    std::multimap<size_t, T> all_numbers;   
    std::vector<T> numbers_collision;
    std::vector<T> numbers_single;

    // std::vector<T> all_numbers;


    size_t expected_hsize = p0_parameter_gen_hsize(collision_chain_count, collision_chain_length);

    if(expected_hsize > hsize){
        throw std::runtime_error("The configuration asks for to many numbers! increase distinct_values or decrease the collision parameters.");    
        return 0; // HSIZE is to small for the given configuration to fit.
    }
    if(seed == 0){
        throw std::runtime_error("The seed may not be zero!");
        return 0; // invalid seed
    }
    

    all_number_gen<T>(all_numbers, hash_function, hsize, collision_chain_length, seed+3);

    size_t pos = noise(hsize * distinct_values, seed) % hsize;
    size_t random_nrs = distinct_values - expected_hsize;
    size_t free_space = hsize - distinct_values;
    // std::cout << random_nrs << "\t" << free_space << std::endl;
    size_t skip_free;
    size_t max_skip = 8;    //todo maybe a good idea;
    size_t random_nrs_inbetween;
    size_t r_counter = 0;
    if(best_case){
        //generate collisions followed by empty space followed by clusters and emptyspace again for as long as there are collisions to generate
        for(size_t i = 0; i < collision_chain_count; i++){
            generate_collision<T>(&numbers_collision, &all_numbers, hsize, pos, collision_chain_length);
            next_position(pos, collision_chain_length, free_space, hsize, collision_chain_length + max_skip, seed + r_counter++);

            random_nrs_inbetween = 0;
            if(random_nrs > 1){
                random_nrs_inbetween = noise(random_nrs + pos, seed) % random_nrs - 1;
                random_nrs_inbetween++;
            }else if(random_nrs == 1){
                random_nrs_inbetween = 1;
            }
            random_nrs -= random_nrs_inbetween;
            
            generate_cluster<T>(&numbers_single, &all_numbers, hsize, pos, random_nrs_inbetween);
            next_position(pos, random_nrs_inbetween, free_space, hsize, random_nrs_inbetween + max_skip, seed + r_counter++);
        }
        generate_cluster<T>(&numbers_single, &all_numbers, hsize, pos, random_nrs);
    }else{
        //generate collision chains on different buckets and move the counter only one. 
        for(size_t i = 0; i < collision_chain_count; i++){
            generate_collision<T>(&numbers_collision, &all_numbers, hsize, pos, collision_chain_length);
            next_position(pos, collision_chain_length, free_space, hsize, 1, seed);
        }

        //generate cluster as long as there are random_nrs to distribute.
        while(free_space > 1 && random_nrs > 0){
            random_nrs_inbetween = 0;
            if(random_nrs > 1){
                random_nrs_inbetween = noise(random_nrs + pos, seed) % random_nrs - 1;
                random_nrs_inbetween++;
            }else if(random_nrs == 1){
                random_nrs_inbetween = 1;
            }
            random_nrs -= random_nrs_inbetween;
            generate_cluster<T>(&numbers_single, &all_numbers, hsize, pos, random_nrs_inbetween);
            next_position(pos, random_nrs_inbetween, free_space, hsize, random_nrs_inbetween + max_skip, seed + r_counter++);
        }
        generate_cluster<T>(&numbers_single, &all_numbers, hsize, pos, random_nrs);
    }
    // std::cout << numbers_collision.size() << "\t" << numbers_single.size() << std::endl;
    if(numbers_collision.size() + numbers_single.size() > distinct_values){
        return 0;
    }

    // for(T x: numbers_single){
    //     std::cout << x << std::endl;
    // }


    // for(T x: numbers_collision){
    //     std::cout << x << std::endl;
    // }

    generate_benchmark_data<T>(result, data_size, numbers_collision, numbers_single, seed+1 );    
    return numbers_collision.size() + numbers_single.size();
}



//only bad case
template<typename T>
size_t generate_data_v4(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t hsize,
    size_t (*hash_function)(T, size_t),
    size_t collision_chain_count,
    size_t collision_chain_length,
    size_t seed,
    size_t space,
    bool just_distinct_values = false // NOTE: this should only be used for datageneration testing
){

    std::multimap<size_t, T> all_numbers;   
    std::vector<T> numbers_collision;
    std::vector<T> numbers_single;

    size_t expected_hsize = p0_parameter_gen_hsize(collision_chain_count, collision_chain_length);

    if(expected_hsize > hsize){
        throw std::runtime_error("The configuration asks for to many numbers! increase distinct_values or decrease the collision parameters.");    
        return 0; // HSIZE is to small for the given configuration to fit.
    }
    if(seed == 0){
        throw std::runtime_error("The seed may not be zero!");
        return 0; // invalid seed
    }

    all_number_gen<T>(all_numbers, hash_function, hsize, collision_chain_length, seed+3);

    size_t pos = noise(hsize * distinct_values, seed) % hsize;
    size_t random_nrs = distinct_values - expected_hsize;
    size_t free_space = hsize - distinct_values;
    // std::cout << random_nrs << "\t" << free_space << std::endl;
    size_t skip_free;

    size_t random_nrs_inbetween;
    size_t r_counter = 0;


    //generate collision chains on different buckets and move the counter only one. 
    for(size_t i = 0; i < collision_chain_count; i++){
        generate_collision<T>(&numbers_collision, &all_numbers, hsize, pos, collision_chain_length);
        next_position(pos, space, hsize);
    }
    generate_cluster<T>(&numbers_single, &all_numbers, hsize, pos, random_nrs);

    // std::cout << numbers_collision.size() << "\t" << numbers_single.size() << std::endl;
    if(numbers_collision.size() + numbers_single.size() > distinct_values){
        return 0;
    }

    generate_benchmark_data<T>(result, data_size, numbers_collision, numbers_single, seed+1 );    
    return numbers_collision.size() + numbers_single.size();
}








/*
    generates data such that every x-th element is a value. random from these values a number of collisions get choosen such that. these values point to position 0.


*/
template<typename T>
size_t generate_data_v5(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t hsize,
    size_t (*hash_function)(T, size_t),
    size_t collision_size,
    size_t seed,
    bool just_distinct_values = false // NOTE: this should only be used for datageneration testing
){
    if(distinct_values > hsize){
        std::cout << "too many values for hash table\n";
        return 0;
    }
    if(collision_size >= distinct_values){
        std::cout << "too many to collide with the first value\n";
        return 0;
    }

    // if there are to many collisions to generate it is easier to just generate the once to stay aka the inverse. 
    bool inverse = collision_size > distinct_values * 0.5; 
    size_t to_generate_ids = (collision_size * !inverse) 
                            + ((distinct_values - 1 - collision_size) * inverse);

    std::cout << "inverse:\t" << inverse << std::endl;
    std::cout << "ids to gen:\t" << to_generate_ids << std::endl;

    std::vector<T> ids;
    for(size_t c = 0; c < to_generate_ids; c++){
        size_t nid = (noise(c, seed++) % (distinct_values - 1))+1;
        bool okay = true;
        for(size_t i: ids){
            okay == nid != i;
        }
        if(!okay){
            c--;
        }else{
            ids.push_back(nid);
        }
    }

    std::cout << "ids:\n";
    print_vector<T>(ids);

    std::vector<T> collision_ids;    
    for(size_t i = 1; i < distinct_values; i++){
        bool is_in = vector_contains<T>(&ids, i);
        bool insert = (is_in != inverse);
        if(insert){
            collision_ids.push_back(i);
        }
    }

    std::cout << "collision_ids:\n";
    print_vector<T>(collision_ids);

    std::multimap<size_t, T> all_numbers;
    std::vector<T> numbers;
    
    all_number_gen<T>(all_numbers, hash_function, hsize, collision_size +3, seed++);
    
    size_t start_pos = 0; // TODO change this to have a different starting position than 0.
    size_t save_pos = 0;
    typename std::multimap<size_t, T>::iterator itLow, itUp, it;

    itLow = all_numbers.lower_bound(start_pos);
    itUp = all_numbers.upper_bound(start_pos);
    itLow++;
    size_t c;
    for(it = itLow, c = 0; it != itUp && c < collision_size; it++, c++){
        numbers.push_back(it->second);   
    }

    std::cout << "positions:" << std::endl;
    double step_size = (hsize * 1.0) / distinct_values;
    double current_pos = start_pos;
    for(size_t i = 0; i < distinct_values; i++){
        bool is_in = vector_contains<T>(&collision_ids, i);
        if(!is_in){
            size_t pos = ((size_t)current_pos) % hsize;
            std::cout << pos << "\t";
            size_t value = all_numbers.lower_bound(pos)->second;
            numbers.push_back(value);
        }
        current_pos += step_size;
    }

    std::cout << std::endl;
    std::cout << "numbers:\n";
    print_vector<T>(numbers);

    generate_benchmark_data<T>(result, data_size, &numbers, seed++);
    return numbers.size();
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