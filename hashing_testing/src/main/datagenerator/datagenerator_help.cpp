#include <iostream>
#include <stdexcept>

#include "main/datagenerator/datagen_help.hpp"
#include "main/hash_function.hpp"

#include <limits>

template<typename T> 
bool enough_values_per_bucket(std::multimap<size_t, T> &numbers, size_t HSIZE, size_t min_numbers){
    for(size_t i = 0; i < HSIZE; i++){
        if(numbers.count(i) < min_numbers){
            return false;
        }
    }
    return true;
}
void print_count(size_t *numbers, size_t HSIZE){
    for(size_t i = 0; i < HSIZE; i++){
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
}

bool enough_values_per_bucket(size_t *numbers, size_t HSIZE, size_t min_numbers){
    for(size_t i = 0; i < HSIZE; i++){
        if(numbers[i] < min_numbers){
            return false;
        }
    }
    return true;
}

template<typename T> 
bool count_reset(std::multimap<size_t, T> &all_numbers, size_t * numbers, size_t HSIZE){
    // print_count(numbers, HSIZE);
    
    for(size_t i = 0; i < HSIZE; i++){
        numbers[i] = all_numbers.count(i);
    }
    // print_count(numbers, HSIZE);
    return false;//enough_values_per_bucket(numbers, HSIZE, min_numbers);
}


size_t noise(size_t position, size_t seed){
    size_t BIT_NOISE1 = 0x68E31DA4;
    size_t BIT_NOISE2 = 0xB5297A4D;
    size_t BIT_NOISE3 = 0x1B56C4E9;

    uint64_t mangled = position;
    mangled *= BIT_NOISE1;
    mangled += seed;
    mangled ^= (mangled << 13);
    mangled += BIT_NOISE2;
    mangled ^= (mangled >> 7);
    mangled *= BIT_NOISE3;
    mangled ^= (mangled << 17);
    return mangled;
}

/// @brief Checks if the given value is contained in the vector
/// @tparam T 
/// @param vec 
/// @param value 
/// @return true if the value is inside the vector false if not
template<typename T>
bool vector_contains(std::vector<T> *vec, T value){
    for(T val: *vec){
        if(value == val){
            return true;
        }
    }
    return false;
}
/// @brief Checks if the given value is contained in the vector
/// @tparam T 
/// @param vec 
/// @param value 
/// @return true if the value is inside the vector false if not

/// @brief given the budget it calculates the next position
/// @param pos current position
/// @param budget how many free spaces are still available
/// @param HSIZE hash table size
/// @param seed the seed for the random number generator
void next_position(size_t &pos, size_t &budget, size_t HSIZE, size_t seed){
    if(budget == 0){
        budget = 1;
    }
    if(budget == 1){
        pos = (pos + 1) % HSIZE;
        return;
    }
    
    size_t offset = noise(budget + pos, seed) % budget;
    budget -= offset;
    pos = (pos + 1 + offset) % HSIZE;
}

void next_position(size_t &pos, size_t move, size_t HSIZE){
    pos = (pos + move + 1) % HSIZE;
}

void next_position(size_t &pos, size_t move, size_t &budget, size_t HSIZE, size_t max_move, size_t seed){
    size_t skip = 0;
    int64_t max_budget_move = max_move - move;
    if(budget > 0 && max_budget_move > 0){
        skip = noise(pos + move + budget, seed);
        skip %= budget;
        skip %= max_budget_move;
        budget -= skip;
    }
    if(move > max_move){
        move = max_move;
    }
    pos = pos + move + skip;
    pos %= HSIZE;
}

/// @brief tries to generate enough data for each bucket for the given hash function. THROWS error if the hash function can't generate a given number for a bucket
/// @tparam T what number gets generated
/// @param numbers result numbers associated to their bucket
/// @param hash_function the hash function to create the values
/// @param number_of_values how many values at least should be generated
/// @param different_values how big the hash size is (important for the hash function)
/// @param seed for the random number generator

template<typename T>
void generate_random_values(
    std::multimap<size_t, T> &numbers,
    size_t (*hash_function)(T, size_t),
    size_t different_values,
    size_t number_of_values,
    size_t seed
){
    size_t wanted_values = numbers.size() + number_of_values;
    size_t id = 0;

    while(numbers.size() < wanted_values){
        T num;
        //keep generating until num is not 0
        do{
            num = noise(id, seed);
            id++;
        }while(num == 0);
        
        numbers.insert(std::pair<size_t, T>(hash_function(num, different_values), (T)(num)));
    }
}



template<typename T>
bool generate_random_values(
    std::multimap<size_t, T> &numbers,
    size_t * nr_count,
    size_t (*hash_function)(T, size_t),
    size_t different_values,
    size_t number_of_values,
    size_t seed,
    size_t start_sequence
){
    size_t generated = 0;
    size_t wanted_values = numbers.size() + number_of_values;
    size_t id = 0;
    T num;
    if(start_sequence == 0){
        num = noise(0x837A, seed);
    }else{
        num = start_sequence;
    }

    while(numbers.size() < wanted_values){
        //keep generating until num is not 0
        do{
            num++;
        }while(num == 0);
        size_t hash = hash_function(num, different_values);
        if(nr_count[hash] <= number_of_values + 4){
            numbers.insert(std::pair<size_t, T>(hash, (T)(num)));
            nr_count[hash]++;
            generated++;
        }
    }
    return generated >= 0;
}

template<typename T> 
void malloc_all_numbers(
    T**& numbers,
    size_t bucket_count,
    size_t bucket_size
){
    if(numbers == nullptr){
        numbers = (T **) malloc(bucket_count * sizeof(T*));
        for(size_t i = 0; i < bucket_count; i++){
            numbers[i] = (T *) malloc((bucket_size + 2) * sizeof(T));
        }
        for(size_t i = 0; i < bucket_count; i++){
            numbers[i][0] = 0;
        }
    }
}

template<typename T> 
void free_all_numbers(
    T**& numbers,
    size_t bucket_count
){
    if(numbers != nullptr){
        for(size_t i = 0; i < bucket_count; i++){
        free(numbers[i]);
        }
        free(numbers);
        numbers = nullptr;
    }
}

template<typename T>
void print(T** n, size_t bc){
    std::cout << "----------------------" << bc<< std::endl;
    for(size_t i = 0; i < bc; i++){
        for(size_t e = 0; e <= n[i][0]; e++){
            std::cout << n[i][e] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------" << std::endl;
}


template<typename T> 
bool add_number(
    T**& numbers,
    size_t bucket_count,
    size_t bucket_size,
    size_t bucket,
    T value,
    bool force
){
    // std::cout << "\nadd:\t(" << bucket << "," << value << ")" << std::flush;
    bool insert = bucket < bucket_count;
    // std::cout << "\t" << insert << std::flush;
    if(!force){
        // std::cout << "\tnf";
        for(size_t i = 1; i <= numbers[bucket][0] && insert; i++){
            insert = numbers[bucket][i] != value;
        }
    }
    // std::cout << "\t" << insert << std::flush;
    insert = insert && numbers[bucket][0] < bucket_size;
    // std::cout << "\t" << insert << std::endl;
    if(insert){
        size_t pos = numbers[bucket][0] + 1;
        numbers[bucket][pos] = value;
        numbers[bucket][0]++;
    } 
    // print(numbers, bucket_count);
    return insert;
}

// bool unsave_add_number(
//     T**& numbers,
//     size_t bucket_count,
//     size_t bucket_size, 
//     size_t bucket,
//     T value
// ){
//     bool insert = bucket < bucket_count;
//     insert = insert && numbers[bucket][0] < bucket_size;
//     if(insert){
//         size_t pos = numbers[bucket][0] + 1;
//         numbers[bucket][pos] = value;
//         numbers[bucket][0]++;
//     }  
//     return insert;
// }

template<typename T> 
size_t generate_numbers(
    T **& numbers,
    size_t bucket_count,
    size_t bucket_size,
    size_t (*hash_function)(T, size_t),
    size_t to_gen,
    size_t pos
){
    size_t inserted = 0;
    T value = pos;
    for(size_t i = 0; i < to_gen; i++){
        value += value == 0;
        size_t bucket = hash_function(value, bucket_count);
        inserted += add_number(numbers, bucket_count, bucket_size, bucket, value);
        value ++;
        // print(numbers, bucket_count);
    }
    return inserted;
}

/// @brief 
/// @tparam T 
/// @param numbers 
/// @param hash_function 
/// @param different_values 
/// @param collision_size 
/// @param seed 
template<typename T>
void all_number_gen(
    T **& numbers,
    size_t (*hash_function)(T, size_t),
    size_t different_values,
    size_t collision_size,
    size_t seed
){
    if(different_values < 1){
        different_values = 1;
    }
    if(collision_size < 1){
        collision_size = 1;
    }
    T pos = (T)(noise(0, seed));
    
    size_t total_max_values = (std::numeric_limits<T>::max() - 1) - std::numeric_limits<T>::min();
    size_t max_wanted_values = different_values * collision_size;
    size_t free_values = total_max_values - max_wanted_values;
    
    malloc_all_numbers(numbers, different_values, collision_size);
    size_t tries = 0;
    size_t max_tries = 100;
    size_t min_steps = (max_wanted_values / 5) + 1;
    // if(min_steps < different_values){
    //     min_steps = different_values;
    // }
    size_t count = 0;
    do{
        count = 0;
        size_t generated = generate_numbers(numbers, different_values, collision_size, hash_function, min_steps, pos);


        size_t skip_step = 0;
        if(generated > 0 && free_values > 1){
            skip_step = noise(pos, seed) % free_values;
        }

        pos += min_steps;
        pos += skip_step;
        free_values -= skip_step;
        
        for(size_t i = 0; i < different_values; i++){
            count += numbers[i][0];
        }
        std::cout << "." << std::flush;
        // print(numbers, different_values);
    }while(count < max_wanted_values && tries++ < max_tries);
    std::cout  << "\t" << max_wanted_values << " " << count << " " << (max_wanted_values <= count) << ".\t"; 
    // print(numbers, different_values);
}  


/// @brief 
/// @tparam T 
/// @param numbers 
/// @param hash_function 
/// @param different_values 
/// @param collision_size 
/// @param seed 
template<typename T>
void all_number_gen_o(
    std::multimap<size_t, T> &numbers,
    size_t (*hash_function)(T, size_t),
    size_t different_values,
    size_t collision_size,
    size_t seed
){

    if(different_values == 0){
        std::cout << "quickwayout\n";
        return;
    }

    size_t step = collision_size;
    if(step < 2){
        step = 2;
    }
    step *= 2;
    step--;
    size_t min_nr_vals = step * different_values;
    const size_t max_retry = different_values*2;


    size_t * nr_count = (size_t *) malloc(different_values * sizeof(size_t));
    
    count_reset<T>(numbers, nr_count, different_values);
    bool generated = false;
    // std::cout << "generation\n";
    size_t sequence = noise(0x837A, seed);
    generate_random_values(numbers, nr_count, hash_function, 
        different_values, min_nr_vals, seed, sequence);
    sequence = (sequence + min_nr_vals) % different_values;
    count_reset<T>(numbers, nr_count, different_values);
    
    size_t retry = 0;
    size_t ENOUGH_VALS = 2;
    while(!enough_values_per_bucket(nr_count, different_values, collision_size) 
        || (!generated && !count_reset<T>(numbers, nr_count, different_values)))
    {
        if(ENOUGH_VALS == 2 && retry > max_retry/3){
            ENOUGH_VALS = enough_values_per_bucket(nr_count, different_values, 1);
        }else if(ENOUGH_VALS == 0){
            throw std::runtime_error("bad hash function! Not all hash values where able to be generated!");
        }else if(retry > max_retry){
            throw std::runtime_error("generation is takeing to long. abort!");
        }
        // std::cout << "generation:\t" << retry << std::endl;
        generated = generate_random_values(numbers, nr_count, hash_function, 
            different_values, step, seed + ++retry, sequence);
        sequence = (sequence + step) % different_values;
    }
    free(nr_count);    
    // print_count(nr_count, different_values);
}   

/// @brief generates one collision at the starting position with the given collision length. if the length is to big a neighboring bucket might get used to help fill the collision
/// @tparam T the data type of the numbers
/// @param result contains all random numbers. The new collision will be added into it
/// @param numbers multimap containg random numbers based on their hash
/// @param HSIZE the hash table size
/// @param start_pos where the collision shall start
/// @param collision_lenght how long the collision shall be
template<typename T>
void generate_collision(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t start_pos,
    size_t collision_lenght
){
    if(collision_lenght > HSIZE || HSIZE == 0){
        return;
    }

    size_t pos = start_pos % HSIZE;
    size_t data_to_generate = collision_lenght;
    
    bool GENERATE = data_to_generate > 0;
    
    typename std::multimap<size_t, T>::iterator itLow, itUp, it;
    while(GENERATE){
        itLow = numbers->lower_bound(pos);
        itUp = numbers->upper_bound(pos);

        for(it = itLow; it != itUp && GENERATE; it++){
            T n_val = it->second;
            bool contains = vector_contains<T>(result, n_val);
            if(!contains){
                result->push_back(it->second);
                GENERATE = --data_to_generate > 0;
            }
        }
        pos = (pos + 1) % HSIZE;
    }
}

/// @brief generates one collision at the starting position with the given collision length. if the length is to big a neighboring bucket might get used to help fill the collision for the the soaov approach
/// @tparam T the data type of the numbers
/// @param result contains all random numbers. The new collision will be added into it
/// @param numbers multimap containg random numbers based on their hash
/// @param HSIZE the hash table size
/// @param h_pos part of the starting position. The vector element 
/// @param e_pos part of the starting position. The element inside of the vector
/// @param elements the number of elements inside the one vector
/// @param collision_lenght how long the collision shall be
template<typename T>
void generate_collision_soaov(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t &h_pos,
    size_t &e_pos,
    size_t elements,
    size_t collision_lenght
){
    if(collision_lenght > (HSIZE * elements) || (HSIZE * elements) == 0){
        return;
    }

    size_t pos = h_pos % HSIZE;
    size_t data_to_generate = collision_lenght;
    
    bool GENERATE = data_to_generate > 0;
    
    typename std::multimap<size_t, T>::iterator itLow, itUp, it;
    while(GENERATE){
        itLow = numbers->lower_bound(pos);
        itUp = numbers->upper_bound(pos);

        for(it = itLow; it != itUp && GENERATE; it++){
            T n_val = it->second;
            bool CONTAINS = vector_contains<T>(result, n_val);
            
            if(!CONTAINS){
                result->push_back(n_val);
                GENERATE = --data_to_generate > 0;
                e_pos = (e_pos + 1) % elements;
                h_pos = (h_pos + (e_pos == 0)) % HSIZE;
            }
        }
        pos = (pos + 1) % HSIZE;
    }
}

/// @brief generates one cluster at the starting position with the cluster length for the the soaov approach
/// @param start_pos where the cluster shall start
/// @tparam T the data type of the numbers
/// @param result contains all random numbers. The new cluster will be added into it
/// @param numbers multimap containg random numbers based on their hash
/// @param HSIZE the hash table size
/// @param h_pos part of the starting position. The vector element 
/// @param e_pos part of the starting position. The element inside of the vector
/// @param elements the number of elements inside the one vector
/// @param cluster_lenght how long the cluster shall be
template<typename T>
void generate_cluster_soaov(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t &h_pos,
    size_t &e_pos,
    size_t elements,
    size_t cluster_lenght
){
    if(cluster_lenght > (HSIZE * elements) || (HSIZE * elements) == 0){
        return;
    }

    typename std::multimap<size_t, T>::iterator itLow, itUp, it;
    itLow = numbers->lower_bound(h_pos);
    itUp = numbers->upper_bound(h_pos);
    it = itLow;
    size_t i = 0;
    while(i < cluster_lenght){
        T n_val = it->second;
        bool CONTAINS = vector_contains<T>(result, n_val);
        if(!CONTAINS){
            result->push_back(n_val);
            e_pos = (e_pos + 1) % elements;
            h_pos = (h_pos + (e_pos == 0)) % HSIZE;
            i++;
        }

        if(e_pos == 0 && !CONTAINS){
            itLow = numbers->lower_bound(h_pos);
            itUp = numbers->lower_bound(h_pos);
            it = itLow;
        }else{
            it++;
        }
    }
}

/// @brief generates one cluster at the starting position with the cluster length
/// @tparam T the data type of the numbers
/// @param result contains all random numbers. The new cluster will be added into it
/// @param numbers multimap containg random numbers based on their hash
/// @param HSIZE the hash table size
/// @param start_pos where the cluster shall start
/// @param cluster_lenght how long the cluster shall be
template<typename T>
void generate_cluster(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t start_pos,
    size_t cluster_lenght
){
    if(cluster_lenght > HSIZE || HSIZE == 0){
        return;
    }

    size_t pos = start_pos % HSIZE;
    for(size_t i = 0; i < cluster_lenght; i++){
        typename std::multimap<size_t, T>::iterator it;
        it = numbers->find(pos);
        result->push_back(it->second);
        pos = (pos + 1) % HSIZE;
    }
}

/// @brief Fills the result array with a random order of the numbers provided
/// @tparam T is the type of the result array
/// @param result an array that shall be filled
/// @param data_size 
/// @param numbers a vector that contains values that shall be inserted into the result
/// @param seed a seed that gets used by the random number generator
template<typename T>
void generate_benchmark_data(T*& result, size_t data_size, std::vector<T> *numbers, size_t seed){
    size_t count[numbers->size()];
    size_t nr_values_per_bucket = (data_size / numbers->size()) + 1;
    for(size_t i = 0; i < numbers->size(); i++){
        count[i] = nr_values_per_bucket;
    }
    
    for(size_t i = 0; i < data_size; i++){
        size_t ran_id = noise(i, seed);
        bool placed = false;

        do{
            ran_id %= numbers->size();
            if(count[ran_id] > 0){
                result[i] = (T)(numbers->at(ran_id));
                count[ran_id]--;
                placed = true;
            }
            ran_id++;
        }while(!placed);
    }
}

/// @brief Fills the result array with a random order of the numbers provided
/// @tparam T is the type of the result array
/// @param result an array that shall be filled
/// @param data_size 
/// @param numbers a vector that contains values that shall be inserted into the result
/// @param seed a seed that gets used by the random number generator
template<typename T>
void generate_benchmark_data(T*& result, size_t data_size, std::vector<T> numbers_collision, std::vector<T> numbers_cluster, size_t seed){
    std::vector<T> numbers;
    size_t val_count = numbers_collision.size() + numbers_cluster.size();
    size_t count[val_count];

    size_t nr_values_per_bucket = (data_size / val_count) + 1;

    size_t data_gen_i = 0;
    size_t count_i = 0;

    for(T x: numbers_cluster){
        result[data_gen_i++] = x;
        count[count_i++] = nr_values_per_bucket - 1;
        numbers.push_back(x);
    }

    for(T x: numbers_collision){
        count[count_i++] = nr_values_per_bucket;
        numbers.push_back(x);
    }

    for(size_t i = data_gen_i; i < data_size; i++){
        size_t ran_id = noise(i, seed);
        bool placed = false;

        do{
            ran_id %= numbers.size();
            if(count[ran_id] > 0){
                result[i] = (T)(numbers.at(ran_id));
                count[ran_id]--;
                placed = true;
            }
            ran_id++;
        }while(!placed);
    }
}


template void all_number_gen<uint64_t>(uint64_t**& numbers, size_t (*hash_function)(uint64_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<uint32_t>(uint32_t**& numbers, size_t (*hash_function)(uint32_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<uint16_t>(uint16_t**& numbers, size_t (*hash_function)(uint16_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<uint8_t>(uint8_t**& numbers, size_t (*hash_function)(uint8_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<int64_t>(int64_t**& numbers, size_t (*hash_function)(int64_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<int32_t>(int32_t**& numbers, size_t (*hash_function)(int32_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<int16_t>(int16_t**& numbers, size_t (*hash_function)(int16_t, size_t), size_t dv, size_t cs, size_t seed);
template void all_number_gen<int8_t>(int8_t**& numbers, size_t (*hash_function)(int8_t, size_t), size_t dv, size_t cs, size_t seed);

template void free_all_numbers<uint64_t>(uint64_t**&numbers, size_t bc);
template void free_all_numbers<uint32_t>(uint32_t**&numbers, size_t bc);
template void free_all_numbers<uint16_t>(uint16_t**&numbers, size_t bc);
template void free_all_numbers<uint8_t>(uint8_t**&numbers, size_t bc);
template void free_all_numbers<int64_t>(int64_t**&numbers, size_t bc);
template void free_all_numbers<int32_t>(int32_t**&numbers, size_t bc);
template void free_all_numbers<int16_t>(int16_t**&numbers, size_t bc);
template void free_all_numbers<int8_t>(int8_t**&numbers, size_t bc);

template void malloc_all_numbers<uint64_t>(uint64_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<uint32_t>(uint32_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<uint16_t>(uint16_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<uint8_t>(uint8_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<int64_t>(int64_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<int32_t>(int32_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<int16_t>(int16_t**&numbers, size_t bc, size_t bs);
template void malloc_all_numbers<int8_t>(int8_t**&numbers, size_t bc, size_t bs);

template bool add_number<uint64_t>(uint64_t**&numbers, size_t bc, size_t bs, size_t b, uint64_t v, bool force);
template bool add_number<uint32_t>(uint32_t**&numbers, size_t bc, size_t bs, size_t b, uint32_t v, bool force);
template bool add_number<uint16_t>(uint16_t**&numbers, size_t bc, size_t bs, size_t b, uint16_t v, bool force);
template bool add_number<uint8_t>(uint8_t**&numbers, size_t bc, size_t bs, size_t b, uint8_t v, bool force);
template bool add_number<int64_t>(int64_t**&numbers, size_t bc, size_t bs, size_t b, int64_t v, bool force);
template bool add_number<int32_t>(int32_t**&numbers, size_t bc, size_t bs, size_t b, int32_t v, bool force);
template bool add_number<int16_t>(int16_t**&numbers, size_t bc, size_t bs, size_t b, int16_t v, bool force);
template bool add_number<int8_t>(int8_t**&numbers, size_t bc, size_t bs, size_t b, int8_t v, bool force);

template void print<uint64_t>(uint64_t** n, size_t bc);
template void print<int64_t>(int64_t** n, size_t bc);
template void print<uint32_t>(uint32_t** n, size_t bc);
template void print<int32_t>(int32_t** n, size_t bc);
template void print<uint16_t>(uint16_t** n, size_t bc);
template void print<int16_t>(int16_t** n, size_t bc);
template void print<uint8_t>(uint8_t** n, size_t bc);
template void print<int8_t>(int8_t** n, size_t bc);

// template void all_number_gen_o<uint64_t>(std::multimap<size_t, uint64_t> &numbers, size_t (*hash_function)(uint64_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<uint32_t>(std::multimap<size_t, uint32_t> &numbers, size_t (*hash_function)(uint32_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<uint16_t>(std::multimap<size_t, uint16_t> &numbers, size_t (*hash_function)(uint16_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<uint8_t>(std::multimap<size_t, uint8_t> &numbers, size_t (*hash_function)(uint8_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<int64_t>(std::multimap<size_t, int64_t> &numbers, size_t (*hash_function)(int64_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<int32_t>(std::multimap<size_t, int32_t> &numbers, size_t (*hash_function)(int32_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<int16_t>(std::multimap<size_t, int16_t> &numbers, size_t (*hash_function)(int16_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);
// template void all_number_gen_o<int8_t>(std::multimap<size_t, int8_t> &numbers, size_t (*hash_function)(int8_t, size_t),
//     size_t different_values, size_t collision_size, size_t seed);