#ifndef TUD_HASHING_TESTING_DATAGEN_HELP
#define TUD_HASHING_TESTING_DATAGEN_HELP

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>



/// @brief a noise function that we use as a semi random function
/// @param position 
/// @param seed 
/// @return a number that depends on both the position and the seed
uint64_t noise(size_t position, size_t seed){
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

/// @brief checks if the numbers in each bucket are enough
/// @tparam T 
/// @param numbers 
/// @param HSIZE 
/// @param min_numbers 
/// @return true iff all buckets have more than min_number, false otherwise
template<typename T>
bool enough_values_per_bucket(std::multimap<size_t, T> &numbers, size_t HSIZE, size_t min_numbers){
    for(size_t i = 0; i < HSIZE; i++){
        if(numbers.count(i) < min_numbers){
            return false;
        }
    }
    return true;
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
            num = noise(id, seed);// % 1000; //todo remove the mod 1000
            id++;
        }while(num == 0);
        
        numbers.insert(std::pair<size_t, T>(hash_function(num, different_values), (T)(num)));
    }
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
    std::multimap<size_t, T> &numbers,
    size_t (*hash_function)(T, size_t),
    size_t different_values,
    size_t collision_size,
    size_t seed
){
    if(different_values == 0){
        return;
    }

    size_t vals_per_bucket = collision_size + 1;
    if(vals_per_bucket < 2){
        vals_per_bucket = 2;
    }

    size_t mul = vals_per_bucket;
    if(mul > 32){
        mul = 32;
    }

    size_t number_of_values = different_values * mul;
    
    //kick of data generation as long as there isn't enough values per bucket
    size_t retry = 0;
    size_t ENOUGH_VALS = 2;
    const size_t max_retry = 100;
    do{
        if(ENOUGH_VALS == 2 && retry > 10){
            ENOUGH_VALS = enough_values_per_bucket(numbers, different_values, 1);
        }else if(ENOUGH_VALS == 0){
            throw std::runtime_error("bad hash function! Not all hash values where able to be generated!");
        }else if(retry > max_retry){
            throw std::runtime_error("generation is takeing to long. abort!");
        }

        generate_random_values(numbers, hash_function, different_values, number_of_values, seed + retry);
        retry++;
    }while(!enough_values_per_bucket(numbers, different_values, vals_per_bucket));
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
void generate_benchmark_data(T*& result, size_t data_size, std::vector<T> numbers_collision,  std::vector<T> numbers_cluster,size_t seed){
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


#endif //TUD_HASHING_TESTING_DATAGEN_HELP