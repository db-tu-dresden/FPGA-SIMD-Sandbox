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

/// @brief tries to generate enough data for each bucket for the given hash function. THROWS error if the hash function can't generate a given number for a bucket
/// @tparam T what number gets generated
/// @param numbers result numbers associated to their bucket
/// @param hash_function the hash function to create the values
/// @param number_of_values how many values at least should be generated
/// @param HSIZE how big the hash size is (important for the hash function)
/// @param seed for the random number generator
/// @param min_numbers how many numbers should be atleast in one bucket.
template<typename T>
void generate_random_values(
    std::multimap<size_t, T> &numbers,
    size_t (*hash_function)(T, size_t),
    size_t number_of_values,
    size_t HSIZE,
    size_t seed,
    size_t min_numbers = 2
){
    size_t number_retry = 0;
    size_t retry = 0;
    size_t wanted_values = number_of_values;
    do{
        retry++;
        if(retry > 10 && !enough_values_per_bucket(numbers, HSIZE, 1)){    // second part tells us if every value got generated atleast once.
            throw std::runtime_error("bad hash function!");
        }else if(retry > ((min_numbers + 1) * 10) ){
            throw std::runtime_error("generation took to long!");
        }

        while(numbers.size() < wanted_values){
            T num;
            do{
                num = noise(numbers.size() + number_retry, seed);
                number_retry += num == 0;
            }while(num == 0);
            
            numbers.insert(std::pair<size_t, T>(hash_function(num, HSIZE), (T)(num)));
        }
        wanted_values += number_of_values;

    }while(!enough_values_per_bucket(numbers, HSIZE, min_numbers));
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
    for(size_t i = 0; i < data_size; i++){
        size_t ran_id = noise(i, seed) % numbers->size();
        result[i] = (T)(numbers->at(ran_id));
    }
}

#endif //TUD_HASHING_TESTING_DATAGEN_HELP