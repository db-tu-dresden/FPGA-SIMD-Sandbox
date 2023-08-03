#ifndef TUD_HASHING_TESTING_DATAGEN_HELP
#define TUD_HASHING_TESTING_DATAGEN_HELP

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>



// /// @brief a noise function that we use as a semi random function
// /// @param position 
// /// @param seed 
// /// @return a number that depends on both the position and the seed
size_t noise(size_t postion, size_t seed);


/// @brief checks if the numbers in each bucket are enough
/// @tparam T 
/// @param numbers 
/// @param HSIZE 
/// @param min_numbers 
/// @return true iff all buckets have more than min_number, false otherwise
template<typename T>
bool enough_values_per_bucket(std::multimap<size_t, T> &numbers, size_t HSIZE, size_t min_numbers);

/// @brief Checks if the given value is contained in the vector
/// @tparam T 
/// @param vec 
/// @param value 
/// @return true if the value is inside the vector false if not
template<typename T>
bool vector_contains(std::vector<T> *vec, T value);

/// @brief given the budget it calculates the next position
/// @param pos current position
/// @param budget how many free spaces are still available
/// @param HSIZE hash table size
/// @param seed the seed for the random number generator
void next_position(size_t &pos, size_t &budget, size_t HSIZE, size_t seed);
void next_position(size_t &pos, size_t move, size_t HSIZE);
void next_position(size_t &pos, size_t move, size_t &budget, size_t HSIZE, size_t max_move, size_t seed);

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
);

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
);

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
);

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
);

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
);

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
);

/// @brief Fills the result array with a random order of the numbers provided
/// @tparam T is the type of the result array
/// @param result an array that shall be filled
/// @param data_size 
/// @param numbers a vector that contains values that shall be inserted into the result
/// @param seed a seed that gets used by the random number generator
template<typename T>
void generate_benchmark_data(T*& result, size_t data_size, std::vector<T> *numbers, size_t seed);

/// @brief Fills the result array with a random order of the numbers provided
/// @tparam T is the type of the result array
/// @param result an array that shall be filled
/// @param data_size 
/// @param numbers a vector that contains values that shall be inserted into the result
/// @param seed a seed that gets used by the random number generator
template<typename T>
void generate_benchmark_data(T*& result, size_t data_size, std::vector<T> numbers_collision,  std::vector<T> numbers_cluster,size_t seed);

template<typename T>
void print_vector(std::vector<T> numbers);

#endif //TUD_HASHING_TESTING_DATAGEN_HELP