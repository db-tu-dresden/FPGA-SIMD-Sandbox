#ifndef TUD_HASHING_TESTING_HASH_FUNCTIONS
#define TUD_HASHING_TESTING_HASH_FUNCTIONS

#include <stdlib.h>
#include <string>
// #include "main/datagenerator/datagen_help.hpp"

enum HashFunction{
    MODULO,
    TABULATION,
    MULITPLY_SHIFT,
    MULTIPLY_ADD_SHIFT,
    MULTIPLY_PRIME,
    MURMUR,
    NOISE,
    SIP_HASH
};

template<typename T>
using hash_fptr = size_t (*)(T, size_t);


/// @brief a necessary function for tabulation hashing (tab). This function fills the table with random values
void fill_tab_table();

//transform val into [0, N) // mod possible but slow?!
inline size_t range_bit_based(uint32_t val, size_t N);

//transform val into [0, N) // mod possible but slow?!
size_t range_bit_based2(uint32_t val, size_t N);

template<size_t bits>
size_t rotate_left(const size_t v);

void compress(size_t& v0, size_t& v1, size_t& v2, size_t& v3, const size_t rounds);

//---------------------------------------
// hash function
//---------------------------------------


/// @brief based on highwayhash sip_hash. Instead of an array we use key and N and act like they were in an array
/// @tparam T 
/// @param key 
/// @param N 
/// @return [0, N)
template<typename T>
size_t sip_hash(T key, size_t N);

/// @brief legcy hash function. Replaced by multiply shift
/// @param key 
/// @param N 
/// @return 
template<typename T>
size_t hashx(uint32_t key, size_t N);

/// @brief random new hash function. SHOULD NOT BE USED!!
/// @param key 
/// @param N 
/// @return 
template<typename T> 
size_t noise_hash(T key, size_t N);

/// @brief modulo function
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t modulo(T key, size_t N);

/// @brief tabulation hashing function. Uses bytes from the key to get random values from a table and xors them together. after wards the value gets mapped on to a range using fast range
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N) 
template<typename T>
size_t tab(T key, size_t N);

/// @brief multiply prime hashing function. Multiplies the key with a prime number and uses fast range to reduce the value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t multiply_prime(T key, size_t N);

/// @brief multiply add shift hashing function. Multiplies the key with a prime number and uses fast range to reduce the value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t multiply_add_shift(T key, size_t N);

/// @brief multiply shift hashing function. like multiply shift but does not multiply by prime. Uses fast range to reduce the value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t multiply_shift(T key, size_t N);


// BASED ON: "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing"'s 
// murmur3_64_finalizer

/// @brief hashing function based on murmur3_64_finalizer as given in "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing". Uses fast range to reduce value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N) 
template<typename T>
size_t murmur(T key, size_t N);

/// @brief given the enum value it returns the wanted hash function
/// @tparam T the template parameter for the return value
/// @param x hash function enum specifying which to return
/// @return a function ptr to the hash function
template<typename T>
hash_fptr<T> get_hash_function(HashFunction x);

/// @brief returns a string with a unique string to identify the hash function
/// @param x hash function enum specifying for which hash function to return
/// @return the name of the function
std::string get_hash_function_name(HashFunction x);

#endif //TUD_HASHING_TESTING_LHASH_FUNCTIONS