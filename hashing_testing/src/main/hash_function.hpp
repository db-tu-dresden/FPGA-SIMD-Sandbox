#ifndef TUD_HASHING_TESTING_HASH_FUNCTIONS
#define TUD_HASHING_TESTING_HASH_FUNCTIONS

#include <stdlib.h>

#include "datagenerator/datagen_help.hpp"


size_t table[8][256];

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
void fill_tab_table(){
    size_t seed = 0xfa7343;
    for(size_t i = 0; i < 8; i++){
        for(size_t e = 0; e < 256; e++){
            table[i][e] = noise(e, seed + i) & 0xFFFFFFFF;
        }
    }
}

//transform val into [0, N) // mod possible but slow?!
inline size_t range_bit_based(uint32_t val, size_t N){
    uint64_t a = val;
    uint64_t b = N;

    size_t bit = 32 - __builtin_clz(N);
    uint64_t c = (a >> bit) * b;
    return (c) >> (32 - bit);
}

//transform val into [0, N) // mod possible but slow?!
size_t range_bit_based2(uint32_t val, size_t N){
    uint64_t a = val;
    uint64_t b = N;

    size_t bit = 32 - __builtin_clz(N) + 4;
    uint64_t c = (a & (0xFFFFFFFF >> (32 - bit))) * b;
    return (c) >> (bit);
}

template<size_t bits>
size_t rotate_left(const size_t v){
    const size_t left = v << bits;
    const size_t right = v >> (64-bits);
    return left | right;
}

void compress(size_t& v0, size_t& v1, size_t& v2, size_t& v3, const size_t rounds){
    for(size_t i = 0; i < rounds; i++){
        v0 += v1;
        v2 += v3;
        v1 = rotate_left<13>(v1);
        v3 = rotate_left<16>(v3);
        v1 ^= v0;
        v3 ^= v2;

        v0 = rotate_left<32>(v0);

        v2 += v1;
        v0 += v3;
        v1 = rotate_left<17>(v1);
        v3 = rotate_left<21>(v3);
        v1 ^= v2;
        v3 ^= v0;

        v2 = rotate_left<32>(v2);
    }
}

//---------------------------------------
// hash function
//---------------------------------------

size_t _k_update_rounds = 2;
size_t _k_finalize_rounds = 2;

/// @brief based on highwayhash sip_hash. Instead of an array we use key and N and act like they were in an array
/// @tparam T 
/// @param key 
/// @param N 
/// @return [0, N)
template<typename T>
size_t sip_hash(T key, size_t N){
    size_t v0 = 0x736f6d6570736575ull;
    size_t v1 = 0x646f72616e646f6dull;
    size_t v2 = 0x6c7967656e657261ull;
    size_t v3 = 0x7465646279746573ull;

    //packet 1 with the key
    v3 ^= key;
    compress(v0, v1, v2, v3, _k_update_rounds);
    v0 ^= key;

    //packet 2 with N
    v3 ^= N;
    compress(v0, v1, v2, v3, _k_update_rounds);    
    v0 ^= N;

    //FINALIZE (we don't really need this but to stay true to SIP HASH we do it too)
    v2 ^= 0xFF;

    compress(v0, v1, v2, v3, _k_finalize_rounds);

    size_t val = (v0 ^ v1) ^ (v2 ^ v3);
    return range_bit_based2(val, N);
}

/// @brief legcy hash function. Replaced by multiply shift
/// @param key 
/// @param N 
/// @return 
size_t hashx(uint32_t key, size_t N) {
    return ((unsigned long)((unsigned int)1300000077*key)* N)>>32;
}

/// @brief random new hash function. SHOULD NOT BE USED!!
/// @param key 
/// @param N 
/// @return 
size_t noise_hash(uint32_t key, size_t N) {
    size_t BIT_NOISE1 = 0x300a8352005996ae;
    size_t BIT_NOISE2 = 0x512eb6f10ed4909d;
    size_t BIT_NOISE3 = 0xae2008421fd52b1f;
    
    // size_t BIT_NOISE1 = 0x68E31DA4;
    // size_t BIT_NOISE2 = 0xB5297A4D;
    // size_t BIT_NOISE3 = 0x1B56C4E9;


    uint64_t mangled = key;
    

    mangled *= BIT_NOISE1;
    mangled += N;
    // mangled += noise_hash(N, N, 0);
    mangled ^= (mangled << 13);
    mangled += BIT_NOISE2;
    mangled ^= (mangled >> 7);
    mangled *= BIT_NOISE3;
    mangled ^= (mangled << 17);
    // size_t _bit = 11;
    return range_bit_based2(mangled, N);
    // return mangled % N;
}

/// @brief modulo function
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t modulo(T key, size_t N) {
    size_t k = key;
    return k % N;
}

/// @brief tabulation hashing function. Uses bytes from the key to get random values from a table and xors them together. after wards the value gets mapped on to a range using fast range
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N) 
template<typename T>
size_t tab(T key, size_t N){
    uint32_t r = 0;
    for(size_t i = 0; i < sizeof(T); i++){
        r ^= table[i][(char)(key >> 8*i)];
    }
    return (uint32_t)(((uint64_t)(r) * (N)) >> 32);
}

/// @brief multiply prime hashing function. Multiplies the key with a prime number and uses fast range to reduce the value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t multiply_prime(T key, size_t N){
    size_t a = 0x618FE02F;
    size_t b = 0x7FFFFFFE ^ a;
    size_t y = a * key + b;
    size_t p = ((uint64_t)(1) << 31) - 1;
    size_t z = (y&p) + (y>>31);
    if(z >= p){
        z -= p;
    }
    return (uint64_t)(z) * (uint64_t)(N) >> 31;
}

/// @brief multiply add shift hashing function. Multiplies the key with a prime number and uses fast range to reduce the value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t multiply_add_shift(T key, size_t N){
    // size_t a = 0xE18FE02F;
    // size_t b = 0xFFFFFFFE ^ a;
    size_t a = 0xE18FA2826236E02F;
    size_t b = 0x1E705D7D9DC91FD1;

    size_t r = key;
    // r += N;
    r *= a;
    r += b;
    // r *= a;
    // return r % N;

    // size_t z = (uint32_t)((r>>32) ^ (y));
    // return (uint64_t)(z) * (uint64_t)(N) >> 32;
    return range_bit_based2(r, N);
}

/// @brief multiply shift hashing function. like multiply shift but does not multiply by prime. Uses fast range to reduce the value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N)
template<typename T>
size_t multiply_shift(T key, size_t N){
    size_t a = 0x4D7C6D4D;
    a ^= N;
    size_t y = a * key;
    // uint32_t z = (uint32_t)((y>>32) ^ (y));
    // return (uint64_t)(z) * (uint64_t)(N) >> 32;
    return range_bit_based2(y, N);
}


// BASED ON: "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing"'s 
// murmur3_64_finalizer

/// @brief hashing function based on murmur3_64_finalizer as given in "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing". Uses fast range to reduce value
/// @tparam T gives the type the key has
/// @param key the key that shall be maped on the range
/// @param N the upper bound of the range
/// @return a value in [0, N) 
template<typename T>
size_t murmur(T key, size_t N){
    size_t k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    // k &= 0xFFFFFFFF;
   
    return range_bit_based2(k, N);
    // return (uint64_t)(k) * (uint64_t)(N) >> 32;
}

/// @brief given the enum value it returns the wanted hash function
/// @tparam T the template parameter for the return value
/// @param x hash function enum specifying which to return
/// @return a function ptr to the hash function
template<typename T>
hash_fptr<T> get_hash_function(HashFunction x){
    switch(x){
        case HashFunction::MODULO:
            return &modulo;
        case HashFunction::TABULATION:
            return &tab;
        case HashFunction::MULITPLY_SHIFT:
            return &multiply_shift;
        case HashFunction::MULTIPLY_ADD_SHIFT:
            return &multiply_add_shift;
        case HashFunction::MULTIPLY_PRIME:
            return &multiply_prime;
        case HashFunction::MURMUR:
            return &murmur;
        case HashFunction::NOISE:
            return &noise_hash;
        case HashFunction::SIP_HASH:
            return &sip_hash;
        default:
            throw std::runtime_error("Unknown Hash Function");
    }
}

/// @brief returns a string with a unique string to identify the hash function
/// @param x hash function enum specifying for which hash function to return
/// @return the name of the function
std::string get_hash_function_name(HashFunction x){
    switch(x){
        case HashFunction::MODULO:
            return "MODULO";
        case HashFunction::TABULATION:
            return "TABULATION";
        case HashFunction::MULITPLY_SHIFT:
            return "MULTIPLY_SHIFT";
        case HashFunction::MULTIPLY_ADD_SHIFT:
            return "MULTIPLY_ADD_SHIFT";
        case HashFunction::MULTIPLY_PRIME:
            return "MULTIPLY_PRIME";
        case HashFunction::MURMUR:
            return "MURMUR";
        case HashFunction::NOISE:
            return "NOISE";
        case HashFunction::SIP_HASH:
            return "SIP_HASH";
        default:
            throw std::runtime_error("Unknown Hash Function");
    }
}


#endif //TUD_HASHING_TESTING_HASH_FUNCTIONS