#ifndef TUD_HASHING_TESTING_HASH_FUNCTIONS
#define TUD_HASHING_TESTING_HASH_FUNCTIONS

#include <stdlib.h>

#include "datagen.hpp"


size_t table[8][256];

enum HashFunction{
    MODULO,
    TABULATION,
    MULITPLY_SHIFT,
    MULTIPLY_ADD_SHIFT,
    MULTIPLY_PRIME,
    MURMUR
};

template<typename T>
using hash_fptr = size_t (*)(T, size_t);


void fill_tab_table(){
    size_t seed = 0xfa7343;
    for(size_t i = 0; i < 8; i++){
        for(size_t e = 0; e < 256; e++){
            table[i][e] = noise(e, seed + i) & 0xFFFFFFFF;
        }
    }
}

//---------------------------------------
// hash function
//---------------------------------------

// simple multiplicative hashing function
size_t hashx(uint32_t key, size_t HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

template<typename T>
size_t modulo(T key, size_t HSIZE) {
    size_t k = key;
    return k % HSIZE;
}

template<typename T>
size_t tab(T key, size_t HSIZE){
    uint32_t r = 0;
    for(size_t i = 0; i < sizeof(T); i++){
        r ^= table[i][(char)(key >> 8*i)];
    }
    return (uint32_t)(((uint64_t)(r) * (HSIZE)) >> 32);
}

template<typename T>
size_t multiply_prime(T x, size_t N){
    size_t a = 0x618FE02F;
    size_t b = 0x7FFFFFFE ^ a;
    size_t y = a * x + b;
    size_t p = ((uint64_t)(1) << 31) - 1;
    size_t z = (y&p) + (y>>31);
    if(z >= p){
        z -= p;
    }
    return (uint64_t)(z) * (uint64_t)(N) >> 31;
}

template<typename T>
size_t multiply_add_shift(T x, size_t N){
    size_t a = 0xE18FE02F;
    size_t b = 0xFFFFFFFE ^ a;
    size_t y = a * x + b;
    size_t z = (uint32_t)((y>>32) ^ (y));
    return (uint64_t)(z) * (uint64_t)(N) >> 32;
}

template<typename T>
size_t multiply_shift(T x, size_t N){
    size_t a = 0x4D7C6D4D;
    size_t y = a * x;
    uint32_t z = (uint32_t)((y>>32) ^ (y));
    return (uint64_t)(z) * (uint64_t)(N) >> 32;
}

// BASED ON: "A SevenDimensional Analysis of Hashing Methods and its Implications on Query Processing"'s 
// murmur3_64_finalizer
template<typename T>
size_t murmur(T key, size_t HSIZE){
    size_t k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccd;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53;
    k ^= k >> 33;
    return k % HSIZE;
}

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
        default:
            throw std::runtime_error("Unknown Hash Function");
    }
}

std::string get_hash_function_name(HashFunction x){
    switch(x){
        case HashFunction::MODULO:
            return "MODULO";
        case HashFunction::TABULATION:
            return "TABULATION";
        case HashFunction::MULITPLY_SHIFT:
            return "MULTIPLY_SHIFT";
        case HashFunction::MULTIPLY_ADD_SHIFT:
            return "MULTIPLY_SHIFT_ADD";
        case HashFunction::MULTIPLY_PRIME:
            return "MULTIPLY_PRIME";
        case HashFunction::MURMUR:
            return "MURMUR";
        default:
            throw std::runtime_error("Unknown Hash Function");
    }
}


#endif //TUD_HASHING_TESTING_HASH_FUNCTIONS