#include <stdexcept>
#include "main/hash_function.hpp"

size_t table[8][256];

//values for sip_hash
size_t _k_update_rounds = 2;
size_t _k_finalize_rounds = 2;

size_t xor_shift(size_t val){
    size_t x64 = val ^ (val << 13); 
    x64 ^= x64 >> 7;
    x64 ^= x64 << 17;
    return x64;
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

size_t f_noise(size_t position, size_t seed){
    size_t mangled = xor_shift(position);
    mangled ^= xor_shift(seed);
    return mangled;
}

void fill_tab_table(){
    size_t seed = 0xfa7343;
    for(size_t i = 0; i < 8; i++){
        for(size_t e = 0; e < 256; e++){
            table[i][e] = noise_hash<size_t>(e, seed + i) & 0xFFFFFFFF;
        }
    }
}

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

//works for anything 32 and smaller.
template<typename T>
size_t hashx(T key, size_t N) {
    return ((unsigned long)((unsigned int)1300000077*(uint32_t)(key))* N)>>32;
}

template<typename T>
size_t noise_hash(T key, size_t N) {
    size_t BIT_NOISE1 = 0x300a8352005996ae;
    size_t BIT_NOISE2 = 0x512eb6f10ed4909d;
    size_t BIT_NOISE3 = 0xae2008421fd52b1f;

    uint64_t mangled = key;
    

    mangled *= BIT_NOISE1;
    mangled += N;

    mangled ^= (mangled << 13);
    mangled += BIT_NOISE2;
    mangled ^= (mangled >> 7);
    mangled *= BIT_NOISE3;
    mangled ^= (mangled << 17);
    
    return range_bit_based2(mangled, N);
}

template<typename T>
size_t modulo(T key, size_t N) {
    size_t k = key;
    return k % N;
}

template<typename T>
size_t tab(T key, size_t N){
    uint32_t r = 0;
    for(size_t i = 0; i < sizeof(T); i++){
        r ^= table[i][(char)(key >> 8*i)];
    }
    return (uint32_t)(((uint64_t)(r) * (N)) >> 32);
}

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

template<typename T>
size_t multiply_shift(T key, size_t N){
    size_t a = 0x4D7C6D4D;
    a ^= N;
    size_t y = a * key;
    // uint32_t z = (uint32_t)((y>>32) ^ (y));
    // return (uint64_t)(z) * (uint64_t)(N) >> 32;
    return range_bit_based2(y, N);
}

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


template hash_fptr<uint64_t> get_hash_function<>(HashFunction);
template hash_fptr<uint32_t> get_hash_function<>(HashFunction);
template hash_fptr<uint16_t> get_hash_function<>(HashFunction);
template hash_fptr<uint8_t> get_hash_function<>(HashFunction);

template hash_fptr<int64_t> get_hash_function<>(HashFunction);
template hash_fptr<int32_t> get_hash_function<>(HashFunction);
template hash_fptr<int16_t> get_hash_function<>(HashFunction);
template hash_fptr<int8_t> get_hash_function<>(HashFunction);
