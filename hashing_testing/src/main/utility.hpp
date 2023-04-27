#ifndef TUD_HASHING_TESTING_AVX512_MAIN_UTILITY
#define TUD_HASHING_TESTING_AVX512_MAIN_UTILITY

#include <chrono>
#include <iostream>

std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

#endif //TUD_HASHING_TESTING_AVX512_MAIN_UTILITY