#ifndef TUD_HASHING_TESTING_AVX512_OPERATOR_UTILITY
#define TUD_HASHING_TESTING_AVX512_OPERATOR_UTILITY

#include <iostream>
#include <stdint.h>
#include <stdlib.h> 

#include <immintrin.h>

void print512i(__m512i a, bool newline = true){
    uint32_t *res = (uint32_t*) aligned_alloc(64, 1*sizeof(__m512i));
    _mm512_store_epi32 (res, a);
    for(uint32_t i = 16; i > 0; i--){
        std::cout << res[i-1] << "\t";
    }
    free(res);
    if(newline){
        std::cout << std::endl;
    }

}


void printMask(__mmask16 mask, bool newline = true){
    int32_t m = (int32_t)mask;
    for(size_t i = 16; i > 0; i--){
        std::cout << ((m >> (i - 1)) & 0b1) << "\t";
    }
    if(newline){
        std::cout << std::endl;
    }
}

#endif //TUD_HASHING_TESTING_AVX512_OPERATOR_UTILITY