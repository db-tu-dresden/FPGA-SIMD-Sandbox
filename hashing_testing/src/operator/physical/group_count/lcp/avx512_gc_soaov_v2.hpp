#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LCP_SOAOV_v2
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LCP_SOAOV_v2

#include <stdint.h>
#include <stdlib.h> 
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "../../../logical/group_count.hpp"
#include "avx512_gc_soaov_v1.hpp"

/// @brief AVX512_gc_SoAoV_v2 uses AVX512. It has an array of vectors which don't need to be loaded
/// @tparam T 
template <typename T>
class AVX512_gc_SoAoV_v2 : public AVX512_gc_SoAoV_v1<T>{
    protected:

    public:
        AVX512_gc_SoAoV_v2(size_t HSIZE, size_t (*hash_function)(T, size_t));
        virtual ~AVX512_gc_SoAoV_v2();

        std::string identify();
};

#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_v2