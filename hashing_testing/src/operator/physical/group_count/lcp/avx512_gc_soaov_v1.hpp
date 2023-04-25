#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LCP_SOAOV_V1
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LCP_SOAOV_V1

#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "../../../logical/group_count.hpp"
#include "../lp/scalar_gc_soa.hpp"

/// @brief AVX512_gc_SoAoV_v1 uses AVX512. It has an array of vectors which don't need to be loaded
/// @tparam T 
template <typename T>
class AVX512_gc_SoAoV_v1 : public Scalar_gc_SoA<T>{
    protected:
        size_t m_elements_per_vector;
        size_t m_HSIZE_v;

        __m512i * m_hash_map_v;
        __m512i * m_count_map_v;

        __mmask16 * m_masks;

        bool m_transfer = false;
        
    public:
        AVX512_gc_SoAoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t));
        virtual ~AVX512_gc_SoAoV_v1();
        
        void create_hash_table(uint32_t* input, size_t data_size);

        void print(bool horizontal);

        std::string identify();

        size_t get_HSIZE(){
            return m_HSIZE_v * m_elements_per_vector;
        }

        T get(T value);
};

#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1