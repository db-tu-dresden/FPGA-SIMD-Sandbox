#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1

#include <stdint.h>
#include <stdlib.h> 

#include <immintrin.h>

#include "../../logical/group_count.hpp"
#include "scalar_group_count.hpp"

/// @brief AVX512_group_count_SoAoV_v1 uses AVX512. It has an array of vectors which don't need to be loaded
/// @tparam T 
template <typename T>
class AVX512_group_count_SoAoV_v1 : public Scalar_group_count<T>{
    protected:
        size_t m_elements_per_vector;
        size_t m_HSIZE_v;
        
    public:
        AVX512_group_count_SoAoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t));
        ~AVX512_group_count_SoAoV_v1();
        
        void create_hash_table(uint32_t* input, size_t data_size);

        std::string identify();

        size_t get_HSIZE(){
            return m_HSIZE_v * m_elements_per_vector;
        }

        T get(T value);
};


void print512i(__m512i a, bool newline = true);

#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1