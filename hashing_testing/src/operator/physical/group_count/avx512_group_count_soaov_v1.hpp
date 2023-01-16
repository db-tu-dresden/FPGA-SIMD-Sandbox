#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1

#include <stdint.h>
#include <stdlib.h> 

#include <immintrin.h>

#include "../../logical/group_count.hpp"

/// @brief AVX512_group_count_SoAoV_v1 uses AVX512. It has an array of vectors which don't need to be loaded
/// @tparam T 
template <typename T>
class AVX512_group_count_SoAoV_v1 : public Group_count<T>{
    protected:
        __m512i* m_hash_vec;
        __m512i* m_count_vec;

        size_t m_elements_per_vector;

    public:
        AVX512_group_count_SoAoV_v1(size_t HSIZE, T (*hash_function)(T, size_t));
        ~AVX512_group_count_SoAoV_v1();
        
        void create_hash_table(uint32_t* input, size_t data_size);
        
        T get(T value);
        
        void print(bool horizontal);

        std::string identify();
};


#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOAOV_V1