#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_v3
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_v3

#include <stdint.h>
#include <stdlib.h> 
#include "../../logical/group_count.hpp"
#include "scalar_group_count.hpp"

/// @brief AVX512_group_count_SoA_v3 uses AVX512. It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template <typename T>
class AVX512_group_count_SoA_v3 : public Scalar_group_count<T>{
    public:
        AVX512_group_count_SoA_v3(size_t HSIZE, size_t (*hash_function)(T, size_t));
        ~AVX512_group_count_SoA_v3();
        
        void create_hash_table(T* input, size_t data_size);
        
        std::string identify();

        T get(T value);
};


#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_v3