#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_COLLISION_v1
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_COLLISION_v1

#include <stdint.h>
#include <stdlib.h> 
#include "../../logical/group_count.hpp"
#include "scalar_group_count.hpp"

/// @brief AVX512_group_count_SoA_collision_v1 uses AVX512. It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template <typename T>
class AVX512_group_count_SoA_collision_v1 : public Scalar_group_count<T>{
    public:
        AVX512_group_count_SoA_collision_v1(size_t HSIZE, size_t (*hash_function)(T, size_t));
        ~AVX512_group_count_SoA_collision_v1();
        
        void create_hash_table(T* input, size_t data_size);
        
        std::string identify();
};


#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_collision_v1