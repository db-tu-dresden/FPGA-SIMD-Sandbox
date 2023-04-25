#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LP_H_AOS_V1
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LP_H_AOS_V1

#include <stdint.h>
#include <stdlib.h> 
#include "../../../logical/group_count.hpp"
#include "../lp/scalar_gc_aos.hpp"

/// @brief AVX512_gc_AoS_v1 uses AVX512. It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template <typename T>
class AVX512_gc_AoS_v1 : public Scalar_gc_AoS<T>{
    public:
        AVX512_gc_AoS_v1(size_t HSIZE, size_t (*hash_function)(T, size_t));
        virtual ~AVX512_gc_AoS_v1();
        
        void create_hash_table(T* input, size_t data_size);
        
        std::string identify();
};


#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_AOS_V1