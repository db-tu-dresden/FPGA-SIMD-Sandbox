#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LP_H_SOA_v3
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LP_H_SOA_v3

#include <stdint.h>
#include <stdlib.h> 
#include "../../../logical/group_count.hpp"
#include "../lp/scalar_gc_soa.hpp"

/// @brief AVX512_gc_SoA_v3 uses AVX512. It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template <typename T>
class AVX512_gc_SoA_v3 : public Scalar_gc_SoA<T>{
    public:
        AVX512_gc_SoA_v3(size_t HSIZE, size_t (*hash_function)(T, size_t));
        virtual ~AVX512_gc_SoA_v3();
        
        void create_hash_table(T* input, size_t data_size);
        
        std::string identify(){
            return "LP horizontal SoA V3";
        }

        T get(T value);
};


#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_v3