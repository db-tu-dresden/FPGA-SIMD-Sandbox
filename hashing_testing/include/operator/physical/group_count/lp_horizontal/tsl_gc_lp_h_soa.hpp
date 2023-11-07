#ifndef TUD_HASHING_TESTING_TSL_GROUP_COUNT_LP_H_SOA
#define TUD_HASHING_TESTING_TSL_GROUP_COUNT_LP_H_SOA

#include <stdint.h>
#include <stdlib.h> 

#include <tslintrin.hpp>

#include "operator/logical/group_count.hpp"
#include "operator/physical/group_count/lp/scalar_gc_soa.hpp"

/// @brief TSL_gc_SoA_v1 uses TSL. It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template<class SimdT, typename T>
class TSL_gc_LP_H_SoA : public Scalar_gc_SoA<T>{
    public:
        TSL_gc_LP_H_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t));
        virtual ~TSL_gc_LP_H_SoA();
        
        void create_hash_table(T* input, size_t data_size);
        
        T get(T value);

        std::string identify(){
            return "TSL LP horizontal SoA";
        }
};


#endif //TUD_HASHING_TESTING_TSL_GROUP_COUNT_SOA_V1