#ifndef TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LP_V_SOA_conflict_v1
#define TUD_HASHING_TESTING_AVX512_GROUP_COUNT_LP_V_SOA_conflict_v1

#include <stdint.h>
#include <stdlib.h> 

#include <tslintrin.hpp>

#include "operator/logical/group_count.hpp"
#include "operator/physical/group_count/lp/scalar_gc_soa.hpp"

/// @brief TSL_gc_LP_V_SoA uses AVX512. It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template <class SimdT, typename T>
class TSL_gc_LP_V_SoA : public Group_Count_TSL_SOA<T>{

    using ps = tsl::simd<T, SimdT>;
    using vec_t = typename ps::register_type;
    using mask_t = typename ps::mask_type;
    using imask_t = typename ps::imask_type;


    public:
        TSL_gc_LP_V_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t), size_t numa_node);
        virtual ~TSL_gc_LP_V_SoA();
        
        void create_hash_table(T* input, size_t data_size);
        
        void clear();

        T get(T value);

        std::string identify(){
            return "TSL LP vertical SoA";
        }
};


#endif //TUD_HASHING_TESTING_AVX512_GROUP_COUNT_SOA_conflict_v1