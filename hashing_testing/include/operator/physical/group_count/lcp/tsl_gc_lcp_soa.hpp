#ifndef TUD_HASHING_TESTING_TSL_GROUP_COUNT_LCP_SOA
#define TUD_HASHING_TESTING_TSL_GROUP_COUNT_LCP_SOA

#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include <tslintrin.hpp>

#include "operator/logical/tsl_group_count_soa.hpp"

/// @brief TSL_gc_LCP_SoA uses AVX512. It has an array of vectors which don't need to be loaded
/// @tparam T 
template <class SimdT, typename T>
class TSL_gc_LCP_SoA : public Group_Count_TSL_SOA<T>{
    
    using ps = tsl::simd<T, SimdT>;
    using vec_t = typename ps::register_type;
    using mask_t = typename ps::mask_type;
    using imask_t = typename ps::imask_type;
    const size_t m_elements_per_vector = ps::vector_element_count(); // todo!

    protected:
        size_t m_HSIZE_v;

        vec_t * m_hash_map_v;
        vec_t * m_count_map_v;

        imask_t * m_masks;  //insertion masks

        bool m_transfer = false;
        
    public:
        TSL_gc_LCP_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t), size_t numa_node);
        
        virtual ~TSL_gc_LCP_SoA();
        
        void create_hash_table(T* input, size_t data_size);

        void print(bool horizontal);

        std::string identify(){
            return "TSL LCP SoA";
        }

        void clear();

        size_t get_HSIZE(){
            return m_HSIZE_v * m_elements_per_vector;
        }

        T get(T value);
};


#endif //TUD_HASHING_TESTING_TSL_GROUP_COUNT_LCP_SOA