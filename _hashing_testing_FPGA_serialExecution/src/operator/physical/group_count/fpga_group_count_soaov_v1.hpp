#ifndef TUD_HASHING_TESTING_FPGA_GROUP_COUNT_SOAOV_V1
#define TUD_HASHING_TESTING_FPGA_GROUP_COUNT_SOAOV_V1

#include <stdint.h>
#include <stdlib.h> 

#include <immintrin.h>

#include "../../logical/group_count.hpp"
#include "../../logical/primitives.hpp"
#include "scalar_group_count.hpp"

/// @brief FPGA_group_count_SoAoV_v1 uses the serial primitives (transformed of Intel Intrinsics of AVX512-version). It has an array of vectors which don't need to be loaded
/// @tparam T 
template <typename T>
class FPGA_group_count_SoAoV_v1 : public Scalar_group_count<T>{
    protected:
        size_t m_elements_per_vector;
        size_t m_HSIZE_v;
        
    public:
        FPGA_group_count_SoAoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t));
        ~FPGA_group_count_SoAoV_v1();
        
        void create_hash_table(uint32_t* input, size_t data_size);

        std::string identify();

        size_t get_HSIZE(){
            return m_HSIZE_v * m_elements_per_vector;
        }
};

#endif //TUD_HASHING_TESTING_FPGA_GROUP_COUNT_SOAOV_V1