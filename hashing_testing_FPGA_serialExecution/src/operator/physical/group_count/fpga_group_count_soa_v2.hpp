#ifndef TUD_HASHING_TESTING_FPGA_GROUP_COUNT_SOA_v2
#define TUD_HASHING_TESTING_FPGA_GROUP_COUNT_SOA_v2

#include <stdint.h>
#include <stdlib.h> 
#include "../../logical/group_count.hpp"
#include "../../logical/primitives.hpp"
#include "scalar_group_count.hpp"

/// @brief FPGA_group_count_SoA_v2 uses the serial primitives (transformed of Intel Intrinsics of AVX512-version). It loads the data unaligned form the memory. Linear Probing.
/// @tparam T 
template <typename T>
class FPGA_group_count_SoA_v2 : public Scalar_group_count<T>{
    public:
        FPGA_group_count_SoA_v2(size_t HSIZE, size_t (*hash_function)(T, size_t));
        ~FPGA_group_count_SoA_v2();
        
        void create_hash_table(T* input, size_t data_size);
        
        std::string identify();
};


#endif //TUD_HASHING_TESTING_FPGA_GROUP_COUNT_SOA_v2