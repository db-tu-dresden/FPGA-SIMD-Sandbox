#ifndef TUD_HASHING_TESTING_HASH_TABLE_ALGORITHMS
#define TUD_HASHING_TESTING_HASH_TABLE_ALGORITHMS

#include "main/hash_function.hpp"

#include "operator/physical/group_count/lp/scalar_gc_soa.hpp"
#include "operator/physical/group_count/lp/scalar_gc_aos.hpp"
#include "operator/physical/group_count/lp/scalar_gc_aos_v2.hpp"


//chained is somehow broken rn.
// #include "operator/physical/group_count/chained/chained.hpp"
// #include "operator/physical/group_count/chained/chained2.hpp"

#ifdef USE_AVX512
#include "operator/physical/group_count/lp_horizontal/avx512_gc_soa_v1.hpp"
#include "operator/physical/group_count/lp_horizontal/avx512_gc_soa_v2.hpp"
#include "operator/physical/group_count/lp_horizontal/avx512_gc_soa_v3.hpp"
#include "operator/physical/group_count/lp_horizontal/avx512_gc_aos_v1.hpp"

#include "operator/physical/group_count/lcp/avx512_gc_soaov_v1.hpp"
#include "operator/physical/group_count/lcp/avx512_gc_soaov_v2.hpp"
#include "operator/physical/group_count/lcp/avx512_gc_aosov_v1.hpp"
#include "operator/physical/group_count/lcp/avx512_gc_aosov_v2.hpp"
#include "operator/physical/group_count/lcp/avx512_gc_aosov_v3.hpp"
#include "operator/physical/group_count/lcp/avx512_gc_aosov_v4.hpp"

#include "operator/physical/group_count/lp_vertical/avx512_gc_soa_conflict_v1.hpp"
#include "operator/physical/group_count/lp_vertical/avx512_gc_soa_conflict_v2.hpp"
#include "operator/physical/group_count/lp_vertical/avx512_gc_aos_conflict_v1.hpp"
#endif

enum Group_Count_Algorithm{
    SCALAR_GROUP_COUNT_SOA, 
    AVX512_GROUP_COUNT_SOA_V1, 
    AVX512_GROUP_COUNT_SOA_V2, 
    AVX512_GROUP_COUNT_SOA_V3, 
    AVX512_GROUP_COUNT_SOAOV_V1,     
    AVX512_GROUP_COUNT_SOAOV_V2, 
    AVX512_GROUP_COUNT_SOA_CONFLICT_V1, 
    AVX512_GROUP_COUNT_SOA_CONFLICT_V2,
    SCALAR_GROUP_COUNT_AOS, 
    SCALAR_GROUP_COUNT_AOS_V2, 
    AVX512_GROUP_COUNT_AOS_V1, 
    AVX512_GROUP_COUNT_AOS_V2, 
    AVX512_GROUP_COUNT_AOS_V3, 
    AVX512_GROUP_COUNT_AOSOV_V1,     
    AVX512_GROUP_COUNT_AOSOV_V2,    
    AVX512_GROUP_COUNT_AOSOV_V3,     
    AVX512_GROUP_COUNT_AOSOV_V4,  
    AVX512_GROUP_COUNT_AOS_CONFLICT_V1, 
    AVX512_GROUP_COUNT_AOS_CONFLICT_V2,
    CHAINED,
    CHAINED2
};


/// @brief creates a new instance of Group_count for the selected Group_Count_Algorithm
/// @tparam T the data type for the execution
/// @param run the instance of the Group_count. If not nullptr then it gets deleted
/// @param test the algorithm that shall be created
/// @param HSIZE the hash table size for the algorithm
/// @param function the hash function that shall be used
template <typename T>
void getGroupCount(Group_count<T> *& run, Group_Count_Algorithm test, size_t HSIZE, size_t (*function)(T, size_t));


#endif //TUD_HASHING_TESTING_HASH_TABLE_ALGORITHMS