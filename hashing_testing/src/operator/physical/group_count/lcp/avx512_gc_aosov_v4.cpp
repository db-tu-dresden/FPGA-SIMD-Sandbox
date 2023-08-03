#include "operator/physical/group_count/lcp/avx512_gc_aosov_v4.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_AoSoV_v4<T>::AVX512_gc_AoSoV_v4(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  AVX512_gc_AoSoV_v3<T>(HSIZE * 8, hash_function)
{}

template <typename T>
AVX512_gc_AoSoV_v4<T>::~AVX512_gc_AoSoV_v4(){}

template class AVX512_gc_AoSoV_v4<uint32_t>;
// template class AVX512_gc_AoSoV_v4<uint64_t>;