#include "avx512_gc_aosov_v2.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_AoSoV_v2<T>::AVX512_gc_AoSoV_v2(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  AVX512_gc_AoSoV_v1<T>(HSIZE * 16, hash_function)
{}

template <typename T>
AVX512_gc_AoSoV_v2<T>::~AVX512_gc_AoSoV_v2(){}

template class AVX512_gc_AoSoV_v2<uint32_t>;
// template class AVX512_gc_AoSoV_v2<uint64_t>;