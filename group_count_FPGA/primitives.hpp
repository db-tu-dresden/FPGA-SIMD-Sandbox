#ifndef PRIMITIVES_HPP
#define PRIMITIVES_HPP

/**
 * This file contains the scalar primitves of the Intel Intrinsics, which are used 
 * in the own AVX512-implementation of the hashbased group_count.
 * These functions will later be used to run the logic of the AVX512 implementation 
 * on an FPGA within the Intel DevCloud.
*/

template<typename T>
struct fpvec {
    [[intel::fpga_register]] std::array<T, 64/sizeof(T)> elements;
};

/**	#1
 * scalar primitive for Intel Intrinsic:
 * _mm512_setzero_epi32
 */
template<typename T>
fpvec<T> setzero_uint32() {
	auto reg = fpvec<T>{};
	uint32_t zero = 0;
#pragma unroll
	for (int i = 0; i < 16; i++) {
		reg.elements[i] = zero;
	}
	return reg;
}

/**	#2
 * scalar primitive for Intel Intrinsic:
 * _mm512_setr_epi32
 */

/**	#3
* scalar primitive for Intel Intrinsic:
* _mm512_set1_epi32
*/
template<typename T>
fpvec<T> set1_uint32(T value) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i = 0; i < 16; i++) {
		reg.elements[i] = value;
	}
	return reg;
}

/**	#4
* scalar primitive for Intel Intrinsic:
* _mm512_maskz_loadu_epi32
*/

/**	#5
* scalar primitive for Intel Intrinsic:
* _mm512_cmpeq_epi32_mask
*/

/**	#6
* scalar primitive for Intel Intrinsic:
* _mm512_mask_loadu_epi32
*/

/**	#7
* scalar primitive for Intel Intrinsic:
* _mm512_mask_add_epi32
*/

/**	#8
* scalar primitive for Intel Intrinsic:
* _mm512_mask_storeu_epi32
*/

/**	#9
* scalar primitive for Intel Intrinsic:
* _mm512_mask2int
*/

/**	#10
* scalar primitive for Intel Intrinsic:
* _mm512_knot
*/

/**	#11
* scalar primitive for Built-in Function Provided by GCC:
* __builtin_clz
*/

/**	#12
* scalar primitive for Intel Intrinsic:
* _mm512_load_epi32
*/

/**	#13
* scalar primitive for Intel Intrinsic:
* _mm512_permutexvar_epi32
*/

#endif // PRIMITIVES_HPP