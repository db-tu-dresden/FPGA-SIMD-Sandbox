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
template<typename T>
fpvec<T> setzero_uint32(T e15, T e14, T e13, T e12, T e11, T e10, T e9,
	T e8, T e7, T e6, T e5, T e4, T e3, T e2, T e1, T e0) {
	auto result = fpvec<T>{};
#pragma unroll
	result[0] = e0;
	result[1] = e1;
	result[2] = e2;
	result[3] = e3;
	result[4] = e4;
	result[5] = e5;
	result[6] = e6;
	result[7] = e7;
	result[8] = e8;
	result[9] = e9;
	result[10] = e10;
	result[11] = e11;
	result[12] = e12;
	result[13] = e13;
	result[14] = e14;
	result[15] = e15;
	return result;
}

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
* original description: "Load packed 32-bit integers from memory into dst 
* using zeromask k (elements are zeroed out when the corresponding mask bit 
* is not set). mem_addr does not need to be aligned on any particular boundary."
*
* customized load-function (only a single one for FPGA implementation):
* param1 writeMask : if bit is set to "1" load related item from data
* param2 data : array which contains the data which should be loaded
* param3 startIndex : first index-position of data from where the data should be loaded
*/
template<typename T>
fpvec<T> maskz_loadu_uint32(fpvec<T> writeMask, fpvec<T> data, T startIndex) {
	auto result = fpvec<T>{};
#pragma unroll
	for (int i = 0; i < 16; i++) {
		if (writeMask[i] == 1) {
			result[i] = data[startIndex+i];
		}
		else {
			result[i] = 0;
		}
	}
	return result;
}

/**	#5
* scalar primitive for Intel Intrinsic:
* _mm512_cmpeq_epi32_mask
* "Compare packed 32-bit integers in a and b for equality, 
* and store the results in mask vector k."
*/
template<typename T>
fpvec<T> cmpeq_uint32_mask(fpvec<T> a, fpvec<T> b) {
	auto resultMask = fpvec<T>{};
#pragma unroll
	for (int i = 0; i < 16; i++) {
		if (a[i] == b[i]) {
			resultMask[i] = 1;
		}
		else {
			resultMask[i] = 0;
		}
	}
	return resultMask;
}

/**	#6
* scalar primitive for Intel Intrinsic:
* _mm512_mask_loadu_epi32
* original description: "Load packed 32-bit integers from memory into dst using writemask k 
* (elements are copied from src when the corresponding mask bit is not 
* set). mem_addr does not need to be aligned on any particular boundary."
*
* customized load-function:
* param1 src : array from which the data is loaded, of related bit of writeMask is set to "0"
* param2 writeMask : if bit is set to "1" load related item from data
* param3 data : array which contains the data which should be loaded
* param4 startIndex : first index-position of data from where the data should be loaded
*/
template<typename T>
fpvec<T> mask_loadu_uint32(fpvec<T> src, fpvec<T> writeMask, fpvec<T> data, T startIndex) {
	auto result = fpvec<T>{};
#pragma unroll
	for (int i = 0; i < 16; i++) {
		if (writeMask[i] == 1) {
			result[i] = data[startIndex + i];
		}
		else {
			result[i] = src[i];
		}
	}
	return result;
}


/**	#7
* scalar primitive for Intel Intrinsic:
* _mm512_mask_add_epi32
* "Add packed 32-bit integers in a and b, and store the 
* results in dst using writemask k (elements are copied 
* from src when the corresponding mask bit is not set)."
*/
template<typename T>
fpvec<T> mask_add_uint32(fpvec<T> src, fpvec<T> writeMask, fpvec<T> a, fpvec<T> b) {
	auto result = fpvec<T>{};
#pragma unroll
	for (int i = 0; i < 16; i++) {
		if (writeMask[i] == 1) {
			result[i] = a[i] + b[i];
		}
		else {
			result[i] = src[i];
		}
	}
	return result;
}

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
* original description: "Load 512-bits (composed of 16 packed 32-bit integers)
* from memory into dst. mem_addr must be aligned on a 64-byte boundary or a
* general-protection exception may be generated."
*
* customized load-function:
*/
template<typename T>
fpvec<T> load_uint32(fpvec<T> data, T startIndex) {
	auto result = fpvec<T>{};
#pragma unroll
	for (int i = 0; i < 16; i++) {
		result[i] = data[startIndex + i];
	}
	return result;
}

/**	#13
* scalar primitive for Intel Intrinsic:
* _mm512_permutexvar_epi32
*/

#endif // PRIMITIVES_HPP