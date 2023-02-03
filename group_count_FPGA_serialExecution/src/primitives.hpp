#ifndef PRIMITIVES_HPP
#define PRIMITIVES_HPP

#include <array>

/**
 * This file contains the scalar primitves of the Intel Intrinsics, which are used 
 * in the own AVX512-implementations of the hashbased group_count.
 * These functions will later be used to run the logic of the AVX512 implementations 
 * on a FPGA within the Intel DevCloud.
*/

template<typename T>
struct fpvec {
    [[intel::fpga_register]] std::array<T, 64/sizeof(T)> elements;
};

/* // print a fpvec<T> result register
	for (int i=0; i<(64/sizeof(T)); i++) {
		std::cout << reg.elements[i] << " ";
	} 
*/

/**	#1
 * serial primitive for Intel Intrinsic:
 * _mm512_setzero_epi32
 */
template<typename T>
fpvec<T> setzero() {
	auto reg = fpvec<T>{};
	uint32_t zero = 0;
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		reg.elements[i] = zero;
	}
	return reg;
}

/**	#2
 * serial primitive for Intel Intrinsic:
 * __m512i _mm512_setr_epi32 (int e15, int e14, int e13, int e12, 
 * int e11, int e10, int e9, int e8, int e7, int e6, int e5, int e4, int e3, 
 * int e2, int e1, int e0)
 * 
 * function will (currently) only be working for arrys with 16 elements of 32bit integers!
 */
template<typename T>
fpvec<T> setr_16slot(uint32_t e15, uint32_t e14, uint32_t e13, uint32_t e12, uint32_t e11, uint32_t e10, uint32_t e9,
	uint32_t e8, uint32_t e7, uint32_t e6, uint32_t e5, uint32_t e4, uint32_t e3, uint32_t e2, uint32_t e1, uint32_t e0) {
	auto reg = fpvec<T>{};
#pragma unroll
	reg.elements[0] = e0;
	reg.elements[1] = e1;
	reg.elements[2] = e2;
	reg.elements[3] = e3;
	reg.elements[4] = e4;
	reg.elements[5] = e5;
	reg.elements[6] = e6;
	reg.elements[7] = e7;
	reg.elements[8] = e8;
	reg.elements[9] = e9;
	reg.elements[10] = e10;
	reg.elements[11] = e11;
	reg.elements[12] = e12;
	reg.elements[13] = e13;
	reg.elements[14] = e14;
	reg.elements[15] = e15;
	return reg;
}

/**	#3
* serial primitive for Intel Intrinsic:
* __m512i _mm512_set1_epi32 (int a)
*/
template<typename T>
fpvec<T> set1(T value) {
	auto reg = fpvec<T>{};
	#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		reg.elements[i] = value;
	}
	return reg;
}

/**	#4
* serial primitive for Intel Intrinsic:
* __mmask16 _cvtu32_mask16 (unsigned int a)
* original description: "Convert integer value a into an 16-bit mask, and store the result in k."*
*/
template<typename T>
fpvec<T> cvtu32_mask16(T n) {
	auto reg = fpvec<T>{};
	int lastElement = ((64/sizeof(T))-1);
	#pragma unroll
	while (lastElement >= 0) {
         // storing remainder in array
        reg.elements[lastElement] = (n >> lastElement) & 0x1;;
		lastElement = lastElement-1;
    }
	return reg;
}

/**	#5
* serial primitive for two Intel Intrinsics:
* __m512i _mm512_maskz_loadu_epi32 (__mmask16 k, void const* mem_addr)
* __m512i _mm512_mask_loadu_epi32 (__m512i src, __mmask16 k, void const* mem_addr)
* _mm512_maskz_loadu_epi32	:	original description: "Load packed 32-bit integers from memory 
*								into dst using zeromask k (elements are zeroed out when the 
*								corresponding mask bit is not set). mem_addr does not need to 
*								be aligned on any particular boundary."
* _mm512_mask_loadu_epi32	:	original description: "Load packed 32-bit integers from memory 
*								into dst using writemask k (elements are copied from src when 
*								the corresponding mask bit is not set). 
*								mem_addr does not need to be aligned on any particular boundary."
*
* customized loadu-function:
* @param writeMask : if bit is set to "1" load related item from data
* @param data : array which contains the data which should be loaded
* @param startIndex : first index-position of data from where the data should be loaded
* @param HSZIZE : HSIZE that describes the size of the arrays of the Hashvector (data array)
*/
template<typename T>
fpvec<T> mask_loadu(fpvec<T>& writeMask, uint32_t* data, uint32_t startIndex, uint64_t HSIZE) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (writeMask.elements[i] == 1) {
			// old reg.elements[i] = data[(startIndex+i)%HSIZE];
			reg.elements[i] = data[startIndex+i];
		}
		else {
			reg.elements[i] = 0;
		}
	}
	return reg;
}

/**	#6
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_mask_cmpeq_epi32_mask (__mmask16 k1, __m512i a, __m512i b)
* original description: "Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k 
* using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set)."
*/
template<typename T>
fpvec<T> mask_cmpeq_epi32_mask(fpvec<T>& zeroMask, fpvec<T>& a, fpvec<T>& b) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (zeroMask.elements[i] == 1) {
			if (a.elements[i] == b.elements[i]) {
				reg.elements[i] = 1;
			}	
			else {
				reg.elements[i] = 0;
			}
		}	
		else {
			reg.elements[i] = 0;
		}	
	}
	return reg;
}

/**	#7
* serial primitive for Intel Intrinsic:
* __m512i _mm512_mask_add_epi32 (__m512i src, __mmask16 k, __m512i a, __m512i b)
* original description: "Add packed 32-bit integers in a and b, and store the results in dst using writemask k 
* (elements are copied from src when the corresponding mask bit is not set)."
*/
template<typename T>
fpvec<T> mask_add_epi32(fpvec<T>& src, fpvec<T>& writeMask, fpvec<T>& a, fpvec<T>& b) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (writeMask.elements[i] == 1) {
			reg.elements[i] = a.elements[i] + b.elements[i];
		}
		else {
			reg.elements[i] = src.elements[i];
		}
	}
	return reg;
}

/**	#8
* serial primitive for Intel Intrinsic:
* void _mm512_mask_storeu_epi32 (void* mem_addr, __mmask16 k, __m512i a)
* original description: "Store packed 32-bit integers from a (=data) into memory using writemask k. 
* mem_addr does not need to be aligned on any particular boundary."
*
* customized store  - function:
* @param result : array, in which the data is stored, if related bit of writeMask is set to "1"
* @param startIndex : first index - position of data from where the data should be stored
* @param HSZIZE : HSIZE that describes the size of the arrays of the Hashvector (result array)
* @param writeMask : if bit is set to "1" -> store related item from data into result array
* @param data : register-array which contains the data that should be stored
*/
template<typename T>
void mask_storeu_epi32(uint32_t* result, uint32_t startIndex, uint64_t HSIZE, fpvec<T>& writeMask, fpvec<T>& data) {
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (writeMask.elements[i] == 1) {
			result[(startIndex+i)%HSIZE] = data.elements[i];
		}
	}
}

/**	#9
* serial primitive for Intel Intrinsic:
* int _mm512_mask2int (__mmask16 k1)
* original description: "Converts bit mask k1 into an integer value, storing the results in dst."
* own (simplified implementation):
* return 1 if at least 1 bit of mask is set;
* return 0 if no bit of mask is set
*/
template<typename T>
uint32_t mask2int(fpvec<T>& mask) {
	uint32_t res = 0;
	#pragma unroll	
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (mask.elements[i] == 1) {
			res = 1;
		}
	}
	return res;
}

/**	#10
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_knot (__mmask16 a)
* original description: "Compute the bitwise NOT of 16-bit mask a, and store the result in k."
*/
template<typename T>
fpvec<T> knot(fpvec<T>& src) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (src.elements[i] == 0) {
			reg.elements[i] = 1;
		}
		else {
			reg.elements[i] = 0;
		}
		// reg.elements[i] = !src.elements[i];
	}
	return reg;
}

/**	#11
* serial primitive for Built-in Function Provided by GCC:
* int __builtin_clz (unsigned int x)
* original description: "Built-in Function: int __builtin_clz (unsigned int x)
* Returns the number of leading 0-bits in x, starting at the most significant bit position. 
* If x is 0, the result is undefined."
*/
template<typename T>
uint32_t clz_onceBultin(fpvec<T>& src) {
	uint32_t res = 0;
	#pragma unroll
	for (int i=((64/sizeof(T))-1); i>=0; i--) {
		if (src.elements[i]==0) {
			res = res+1;
		} else {
			break;
		}
	}
	return res;
}

/**	#12
* serial primitive for Intel Intrinsic:
* __m512i _mm512_load_epi32 (void const* mem_addr)
* original description: "Load 512-bits (composed of 16 packed 32-bit integers) from memory into dst. 
* mem_addr must be aligned on a 64-byte boundary or a general-protection exception may be generated."
*
* customized load-function:
* @param templateMask : Register of type fpvec<uint32_t>
* @param data : array which contains the data which should be loaded
* @param startIndex : first index-position of data from where the data should be loaded
* @param HSZIZE : HSIZE that describes the size of the arrays of the Hashvector (data array)
*/
template<typename T>
fpvec<T> load_epi32(fpvec<T>& templateMask, uint32_t* data, uint32_t startIndex, uint64_t HSIZE) {		// testen - fehlt Parameter <T> ?
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		reg.elements[i] = data[startIndex+i];
	}
	return reg;
}

/**	#13
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_cmpeq_epi32_mask (__m512i a, __m512i b)
* original description: "Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k."
*/
template<typename T>
fpvec<T> cmpeq_epi32_mask(fpvec<T>& a, fpvec<T>& b) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (a.elements[i] == b.elements[i]) {
			reg.elements[i] = 1;
		}
		else {
			reg.elements[i] = 0;
		}
	}
	return reg;
}

/**	#14
* serial primitive for Intel Intrinsic:
* __m512i _mm512_permutexvar_epi32 (__m512i idx, __m512i a)
* original description: "Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst."
*/
template<typename T>
fpvec<T> permutexvar_epi32(fpvec<T>& idx, fpvec<T>& a) {
	auto reg = fpvec<T>{};
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		uint32_t id = idx.elements[i];
		uint32_t value = a.elements[id];
		reg.elements[i] = value;
	}
	return reg;
}

/**	#15
* serial primitive for Built-in Function Provided by GCC:
* int __builtin_ctz (unsigned int x)
* original description: "Built-in Function: int __builtin_ctz (unsigned int x)
* Returns the number of trailing 0-bits in x, starting at the least significant bit position. 
* If x is 0, the result is undefined."
*/
template<typename T>
uint32_t ctz_onceBultin(fpvec<T>& src) {
	uint32_t res = 0;
	#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if (src.elements[i]==0) {
			res = res+1;
		} else {
			break;
		}
	}
	return res;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//// New functions for SoAoV approach only - not in SoA-implementations ////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**	#16 
* serial primitive for Intel Intrinsic:
* __m512i _mm512_mask_set1_epi32 (__m512i src, __mmask16 k, int a)
* original description: "Broadcast 32-bit integer a to all elements of dst using 
* writemask k (elements are copied from src when the corresponding mask bit is not set)."
*/
template<typename T>
fpvec<T> mask_set1(fpvec<T>& src, fpvec<T>& writeMask, uint32_t value) {
	auto reg = fpvec<T>{};
	#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		if(writeMask.elements[i] == 1) {
			reg.elements[i] = value;
		} else {
			reg.elements[i] = src.elements[i];
		}		
	}
	return reg;
}

/**	#17
* serial primitive for Intel Intrinsic:
* void _mm512_store_epi32 (void* mem_addr, __m512i a)
* original description: "Store 512-bits (composed of 16 packed 32-bit integers) from a into memory. 
* mem_addr must be aligned on a 64-byte boundary or a general-protection exception may be generated."
*
* customized store  - function:
* @param result : array, in which the data is stored; function store 512-bits
* @param startIndex : first index - position of data from where the data should be stored
* @param data : register-array which contains the data that should be stored
*/
template<typename T>
void store_epi32(uint32_t* result, uint32_t startIndex, fpvec<T>& data) {
#pragma unroll
	for (int i=0; i<(64/sizeof(T)); i++) {
		result[(startIndex+i)] = data.elements[i];
	}
}


#endif // PRIMITIVES_HPP