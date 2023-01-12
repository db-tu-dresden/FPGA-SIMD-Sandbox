#ifndef PRIMITIVES_HPP
#define PRIMITIVES_HPP

#include <array>

/**
 * This file contains the scalar primitves of the Intel Intrinsics, which are used 
 * in the own AVX512-implementation of the hashbased group_count.
 * These functions will later be used to run the logic of the AVX512 implementation 
 * on a FPGA within the Intel DevCloud.
*/

template<typename T>
struct fpvec {
    [[intel::fpga_register]] std::array<T, 64/sizeof(T)> elements;
};

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
 * _mm512_setr_epi32
 * 
 * function will (currently) only be working for arrys with 16 elements a 32bit integers!
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
* _mm512_set1_epi32
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
* _cvtu32_mask16
* original description: "Convert integer value a into an 16-bit mask, and store the result in k."*
*/
template<typename T>
fpvec<T> cvtu32_mask16(T n) {
	auto reg = fpvec<T>{};
	int lastElement = ((64/sizeof(T))-1);
	#pragma unroll
	while (lastElement >= 0) {
         // storing remainder in array
        reg.elements[lastElement] = n % 2;
		//std::cout << reg.elements[lastElement] << std::endl;
		if (n>0) {
			n = n / 2;
		} else {
			n = n;
		}
                lastElement = lastElement-1;
    } 
	/* // print fpvec result register
	for (int i=0; i<(64/sizeof(T)); i++) {
		std::cout << reg.elements[i] << " ";
	} */

	return reg;
}

/**	#5
* serial primitive for two Intel Intrinsics:
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
			reg.elements[i] = data[(startIndex+i)%HSIZE];
		}
		else {
			reg.elements[i] = 0;
		}
	}
	return reg;
}

/**	#6
* serial primitive for Intel Intrinsic:
* _mm512_mask_cmpeq_epi32_mask
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
* _mm512_mask_add_epi32
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
* _mm512_mask_storeu_epi32
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
		else {
			result[(startIndex+i)%HSIZE] = result[(startIndex+i)%HSIZE];		// not necessary? do nothing?
		}
	}
}

/**	#9
* serial primitive for Intel Intrinsic:
* _mm512_mask2int
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
* _mm512_knot
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
	}
	return reg;
}

/**	#11
* serial primitive for Built-in Function Provided by GCC:
* __builtin_clz
* original description: "Built-in Function: int __builtin_clz (unsigned int x)
* Returns the number of leading 0-bits in x, starting at the most significant bit position. 
* If x is 0, the result is undefined."
*/
template<typename T>
uint32_t clz_onceBultin(fpvec<T>& src) {
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

/**	#12
* serial primitive for Intel Intrinsic:
* _mm512_load_epi32
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
		reg.elements[i] = data[(startIndex+i)%HSIZE];
	}
	return reg;
}

/**	#13
* serial primitive for Intel Intrinsic:
* _mm512_cmpeq_epi32_mask
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
* _mm512_permutexvar_epi32
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
* __builtin_ctz
* original description: "Built-in Function: int __builtin_ctz (unsigned int x)
* Returns the number of trailing 0-bits in x, starting at the least significant bit position. 
* If x is 0, the result is undefined."
*/
template<typename T>
uint32_t ctz_onceBultin(fpvec<T>& src) {
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

#endif // PRIMITIVES_HPP