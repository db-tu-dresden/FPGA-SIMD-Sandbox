#ifndef PRIMITIVES_HPP
#define PRIMITIVES_HPP

#include <array>
#include <cmath>	//for pow() function
#include "global_settings.hpp"

/**
 * This file contains the scalar primitves of the Intel Intrinsics, which are used 
 * in the own AVX512-implementations of the hashbased group_count.
 * These functions will be used to run the logic of the AVX512 implementations 
 * on a FPGA within the Intel DevCloud.
 *
 * Some of these functions are adapted to the peculiarities of our implementation. 
 * This is usually due to the goal of simplified data processing. 
 * In some cases, however, the logic of the implemented LinearProbing algorithms required a "special solution".
*/

template<typename T, int B>
struct fpvec {
    [[intel::fpga_register]] std::array<T, (B/sizeof(T))> elements;
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
template<typename T, int B>
fpvec<T,B> setzero() {
	auto reg = fpvec<T,B>{};
	Type zero = 0;
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
/** Not used in current code version --> due to limitations regarding dynamic change of size and amount of elements
template<typename T, int B>
fpvec<T> setr_16slot(uint32_t e15, uint32_t e14, uint32_t e13, uint32_t e12, uint32_t e11, uint32_t e10, uint32_t e9,
	uint32_t e8, uint32_t e7, uint32_t e6, uint32_t e5, uint32_t e4, uint32_t e3, uint32_t e2, uint32_t e1, uint32_t e0) {
	auto reg = fpvec<T>{};
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
*/

/**	#3
* serial primitive for Intel Intrinsic:
* __m512i _mm512_set1_epi32 (int a)
*
* The original Intrinsic was only for 32-bit integers.
* This implemenation is working with uint32_t & uint64_t, etc.
* But be careful with matching ratio of <Type, B> and related T value which is overhanded to the function!
*/
template<typename T, int B>
fpvec<T,B> set1(T value) {
	auto reg = fpvec<T,B>{};
	#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		reg.elements[i] = value;
	}
	return reg;
}

/**	#4
* serial primitive for Intel Intrinsic:
* __mmask16 _cvtu32_mask16 (unsigned int a)
* original description: "Convert integer value a into an 16-bit mask, and store the result in k."*
*/
template<typename T, int B>
fpvec<T,B> cvtu32_mask16(T n) {
	auto reg = fpvec<T,B>{};
	int lastElement = ((B/sizeof(T))-1);
	#pragma unroll
	while (lastElement >= 0) {
         // storing remainder in array
        reg.elements[lastElement] = (n >> lastElement) & 0x1;;
		lastElement = lastElement-1;
    }
	return reg;
}

/**	#4.1
* serial primitive adaption of Intel Intrinsic:
* __mmask16 _cvtu32_mask16 (unsigned int a)
* Function creates array with elements of <fpvec<Type,B>; every element consists of a <fpvec<Type,B> register with all 0 except at position i
* 10000000
* 01000000
* 00100000
* 00010000
* ...
* Function automatically adjusts all sizes depending on the data type and the regSize parameter.
*/
template<typename T, int B>
std::array<fpvec<T, B>, (B/sizeof(T))> cvtu32_create_writeMask_Matrix() {
	Type zero = 0;
	Type one = 1;
	std::array<fpvec<Type, B>, (B/sizeof(T))> result {};

	#pragma unroll
	for(Type i = 0; i < (B/sizeof(T)); i++){
		auto tmp = fpvec<T,B>{};
		#pragma unroll
		for (int j=0; j<(B/sizeof(T)); j++) {
			tmp.elements[j] = zero;
		}
		tmp.elements[i] = one;
		result[i] = tmp;
	}
	return result;
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
*/
template<typename T, int B>
fpvec<T, B> mask_loadu(fpvec<T,B>& writeMask, uint32_t* data, uint32_t startIndex) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
template<typename T, int B>
fpvec<T,B> mask_cmpeq_epi32_mask(fpvec<T,B>& zeroMask, fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
template<typename T, int B>
fpvec<T,B> mask_add_epi32(fpvec<T,B>& src, fpvec<T,B>& writeMask, fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
* @param writeMask : if bit is set to "1" -> store related item from data into result array
* @param data : register-array which contains the data that should be stored
*/
template<typename T, int B>
void mask_storeu_epi32(uint32_t* result, uint32_t startIndex, fpvec<T,B>& writeMask, fpvec<T,B>& data) {
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (writeMask.elements[i] == 1) {
			// result[(startIndex+i)%HSIZE] = data.elements[i];
			result[startIndex+i] = data.elements[i];
		}
	}
}

/**	#9
* serial primitive for Intel Intrinsic:
* int _mm512_mask2int (__mmask16 k1)
* original description: "Converts bit mask k1 into an integer value, storing the results in dst."
* own (simplified implementation):
* 
* IMPORTANT: 	This is an adjustet implementation of the intrinsic mentioned above. 
				This solution is specially tailored to the logical flow of LinearProbing_v1 - v5 
				and its functionality is reduced to its necessities.
* @return 1 if at least 1 bit of mask is set;
* @return 0 if no bit of mask is set
*/
template<typename T, int B>
Type mask2int(fpvec<T,B>& mask) {
	Type res = 0;
	#pragma unroll	
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (mask.elements[i] == 1) {
			res = 1;
		}
	}
	return res;
}

/**	#9.1
* adaption of Intel Intrinsic:
* int _mm512_mask2int (__mmask16 k1)
* IMPORTANT: 	This is an own adapted implementation of _mm512_mask2int, which return an uint32_t value
*				as representation of overhanded mask.
*				This function can handle masks up to 32 elements!
*
* @return uint32 value as representation of overhanded mask
* @return 0 if no bit of mask is set
*/
template<typename T, int B>
uint32_t mask2int_uint32_t(fpvec<T,B>& mask) {
	uint32_t res = 0;
	#pragma unroll	
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (mask.elements[i] == 1) {
			res += pow(2,i);
		}
	}
	return res;
}

/**	#9.2
* adaption of Intel Intrinsic:
* int _mm512_mask2int (__mmask16 k1)
* IMPORTANT: 	This is an own adapted implementation of _mm512_mask2int, which return an uint64_t value
*				as representation of overhanded mask.
*				This function can handle masks up to 64 elements!
*
* @return uint64_t value as representation of overhanded mask
* @return 0 if no bit of mask is set
*/
template<typename T, int B>
uint64_t mask2int_uint64_t(fpvec<T,B>& mask) {
	uint64_t res = 0;
	#pragma unroll	
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (mask.elements[i] == 1) {
			res += pow(2,i);
		}
	}
	return res;
}

/**	#10
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_knot (__mmask16 a)
* original description: "Compute the bitwise NOT of 16-bit mask a, and store the result in k."
*/
template<typename T, int B>
fpvec<T,B> knot(fpvec<T,B>& src) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
* int __builtin_clz (unsigned int x)
* original description: "Built-in Function: int __builtin_clz (unsigned int x)
* Returns the number of leading 0-bits in x, starting at the most significant bit position. 
* If x is 0, the result is undefined."
*/
template<typename T, int B>
Type clz_onceBultin(fpvec<T,B>& src) {
	Type res = 0;
	#pragma unroll
	for (int i=((B/sizeof(T))-1); i>=0; i--) {
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
* @param templateMask : Register of type fpvec<T,B>
* @param data : array which contains the data which should be loaded
* @param startIndex : first index-position of data from where the data should be loaded
*/
template<typename T, int B>
fpvec<T,B> load_epi32(fpvec<T,B>& templateMask, uint32_t* data, uint32_t startIndex) {		// testen - fehlt Parameter <T> ?
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		reg.elements[i] = data[startIndex+i];
	}
	return reg;
}

/**	#13
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_cmpeq_epi32_mask (__m512i a, __m512i b)
* original description: "Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k."
*/
template<typename T, int B>
fpvec<T,B> cmpeq_epi32_mask(fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
template<typename T, int B>
fpvec<T,B> permutexvar_epi32(fpvec<T,B>& idx, fpvec<T,B>& a) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		T id = idx.elements[i];
		T value = a.elements[id];
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
template<typename T, int B>
Type ctz_onceBultin(fpvec<T,B>& src) {
	Type res = 0;
	#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
template<typename T, int B>
fpvec<T,B> mask_set1(fpvec<T,B>& src, fpvec<T,B>& writeMask, Type value) {
	auto reg = fpvec<T,B>{};
	#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
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
* @param result : array, in which the data is stored; function store 512, 1024, 1536 or 2048 bits
* @param startIndex : first index - position of data from where the data should be stored
* @param data : register-array which contains the data that should be stored
*/
template<typename T, int B>
void store_epi32(uint32_t* result, uint32_t startIndex, fpvec<T,B>& data) {
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		result[(startIndex+i)] = data.elements[i];
	}
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////// New functions for compile and execute on FPGA hardware //////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**	#18
* Old function - Functionality not compatible with parallel utilization of all 4 memory controllers!
* Own function to load 16/32/48/64 elements (= complete CL (register)) in one clock cycle from input array
*/
template<typename T, int B>
fpvec<T,B> load(T* p, int i_cnt) {
	const int i_localConst = i_cnt;
    auto reg = fpvec<T,B> {};
    #pragma unroll
    for (uint idx = 0; idx < (B/sizeof(T)); idx++) {
          reg.elements[idx] = p[idx + i_localConst * (B/sizeof(T))];
    }
    return reg;
}


/**	#19
* Own function to load 4*512bit (2048bit, 256 byte) (= complete CL (register)) in one clock cycle from input array
* Load complete CL (register) in one clock cycle (same for PCIe and DDR4) 
* Function is based on the approach of parallel load with all 4 memory controller 
*/
template<typename T, int B>
fpvec<T,B> maxLoad_per_clock_cycle(T* input, int i_cnt, size_t kNumLSUs, size_t kValuesPerLSU, const Type elementCount) {
	auto reg = fpvec<T,B> {};
	const int i_cnt_const = i_cnt;
	#pragma unroll
	for (size_t l = 0; l < kNumLSUs; l++) {
		#pragma unroll
		for (size_t k = 0; k < kValuesPerLSU; k++) {
					
			const int idx = (i_cnt_const*elementCount)
							+ (l*kValuesPerLSU)
							+ k;

			reg.elements[l*kValuesPerLSU+k] = input[idx];
		}
	}	
    return reg;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////// New functions to calculate overflow -  independant of elementCount //////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/**	#20
* Own function to create the overflow_correction_mask
*/
template<typename T, int B>
fpvec<T,B> createOverflowCorrectionMask(T oferflowUnsigned) {
	auto reg = fpvec<T,B>{};
	const int overflow = (B/sizeof(T)) - oferflowUnsigned;
	Type one = 1;
	Type zero = 0;
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (i<overflow) {
			reg.elements[i] = one;
		}
		else {
			reg.elements[i] = zero;
		}		
	}
	return reg;
} 

/**	#21
* Own function to create the cutlow_mask
*/
template<typename T, int B>
fpvec<T,B> createCutlowMask(T cutlowUnsigned) {
	auto reg = fpvec<T,B>{};
	const int cutlow_const = (B/sizeof(T)) - cutlowUnsigned;
	Type one = 1;
	Type zero = 0;
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (i<cutlow_const) {
			reg.elements[i] = zero;
		}
		else {
			reg.elements[i] = one;
		}		
	}
	return reg;
} 

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////  New functions for LinearProbing_v5 == soa_conflict_v1  ///////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

/**	#22
* serial primitive for Intel Intrinsic:
* __m512i _mm512_conflict_epi32 (__m512i a)
* original description: "Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst."
* 
* customized conflict_epi32 - function:
* This function check whether an element is already in the vector. 
* Only elements with a lower index are checked. 
* As a result, element 0 in the vector never has a conflict. The bits for each element are then set accordingly. 
* IMPORTANT: At the point where a conflict is found, the position of the first occurrence is written! 
* IMPORTANT: The position is specified from 1 to #elementCount (NOT 0-n-1) !!
*
* adjustment against original Intel Intrinsic:
* 104 71 106 116 82 128 75 109 42 78 59 44 115 124 100 71 --> 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 
* 104 71  71 116 82  71 75 109 42 78 59 44 115 124 100 71 --> 0 0 2 0 0 2 0 0 0 0 0 0 0 0 0 2 
*/
template<typename T, int B>
fpvec<T,B> conflict_epi32(fpvec<T,B>& a) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		Type currentElement = a.elements[i];
		for (int j=0; j<i; j++) {
			if(a.elements[j] == currentElement) {
				reg.elements[i] = (Type)(j+1);
				j=i;	
				break;			
			}
		}
	}	
	return reg;
}


/**	#23
* serial primitive for Intel Intrinsic:
* void _mm512_mask_compressstoreu_epi32 (void* base_addr, __mmask16 k, __m512i a)
* original description: "Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr."
*
*
* @param buffer : 
* @param writeMask : 
* @param data : 
*/
template<typename T, int B>
void mask_compressstoreu_epi32(Type* buffer, fpvec<T,B>& writeMask, fpvec<T,B>& data) {
	int buffer_position = 0;
	#pragma nounroll								// DO NOT UNROLL, because the steps are dependent on each other ?!
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (writeMask.elements[i] == 1) {
			buffer[buffer_position] = (Type)data.elements[i];
			buffer_position++;
		}
	}
}

/**	#24
* serial primitive for Built-in Function Provided by GCC:
* int __builtin_popcount(int number)
* original description: "This function is used to count the number of set bits in an unsigned integer. "
* 
* Adjustment: We don't hand over an integer, we handle a register directly within the function and count the "1" within this register.
* return: count of "!=0" within this register
*/
template<typename T, int B>
Type popcount_builtin(fpvec<T,B>& mask) {
	Type count = 0;
	#pragma unroll								
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (mask.elements[i] != 0) {
			count++;
		}
	}
	return count;
}

/**	#25
* adaption of:
* __m512i _mm512_set1_epi32 (int a)
*
* function create a fpvec<T,B> with all values zero; except at position (value-1) => 1
*/
template<typename T, int B>
fpvec<T,B> setX_singleValue(T value) {
	auto reg = fpvec<T,B>{};
	reg.elements[value-1] = 1;
	return reg;
}

/**	#26
* serial primitive for Intel Intrinsic:
* __m512i _mm512_mask_i32gather_epi32 (__m512i src, __mmask16 k, __m512i vindex, void const* base_addr, int scale)
* original description: "Gather 32-bit integers from memory using 32-bit indices. 32-bit elements are loaded from addresses starting 
* 	at base_addr and offset by each 32-bit element in vindex (each index is scaled by the factor in scale). Gathered elements are merged 
*	into dst using writemask k (elements are copied from src when the corresponding mask bit is not set). scale should be 1, 2, 4 or 8."
* 
* @param src : register of type fpvec<T,B>
* @param mask_k : writemask k (= register of type fpvec<T,B>)
* @param vindex : register of type fpvec<T,B> 
* @param data : void const* base_addr
* @param scale : scale should be 1, 2, 4 or 8
*/
template<typename T, int B>
fpvec<T,B> mask_i32gather_epi32(fpvec<T,B>& src, fpvec<T,B>& mask_k, fpvec<T,B>& vindex, uint32_t* data, int scale) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if(mask_k.elements[i] == (Type)1) {
			size_t addr = 0 + vindex.elements[i];	// * scale * 8;	
						// 0, because hashVec and countVec starting both at index 0
						// omit *8 (because we don't need bit conversion)
						// omit *scale, because we currently work with Type=uint32_t in all stages
						// if we want to use another datatype, we may adjust the scale paramter within
						// this function; now scale doesn't have an usage
			reg.elements[i] = data[addr];													
		} else {
			reg.elements[i] = src.elements[i];
		}
	}
	return reg;
}

/**	#27
* serial primitive for Intel Intrinsic:
* __m512i _mm512_maskz_add_epi32 (__mmask16 k, __m512i a, __m512i b)
* original description: "Add packed 32-bit integers in a and b, and store the results in dst using zeromask k 
* (elements are zeroed out when the corresponding mask bit is not set)."
*/
template<typename T, int B>
fpvec<T,B> maskz_add_epi32(fpvec<T,B>& writeMask, fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
	Type zero = 0;
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (writeMask.elements[i] == 1) {
			reg.elements[i] = a.elements[i] + b.elements[i];
		}
		else {
			reg.elements[i] = zero;
		}
	}
	return reg;
}

/**	#28
* serial primitive for Intel Intrinsic:
* void _mm512_mask_i32scatter_epi32 (void* base_addr, __mmask16 k, __m512i vindex, __m512i a, int scale)
* original description: "Scatter 32-bit integers from a into memory using 32-bit indices. 32-bit elements are stored at 
*		addresses starting at base_addr and offset by each 32-bit element in vindex (each index is scaled by the factor in scale) 
*		subject to mask k (elements are not stored when the corresponding mask bit is not set). scale should be 1, 2, 4 or 8."
* 
* @param datbaseStoragea : void const* base_addr for storage/scatter
* @param mask_k : writemask k (= register of type fpvec<T,B>)
* @param vindex : register of type fpvec<T,B> 
* @param data_to_scatter : register of type fpvec<T,B>
* @param scale : scale should be 1, 2, 4 or 8
* @param tmp_HSIZE : global HashSize (=size of hashVec and countVec) to avoid scatter over the vector borders through false offsets
*/
template<typename T, int B>
void mask_i32scatter_epi32(uint32_t* baseStorage, fpvec<T,B>& mask_k, fpvec<T,B>& vindex, fpvec<T,B>& data_to_scatter, int scale, Type tmp_HSIZE) {
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if(mask_k.elements[i] == (Type)1) {
			Type addr = 0 + vindex.elements[i];	
						// 0, because hashVec and countVec starting both at index 0
						// omit *8 (because we don't need bit conversion)
						// omit *scale, because we currently work with Type=uint32_t in all stages
						// if we want to use another datatype, we may adjust the scale paramter within
						// this function; now scale doesn't have an usage
			if (addr >= tmp_HSIZE) {
				addr = tmp_HSIZE-addr;
			}							
			baseStorage[addr] = (Type)data_to_scatter.elements[i];													
		} 
	}
}

/**	#29
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_kandn (__mmask16 a, __mmask16 b)
* original description: "Compute the bitwise NOT of (16/...)-bit masks a and then AND with b, and store the result in k."
*
* Note: registers a and b may only contain elements of the datatype Type (currently uint32_t) with values ONLY 1 or 0 !!
*/
template<typename T, int B>
fpvec<T,B> kAndn(fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (a.elements[i] == 0) {
			reg.elements[i] = 1;
			if (reg.elements[i] == b.elements[i]) {
				reg.elements[i] = 1;
			} else {
				reg.elements[i] = 0;
			}
		} else {
			reg.elements[i] = 0;
			if (reg.elements[i] == b.elements[i]) {
				reg.elements[i] = 1;
			} else {
				reg.elements[i] = 0;
			}
		}
	}
	return reg;
}

/**	#30
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_kandn (__mmask16 a, __mmask16 b)
* original description: "Compute the bitwise NOT of (16/...)-bit masks a and then AND with b, and store the result in k."
*
* Note: registers a and b may only contain elements of the datatype Type (currently uint32_t) with values ONLY 1 or 0 !!
*/
template<typename T, int B>
fpvec<T,B> kAnd(fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (a.elements[i] == b.elements[i]) {
			reg.elements[i] = (Type)1;
		} else {
			reg.elements[i] = (Type)0;
		}
	}
	return reg;
}

/**	#31
* serial primitive for Intel Intrinsic:
* __m512i _mm512_maskz_conflict_epi32 (__mmask16 k, __m512i a)
* original description: "Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k 
* 			(elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst."
* 
* customized maskz_conflict_epi32 - function:
* This function check whether an element is already in the vector. 
* Only elements with a lower index are checked. 
* As a result, element 0 in the vector never has a conflict. The bits for each element are then set accordingly. 
* IMPORTANT: At the point where a conflict is found, the position of the first occurrence is written! 
* IMPORTANT: The position is specified from 1 to #elementCount (NOT 0-n-1) !!
*
* adjustment against original Intel Intrinsic:
* 104 71 106 116 82 128 75 109 42 78 59 44 115 124 100 71 --> 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 
* 104 71  71 116 82  71 75 109 42 78 59 44 115 124 100 71 --> 0 0 2 0 0 2 0 0 0 0 0 0 0 0 0 2 
*
* Difference against conflict_epi32:	additional fpvec<T,B>& mask_k : if mask_k[i]==0 --> result[0]=0 ; else do conflict_epi32 algorithm
*
* We don't need this simplyfied maskz_conflict function anymore, we use maskz_conflict_ret_uint64_64elements() instead!
*/
template<typename T, int B>
fpvec<T,B> maskz_conflict_epi32(fpvec<T,B>& mask_k, fpvec<T,B>& a) {
	auto reg = fpvec<T,B>{};
	#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if(mask_k.elements[i] == 1) {
			Type currentElement = a.elements[i];
			for (int j=0; j<i; j++) {
				if(a.elements[j] == currentElement) {
					reg.elements[i] = (Type)(j+1);
					j=i;	
					break;			
				}
			}
		} else {
			reg.elements[i] = (Type)(0);
		}	
	}	
	return reg;
}

/**	#32
* serial primitive for Intel Intrinsic:
* __m512i _mm512_and_epi32 (__m512i a, __m512i b)
* original description: "Compute the bitwise AND of packed 32-bit integers in a and b, and store the results in dst."
*
* Note: Compute the AND (here simplified: proof of equality) of every element of two fpvec<T,B> registers.
* @return 1, if elements at this position are equal	
* @return 0, if elements at this position are unequal	
* type of return-value is depending on the data type within the registers
*/
template<typename T, int B>
fpvec<T,B> register_and(fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (a.elements[i] == b.elements[i]) {
			reg.elements[i] = (T)1;
		} else {
			reg.elements[i] = (T)0;
		}
	}
	return reg;
}

/**	#33
* serial primitive for Intel Intrinsic:
* __mmask16 _mm512_mask_cmp_epi32_mask (__mmask16 k1, __m512i a, __m512i b, _MM_CMPINT_ENUM imm8)
* original description: "Compare packed signed 32-bit integers in a and b based on the comparison operand specified by imm8,
* 	and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set)."
*
* NOTE: adjust function to handle ONLY the _MM_CMPINT_NLT (Not less than) comparison, due to the fact that this is the only scenario,
*		which is used in LinearProbing_v5 (SoA_conflict_v1); Thereby an additional parameter for the cmp type isn't necessary anymore.
*/
template<typename T, int B>
fpvec<T,B> mask_cmp_epi32_mask_NLT(fpvec<T,B>& zeroMask, fpvec<T,B>& a, fpvec<T,B>& b) {
	auto reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (zeroMask.elements[i] == 1) {
			if (a.elements[i] < b.elements[i]) {
				reg.elements[i] = 0;
			}	
			else {
				reg.elements[i] = 1;
			}
		}	
		else {
			reg.elements[i] = 0;
		}	
	}
	return reg;
}

/////////////////////////////////////////////////////////////
/////  heavily customized features specific to v5  //////////
/////////////////////////////////////////////////////////////


/**	#34
* serial primitive for Intel Intrinsic:
* __m512i _mm512_maskz_conflict_epi32 (__mmask16 k, __m512i a)
* original description: "Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k 
* 			(elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst."
* 
* customized maskz_conflict_ret_uint64_64elements - function:
* This function check whether an element is already in the vector. 
* Only elements with a lower index are checked. 
* As a result, element 0 in the vector never has a conflict. The bits for each element are then set accordingly.
* 
* example:
* index :							  0  1   2  3  4   5  6   7  8  9 10 11  12  13  14 15
* input array (e.g. loaded values):	104 71 106 82 82 128 75 109 82 94 59 44 115 124 100 94 
* result of conflict_epi32:			  0  0   0  0  8   0  0   0 24  0  0  0   0   0   0 512 
* example input[0..15]
* 1st conflict @ input[4] : 0 0 0 1 ..0 == 0 + 0 + 0 + 2^3 = 8
* 2st conflict @ input[8] : 0 0 0 1 1 ..0 = 0 + 0 + 0 + 2^3 + 2^4 = 24
* 3st conflict @ input[15]: 0 0 0 0 0 0 0 0 1 ..0 = 0 + .. + 0 + 2^9 = 512
*
* Difference against conflict_epi32:	additional fpvec<T,B>& mask_k : if mask_k[i]==0 --> result[0]=0 ; else do conflict_epi32 algorithm from above
*
* IMPORTANT: 	Adjustment of maskz_conflict_ret_uint64_64elements against: _mm512_maskz_conflict_epi32 : Due to the fact, that the input registers for this function
*				containing up to 64 32-bit-elements in our implementation for LinearProbing_V5, we need to return a fpvec<uint64_t,512> which give us the possibility to return
*				64-bit integers which can describe the real conflict situation for 64 positions.
* IMPORTANT: 	This solution is a function specially adapted to LinearProbing_v5 and only works within it as long as the different input and output data formats are observed! 
*				It is aimed to the current FPGA project, which works with 64-element data registers with 32-bit data elements (= 2048 bits per register)! 		
* working example config:
* @param1: fpvec<uint32_t,256>& mask_k
* @param2: fpvec<uint32_t,256>& a
* return: fpvec<uint64_t,512> 				
*/
template<typename T, int B>
fpvec<uint64_t,512> maskz_conflict_ret_uint64_64elements(fpvec<T,B>& mask_k, fpvec<T,B>& a) {
	auto reg = fpvec<uint64_t,512>{};
	#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if(mask_k.elements[i] == 1) {
			Type currentElement = a.elements[i];
			uint64_t conflict_calculation = 0;
			for (int j=0; j<i; j++) {
				if(a.elements[j] == currentElement) {
					conflict_calculation += pow(2,j);		
				}
				reg.elements[i] = (uint64_t)conflict_calculation;
			}
		} else {
			reg.elements[i] = (uint64_t)(0);
		}	
	}	
	return reg;
}

/**	#35
* adaption of Intel Intrinsic:
* __mmask16 _mm512_cmpeq_epi32_mask (__m512i a, __m512i b)
* original description: "Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k."
*
* IMPORTANT: 	This solution is a function specially adapted to LinearProbing_v5 and only works within it as long as the different input and output data formats are observed! 
*				It is aimed to the current FPGA project, which works with 64-element data registers with 32-bit data elements (= 2048 bits per register)! 		
*
* Compare every element of to registers of type fpvec<uint64_t,512> (64 64-bit elements), but return the result as fpvec<uint32_t,256> !
* @return uint32_t 1, if elements at this position are equal	
* @return uint32_t 0, if elements at this position are unequal	
* @return fpvec<uint32_t,256> with 0 or 1 as representation for equality-comparison for every position
*/
template<typename T, int B>
fpvec<T, B> cmpeq_epi64_reg_return_uint32_mask(fpvec<uint64_t,512>& a, fpvec<uint64_t,512>& b) {
	fpvec<uint32_t,256> reg = fpvec<T,B>{};
#pragma unroll
	for (int i=0; i<(B/sizeof(T)); i++) {
		if (a.elements[i] == b.elements[i]) {
			reg.elements[i] = (T)1;
		} else {
			reg.elements[i] = (T)0;
		}
	}
	return reg;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

#endif // PRIMITIVES_HPP