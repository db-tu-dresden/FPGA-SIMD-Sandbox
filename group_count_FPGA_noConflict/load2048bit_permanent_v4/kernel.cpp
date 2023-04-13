#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "kernel.hpp"
#include "../config/global_settings.hpp"
#include "../helper/helper_kernel.hpp"
#include "../primitives/primitives.hpp"

#include "lib/lib.hpp"
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	OVERVIEW about functions in kernel.cpp
//
//	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
// 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
//	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
//	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
// 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of 
//																					process each item individually, even though it occurs multiple times in the currently loaded data		
// 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// Board globals. Can be changed from command line.
// default to values in pac_s10_usm BSP
#ifndef DDR_CHANNELS
#define DDR_CHANNELS 4
#endif

#ifndef DDR_WIDTH
#define DDR_WIDTH 64 // bytes (512 bits)
#endif

#ifndef PCIE_WIDTH
#define PCIE_WIDTH 64 // bytes (512 bits)
#endif

#ifndef DDR_INTERLEAVED_CHUNK_SIZE
#define DDR_INTERLEAVED_CHUNK_SIZE 4096 // bytes
#endif

constexpr size_t kDDRChannels = DDR_CHANNELS;		
constexpr size_t kDDRWidth = DDR_WIDTH;				
constexpr size_t kDDRInterleavedChunkSize = DDR_INTERLEAVED_CHUNK_SIZE;
// constexpr size_t kPCIeWidth = PCIE_WIDTH;
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// declaration of the classes
class kernelV4;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 4 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]				// global defined, not part of paramater list anymore
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])							// global defined, not part of paramater list anymore
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 * @param m_elements_per_vector_v4_2048bit = elements_per_register															// global defined, not part of paramater list anymore
 * @param m_HSIZE_v_v4_2048bit = (HSIZE + m_elements_per_vector_v4_2048bit - 1) / m_elements_per_vector_v4_2048bit;			// global defined, not part of paramater list anymore
 * @param HSIZE_hashMap_v4_v4_2048bit = m_elements_per_vector_v4_2048bit * m_HSIZE_v_v4_2048bit								// global defined, not part of paramater list anymore
 */
void LinearProbingFPGA_variant4(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, size_t size) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(int) == 0);
  	static_assert(kDDRInterleavedChunkSize % sizeof(int) == 0);							

	constexpr size_t kValuesPerInterleavedChunk = kDDRInterleavedChunkSize / sizeof(Type);
	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);		
	static_assert(kValuesPerInterleavedChunk % kValuesPerLSU == 0);

	constexpr size_t kNumLSUs = kDDRChannels;  
	constexpr size_t kIterationsPerChunk = kValuesPerInterleavedChunk / kValuesPerLSU;    

	// ensure size is nice
	assert(size % kValuesPerInterleavedChunk == 0);
	assert(size % kNumLSUs == 0);

	// ensure dataSize is nice
	assert(dataSize % elements_per_register == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);   

	size_t total_chunks = size / kValuesPerInterleavedChunk;
	size_t chunks_per_lsu = total_chunks / kNumLSUs;
	// calculation of iterations; value could be bigger than dataSize/elements_per_register
	const size_t iterations = chunks_per_lsu * kIterationsPerChunk;  
	
	/** 
	 * const size_t iterations =  loops;
	 * Update: We don't use this simple calculation of iterations anymore since we load data with 4 DMA controllers in parallel.
	 * Instead we use the iterations_calculated = 2.500.032 (our "simple" iterations=loops=2.500.000 would be smaller)
	 * This prevents the "losing" of some values at the end of the input array, which is caused by the fact that the four DMA controllers 
	 * only ever load from their own 4k pages. This leads to small offsets, which require a slightly higher number of iterations. 
	*/

	// ensure global defined regSize and inner_regSize is nice
	assert(regSize == 256);
	assert(inner_regSize == 64);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm
	q.submit([&](handler& h) {

		h.single_task<kernelV4>([=]() [[intel::kernel_args_restrict]] {

			device_ptr<Type> input(arr_d);
			device_ptr<Type> hashVec_globalMem(hashVec_d);
			device_ptr<Type> countVec_globalMem(countVec_d);

			////////////////////////////////////////////////////////////////////////////////
			//// declare some basic masks and arrays
			Type one = 1;
			Type zero = 0;
			fpvec<Type, regSize> oneMask = set1<Type, regSize>(one);
			fpvec<Type, regSize> zeroMask = set1<Type, regSize>(zero);
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////
			
			////////////////////////////////////////////////////////////////////////////////
			//// declare private variables for hashVec & countVec
			/* The Intel oneAPI DPC++/C++ Compiler creates a kernel memory in hardware.
			* Kernel memory is sometimes referred to as on-chip memory because it is created from
			* memory sources (such as RAM blocks) available on the FPGA.
			* 
			* Here we want to create the hashVec and CountVec Arrays inside the kernel with local Memory,
			* more accurate with M20K RAM Blocks. This memory type is significantly faster than store/load operations from/to global memory.
			* With this change, we only need to write every element of both arrays once to the global memory at the end of the algorithm.
			*  
			* In the ideal case, the compiler creates both data structures as stall-free. But that depends on whether the algorithm allows it or not.
			* Due to the fact that our HSIZE can also be significantly larger, we consciously use M20K RAM blocks instead of MLAB memory, 
			* since the STRATIX FPGA has approx. 10000 M20K blocks - which corresponds to approx. 20MB and is therefore better suited for larger data structures.
			*/
			// USING M20K RAM BLOCKS on FPGA to implement hashVec and countVec (embedded memory) and initialize these with zero
			[[intel::fpga_memory("BLOCK_RAM")]] std::array<fpvec<Type, regSize>, m_HSIZE_v_v4_2048bit> hash_map;
			[[intel::fpga_memory("BLOCK_RAM")]] std::array<fpvec<Type, regSize>, m_HSIZE_v_v4_2048bit> count_map;

			// loading data. On the first exec this should result in only 0 vals. / or better initalize hash_map and count_map with vectors full of 0
			#pragma unroll 16
			for(size_t i = 0; i < m_HSIZE_v_v4_2048bit; i++){
				hash_map[i] = zeroMask;
				count_map[i] = zeroMask;
			}
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			/**
			 * calculate overflow in last register of hash_map and count_map, to prevent errors from storing elements in hash_map[m_HSIZE_v-1] in positions that are >HSIZE 
			 *	
			 * due to this approach, the hash_map and count_map can have overall more slots than the value of HSIZE
			 * set value of positions of the last register that "overflows" to a value that is bigger than distinctValues
			 * These values can't be part of input data array (because inpute only have values between 1 and distinctValues), 
			 * but these slots will be handled as "no match, but position already filled" within the algorithm.
			 * Since only HSIZE values are stored at the end (back to hashVec and countVec), these slots/values are simply dropped at the end.
			 * This procedure avoids the error that the algorithm stores real values in positions that are not written back later. As a result, values were lost and the end result became incorrect.
			 */
			// define variables and register for overflow calculation
			fpvec<Type, regSize> overflow_correction_mask;
			Type value_bigger_distinctValues;
			fpvec<Type, regSize> value_bigger_distinctValues_mask;

			// caculate overflow and mark positions in last register that will be overflow the value of HSIZE
			Type oferflowUnsigned = (m_HSIZE_v_v4_2048bit * m_elements_per_vector_v4_2048bit) - HSIZE;
			if (oferflowUnsigned > 0) {
				overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);
				value_bigger_distinctValues = (Type)(distinctValues+7); 	
				value_bigger_distinctValues_mask = set1<Type, regSize>(value_bigger_distinctValues);

				hash_map[m_HSIZE_v_v4_2048bit-1] = mask_set1(value_bigger_distinctValues_mask, overflow_correction_mask, zero);
				count_map[m_HSIZE_v_v4_2048bit-1] = set1<Type, regSize>(zero);
			}

			/** CREATING WRITING MASKS
			 * 
			 * Following line isn't needed anymore. Instead of zero_cvtu32_mask, please use zeroMask as mask with all 0 and elements_per_register elements!
			 * fpvec<uint32_t> zero_cvtu32_mask = cvtu32_mask16((uint32_t)0);	
			 *
			 *	old code for creating writing masks:
			 *	std::array<fpvec<uint32_t>, 16> masks {};
			 *	for(uint32_t i = 1; i <= 16; i++){ masks[i-1] = cvtu32_mask16((uint32_t)(1 << (i-1))); }
			 *
			 * new solution is working with (variable) regSize and elements_per_register per register (e.g. 256 byte and 64 elements per register)
			 * It generates a matrix of the required size according to the parameters used.  
			 */
			[[intel::fpga_register]] std::array<fpvec<Type, regSize>, (regSize/sizeof(Type))> masks;
			masks = cvtu32_create_writeMask_Matrix<Type, regSize>();

			/**
			 * ! ATTENTION - changed indizes (compared to the first AVX512 implementation) !
			 * mask with only 0 => zero_cvtu32_mask
			 * masks = array of 16 masks respectively fpvec<uint32_t> with one 1 at unique positions 
			 *
			 * calculation of free position is reworked
			 * old approach: uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
			 * -> omit +1, because masks with only 0 at every position is outsourced to zero_cvtu32_mask --> zeroMask is used instead                
			 */

			// #########################################
			// #### START OF FPGA parallelized part ####
			// #########################################
			// define dataVec register
			fpvec<Type, regSize> dataVec;

			// iterate over input data with a SIMD register size of regSize bytes (elements_per_register elements)
			#pragma nounroll		// compiler should realize that this loop cannot be unrolled
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {

				// calculate chunk_idx and chunk_offset for current iteration step
				const int i_cnt_const = i_cnt;
				const int chunk_idx = i_cnt_const / kIterationsPerChunk;
				const int chunk_offset = i_cnt_const % kIterationsPerChunk;

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

				// iterate over input data / always step by step through the currently 16 (or #elements_per_register) loaded elements
				#pragma nounroll
				for(int p=0; p<elements_per_register; p++) {
					Type inputValue = dataVec.elements[p];
					Type hash_key = hashx(inputValue,m_HSIZE_v_v4_2048bit);
					fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);

// Due to the manipulated data, which we created within main.cpp, we can ignore the while(1) loop here, because there aren't any data conflicts
// We do this to meassure the maximum FPGA performance for our use case without negativ impacts through algorithm structure or data depenencies
// while (1) {
						// compare vector with broadcast value against vector with following elements for equality
						fpvec<Type, regSize> compareRes = cmpeq_epi32_mask(broadcastCurrentValue, hash_map[hash_key]);

						// found match
						if (mask2int(compareRes) != 0) {
							count_map[hash_key] = mask_add_epi32(count_map[hash_key], compareRes, count_map[hash_key], oneMask);
// break;
						} else { // no match found
							// deterime free position within register
							fpvec<Type, regSize> checkForFreeSpace = cmpeq_epi32_mask(zeroMask, hash_map[hash_key]);

// if(mask2int(checkForFreeSpace) != 0) {                // CASE B1   
								Type pos = ctz_onceBultin(checkForFreeSpace);
								//store key
								hash_map[hash_key] = mask_set1<Type, regSize>(hash_map[hash_key], masks[pos], inputValue);
								//set count to one
								count_map[hash_key] = mask_set1<Type, regSize>(count_map[hash_key], masks[pos], (Type)1);
// break;
/* 	}   else    { // CASE B2
		// hash_key = (hash_key + 1) % m_HSIZE_v;
		hash_key = (hash_key + 1);															
		if (hash_key >= m_HSIZE_v_v4_2048bit) {
			hash_key = hash_key-m_HSIZE_v_v4_2048bit;
		}	
	} */
						}
// }
				}				
			}
				
			// #######################################
			// #### END OF FPGA parallelized part ####
			// #######################################

			// store data from hash_map & count_map back to global memory	
			// memcpy(hashVec_globalMem, hash_map, HSIZE * sizeof(Type));
			// memcpy(countVec_globalMem, hash_map, HSIZE * sizeof(Type));		--> will be handled as for-loop with #pragma unroll through the compiler -> not working for large HSIZE
			for(size_t i = 0; i < m_HSIZE_v_v4_2048bit; i++){
				size_t h = i * m_elements_per_vector_v4_2048bit;
				store_epi32(hashVec_globalMem, h, hash_map[i]);
				store_epi32(countVec_globalMem, h, count_map[i]);
			}
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant4()
////////////////////////////////////////////////////////////////////////////////