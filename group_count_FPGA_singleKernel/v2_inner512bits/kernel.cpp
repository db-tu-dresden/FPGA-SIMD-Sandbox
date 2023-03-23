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
class kernelV2;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 2 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant2(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE, size_t size) {
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
	assert(dataSize % elementCount == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);   

	size_t total_chunks = size / kValuesPerInterleavedChunk;
	size_t chunks_per_lsu = total_chunks / kNumLSUs;
	// calculation of iterations; value will be bigger than dataSize/elementCount
	const size_t iterations = chunks_per_lsu * kIterationsPerChunk;  
	
	/** 
	 * recalculate iterations, because we must ierate through all data lines of input array
	 * the input array contains dataSize lines 
	 * per cycle we can load #(regSize/sizeof(Type)) elements
	 * !! That means dataSize must be a multiple of (regSize/sizeof(Type)) !! 
	 * 
	 * const size_t iterations =  loops;
	 * Update: We don't use this simple calculation of iterations anymore.
	 * Instead we use the iterations_calculated = 2.500.032 (our "simple" iterations=loops=2.500.000 would be smaller)
	 * This prevents the "losing" of some values at the end of the input array, which is caused by the fact that the four DDR memory controllers 
	 * only ever load from their own 4k pages. This leads to small offsets, which require a slightly higher number of iterations. 
	*/

	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	// NOTE: 	Due to current data loading approach, regSize must be 256 byte, so that
	//			every register has a overall size of 2048 bit so that it can be loaded in one cycle using the 4 memory controllers
	assert(regSize == 256);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	q.submit([&](handler& h) {
		h.single_task<kernelV2>([=]() [[intel::kernel_args_restrict]] {

			device_ptr<Type> input(arr_d);
			device_ptr<Type> hashVec_globalMem(hashVec_d);
			device_ptr<Type> countVec_globalMem(countVec_d);
			
			////////////////////////////////////////////////////////////////////////////////
			//// declare private variables for hashVec & countVec
			/* The Intel oneAPI DPC++/C++ Compiler creates a kernel memory in hardware.
			* Kernel memory is sometimes referred to as on-chip memory because it is created from
			* memory sources (such as RAM blocks) available on the FPGA.
			* 
			* Here we want to create the hashVec and CountVec Arrays inside the kernel with local Memory,
			* more accurate with MLABs. This memory type is significantly faster than store/load operations to global memory.
			* With this change, we only need to write every element of both arrays once to the global memory at the end of the algorithm.
			*  
			* In the ideal case, the compiler creates both data structures as stall-free. But that depends on whether the algorithm allows it or not.
			*/
			// USING local MLAB on FPGA for hashVec and countVec array
			[[intel::fpga_memory("MLAB") , intel::numbanks(1) , intel::bankwidth(1024) , intel::private_copies(16)]] Type hashVec[globalHSIZE] = {};
			[[intel::fpga_memory("MLAB") , intel::numbanks(1) , intel::bankwidth(1024) , intel::private_copies(16)]] Type countVec[globalHSIZE] = {}; 

			// USING local FPGA-RAM (result of declare these variables without additional attributes)
			//	Type hashVec[globalHSIZE] = {};
			//	Type countVec[globalHSIZE] = {};

			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			////////////////////////////////////////////////////////////////////////////////
			//// declare some basic masks and arrays
			Type one = 1;
			Type zero = 0;
			fpvec<Type, inner_regSize> oneMask = set1<Type, inner_regSize>(one);
			fpvec<Type, inner_regSize> zeroMask = set1<Type, inner_regSize>(zero);
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			// define dataVec register
			fpvec<Type, regSize> dataVec;

			// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
			// #pragma nounroll		// compiler should realize that this loop cannot be unrolled
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
				
				// calculate chunk_idx and chunk_offset for current iteration step
				const int i_cnt_const = i_cnt;
				const int chunk_idx = i_cnt_const / kIterationsPerChunk;
				const int chunk_offset = i_cnt_const % kIterationsPerChunk;

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

				
				// split loaded data into 4 "working register" á 512 bit to work over data segments of 512bits
				std::array<fpvec<Type, inner_regSize>, (inner_regSize/sizeof(Type))> workingData {};
				#pragma unroll
				for (int i=0; i<(regSize/inner_regSize); i++) {				// regSize/inner_regSize should be 4
					#pragma unroll
					for (int j=0; j<inner_elementCount; j++) {
						workingData[i].elements[j] = dataVec.elements[((i*inner_elementCount)+j)];
					}	
				}

				#pragma nounroll
				for (int i=0; i<(regSize/inner_regSize); i++) {				// regSize/inner_regSize should be 4
					// read 512-bit segments of loaded data and work through the algorithm with segments of only 512-bits
					fpvec<Type, inner_regSize> tmp_workingData = workingData[i];
				
					/**
					* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
					* @param p current element of input data array
					**/ 	
					// int p = 0;
					// while (p < inner_elementCount) {
					// #pragma nounroll		// compiler should realize that this loop cannot be unrolled
					for(int p=0; p<inner_elementCount; p++) {
						// get single value from current dataVec register at position p
						Type inputValue = tmp_workingData.elements[p];

						// compute hash_key of the input value
						Type hash_key = hashx(inputValue,HSIZE);

						// compute the aligned start position within the hashMap based the hash_key
						Type remainder = hash_key % inner_elementCount; // should be equal to (hash_key/inner_elementCount)*inner_elementCount;
						Type aligned_start = hash_key - remainder;
						
						/**
						* broadcast element p of input[] to vector of type fpvec<uint32_t>
						* broadcastCurrentValue contains sixteen times value of input[i]
						**/
						fpvec<Type, inner_regSize> broadcastCurrentValue = set1<Type, inner_regSize>(inputValue);

						while(1) {
							// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
							TypeSigned overflow = (aligned_start + inner_elementCount) - HSIZE;
							overflow = overflow < 0 ? 0 : overflow;
							Type oferflowUnsigned = (Type)overflow;

							// use function createOverflowCorrectionMask() to create overflow correction mask
							fpvec<Type, inner_regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, inner_regSize>(oferflowUnsigned);

							// Calculating a cutlow correction mask and a overflow_and_cutlow_mask 
							TypeSigned cutlow = inner_elementCount - remainder; // should be in a range from 1 to inner_elementCount
							Type cutlowUnsigned = (Type)cutlow;
							fpvec<Type, inner_regSize> cutlow_mask = createCutlowMask<Type, inner_regSize>(cutlowUnsigned);
							
							fpvec<Type, inner_regSize> overflow_and_cutlow_mask = mask_cmpeq_epi32_mask(oneMask, cutlow_mask, overflow_correction_mask);

							// Load 16 consecutive elements from hashVec, starting from position hash_key
							fpvec<Type, inner_regSize> nextElements = load_epi32<Type, inner_regSize>(hashVec, aligned_start);

							// compare vector with broadcast value against vector with following elements for equality
							fpvec<Type, inner_regSize> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
						
							/**
							* case distinction regarding the content of the mask "compareRes"
							* 
							* CASE (A):
							* inputValue does match one of the keys in nextElements (key match)
							* just increment the associated count entry in countVec
							**/ 
							if (mask2int(compareRes) != 0) {
								/**
								* old:
								* uint32_t matchPos = (32-clz_onceBultin(compareRes));
								* clz_onceBultin(compareRes) returns 16, if compareRes is 0 at every position 
								* used fix: calculate elements of used fpvev<> registers dynamically with (64/sizeof(uint32_t))
								* 
								* new
								* compute the matching position indicated by a one within the compareRes mask
								* the position can be calculated two ways.
								* example: 00010000 is our matching mask
								* we could count the leading zeros and get the position like 7 - leadingzeros
								* we calculate the trailing zeros and get the position implicitly 
								**/      
								Type matchPos = ctz_onceBultin(compareRes);

								// WE COULD DO THIS LIKE VARIANT ONE.
								// This would mean we wouldn't calculate the match pos since it is clear already.
								// increase the counter in countVec
								countVec[aligned_start+matchPos]++;
								// p++;
								break;
							}   
							else {
								/**
								* CASE (B): 
								* --> inputValue does NOT match any of the keys in nextElements (no key match)
								* --> compare "nextElements" with zero
								* CASE (B1):   resulting mask of this comparison is not 0
								*             --> insert inputValue into next possible slot       
								*                 
								* CASE (B2):  resulting mask of this comparison is 0
								*             --> no free slot in current 16-slot array
								*             --> load next +16 elements (add +16 to hash_key and re-iterate through while-loop without incrementing p)
								*             --> attention for the overflow of hashVec & countVec ! (% HSIZE, continuation at position 0)
								**/ 
								// checkForFreeSpace. A free space is indicated by 1.
								fpvec<Type, inner_regSize> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, zeroMask,nextElements);
								Type innerMask = mask2int(checkForFreeSpace);
								if(innerMask != 0) {                // CASE B1    
									//this does not calculate the correct position. we should rather look at trailing zeros.
									Type pos = ctz_onceBultin(checkForFreeSpace);

									hashVec[aligned_start+pos] = (uint32_t)inputValue;
									countVec[aligned_start+pos]++;
									// p++;
									break;
								}
								else {                   // CASE B2 
									//	aligned_start = (aligned_start+16) % HSIZE;
									// 	since we now use the overflow mask we can do this to change our position
									// 	we ALSO need to set the remainder to 0.  
									remainder = 0;
									aligned_start += inner_elementCount;
									if(aligned_start >= HSIZE){
										aligned_start = 0;
									}
								} 
							}  
						}					
					}	
				}	
			}

			//store results back to global memory
			// #pragma unroll
			// for(int i=0; i<globalHSIZE; i++) {hashVec_globalMem[i]=hashVec[i]; countVec_globalMem[i]=countVec[i]; }
			memcpy(hashVec_globalMem, hashVec, globalHSIZE * sizeof(Type));
 			memcpy(countVec_globalMem, countVec, globalHSIZE * sizeof(Type));
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////