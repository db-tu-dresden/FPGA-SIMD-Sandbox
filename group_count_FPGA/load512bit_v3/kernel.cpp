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
class kernelV3;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 3 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]			// global defined, not part of paramater list anymore
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])						// global defined, not part of paramater list anymore
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant3(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, size_t size) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(int) == 0);
  	static_assert(kDDRInterleavedChunkSize % sizeof(int) == 0);							

	constexpr size_t kValuesPerInterleavedChunk = kDDRInterleavedChunkSize / sizeof(Type);
	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);		
	static_assert(kValuesPerInterleavedChunk % kValuesPerLSU == 0);

	constexpr size_t kNumLSUs = kDDRChannels;  
	// constexpr size_t kIterationsPerChunk = kValuesPerInterleavedChunk / kValuesPerLSU;    

	// ensure size is nice
	assert(size % kValuesPerInterleavedChunk == 0);
	assert(size % kNumLSUs == 0);

	// ensure dataSize is nice
	assert(dataSize % elements_per_register == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);   

	// size_t total_chunks = size / kValuesPerInterleavedChunk;
	// size_t chunks_per_lsu = total_chunks / kNumLSUs;
	// calculation of iterations; value could be bigger than dataSize/elements_per_register
	// const size_t iterations = chunks_per_lsu * kIterationsPerChunk;  

	// recalculate iterations parameter, because we load only single 512bit register per iteration in this approach	
	const size_t iterations = size / elements_per_inner_register;

	// ensure global defined regSize and inner_regSize is nice
	assert(regSize == 256);
	assert(inner_regSize == 64);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	q.submit([&](handler& h) {
		h.single_task<kernelV3>([=]() [[intel::kernel_args_restrict]] {

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
			* more accurate with M20K RAM Blocks. This memory type is significantly faster than store/load operations from/to global memory.
			* With this change, we only need to write every element of both arrays once to the global memory at the end of the algorithm.
			*  
			* In the ideal case, the compiler creates both data structures as stall-free. But that depends on whether the algorithm allows it or not.
			* Due to the fact that our HSIZE can also be significantly larger, we consciously use M20K RAM blocks instead of MLAB memory, 
			* since the STRATIX FPGA has approx. 10000 M20K blocks - which corresponds to approx. 20MB and is therefore better suited for larger data structures.
			*/
			// USING M20K RAM BLOCKS on FPGA to implement hashVec and countVec (embedded memory) and initialize these with zero
			[[intel::fpga_memory("BLOCK_RAM")]] std::array<Type, HSIZE> hashVec;
			[[intel::fpga_memory("BLOCK_RAM")]] std::array<Type, HSIZE> countVec;

			#pragma unroll 16		
			for(int i=0; i<HSIZE; i++) {
				hashVec[i]=0; 
				countVec[i]=0;	
			}
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
			fpvec<Type, inner_regSize> iValues;

			// iterate over input data with a SIMD register size of 512bit (inner_regSize bytes => elements_per_inner_register(=16) elements)
			for(int i_cnt=0; i_cnt<iterations; i_cnt++) {

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				iValues = load<Type, inner_regSize>(input, i_cnt);

				// remove first while loop
				// Due to the parallel processing of the entire register (16 elements), 
				// the outer while loop is no longer necessary, since the elements are not processed individually.
				// int p = 0;
				// while (p < elements_per_inner_register) {

				// int p = 0;
				// while (p < elements_per_inner_register) {
				// #pragma nounroll		// compiler should realize that this loop cannot be unrolled
				for(int k=0; k<elements_per_inner_register; k++) {

					// broadcast single value from input at postion i into a new SIMD register
					fpvec<Type, inner_regSize> idx = set1<Type, inner_regSize>((Type)k);
					fpvec<Type, inner_regSize> broadcastCurrentValue = permutexvar_epi32(idx,iValues);

					Type inputValue = (Type)broadcastCurrentValue.elements[0];
					Type hash_key = hashx(inputValue, HSIZE);

					// compute the aligned start position within the hashMap based the hash_key
					Type remainder = hash_key % elements_per_inner_register; // should be equal to (hash_key/elements_per_inner_register)*elements_per_inner_register;
					Type aligned_start = hash_key - remainder;
						
					while (1) {
						// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
						TypeSigned overflow = (aligned_start + elements_per_inner_register) - HSIZE;
						overflow = overflow < 0 ? 0 : overflow;
						Type oferflowUnsigned = (Type)overflow;

						// use function createOverflowCorrectionMask() to create overflow correction mask
						fpvec<Type, inner_regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, inner_regSize>(oferflowUnsigned);

						// Calculating a cutlow correction mask and a overflow_and_cutlow_mask 
						TypeSigned cutlow = elements_per_inner_register - remainder; // should be in a range from 1 to elements_per_inner_register
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
							// compute the matching position indicated by a one within the compareRes mask
							// the position can be calculated two ways.
							// example: 00010000 is our matching mask
							// we could count the leading zeros and get the position like 7 - leadingzeros
							// we calculate the trailing zeros and get the position implicitly 
							Type matchPos = ctz_onceBultin(compareRes); 
									
							// WE COULD DO THIS LIKE VARIANT ONE.
							// This would mean we wouldn't calculate the match pos since it is clear already.                
							// increase the counter in countVec
							countVec[aligned_start+matchPos]++;			
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
							fpvec<Type, inner_regSize> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, zeroMask, nextElements);
							Type innerMask = mask2int(checkForFreeSpace);
							if(innerMask != 0) {                // CASE B1    
								//this does not calculate the correct position. we should rather look at trailing zeros.
								Type pos = ctz_onceBultin(checkForFreeSpace);
									
								hashVec[aligned_start+pos] = (Type)inputValue;
								countVec[aligned_start+pos]++;				
								break;
							}   
							else {    			               // CASE B2                    
								//aligned_start = (aligned_start+16) % HSIZE;
								// since we now use the overflow mask we can do this to change our position
								// we ALSO need to set the remainder to 0.
								remainder = 0;
								aligned_start += elements_per_inner_register;
								if(aligned_start >= HSIZE){
									aligned_start = 0;
								}
							}
						}
					}
				}	
			}

			// store results back to global memory
			// memcpy(hashVec_globalMem, hashVec, HSIZE * sizeof(Type));
			// memcpy(countVec_globalMem, countVec, HSIZE * sizeof(Type));		--> will be handled as for-loop with #pragma unroll through the compiler -> not working for large HSIZE
			// we can't use #pragma unroll, due to unknown value of HSIZE 
			// -> High value of HSIZE in combination with pragma unroll can cause HIGH RAM UITLIZATION (~199%)
			// Because we know, that we are working often with 16 elements per register (16x32bit=512bit), we unroll with factor 16
			#pragma unroll 16					
			for(int i=0; i<HSIZE; i++) {
				hashVec_globalMem[i]=hashVec[i]; 
				countVec_globalMem[i]=countVec[i]; 	
			}
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////