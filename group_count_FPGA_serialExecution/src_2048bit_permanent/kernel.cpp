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
#include <cassert>
#include <stdexcept>

#include "kernel.hpp"
#include "../config/global_settings.hpp"
#include "../helper/helper_kernel.hpp"
#include "../primitives/primitives.hpp"

#define EMPTY_SPOT 0
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
constexpr size_t kPCIeWidth = PCIE_WIDTH;
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// declaration of the classes
class kernelV1;
class kernelV2;
class kernelV3;
class kernelV4;
class kernelV5;
////////////////////////////////////////////////////////////////////////////////
//// declare some basic masks and arrays
	Type one = 1;
	Type zero = 0;
	fpvec<Type, regSize> oneMask = set1<Type, regSize>(one);
	fpvec<Type, regSize> zeroMask = set1<Type, regSize>(zero);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 1 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]	// global defined, not part of paramater list anymore
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])				// global defined, not part of paramater list anymore
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant1(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size) {
	/** 
	 * define example register
	 * fpvec<uint32_t> testReg;
	 * 
	 * example function call from primitives.hpp
	 * testReg = cvtu32_mask16(n);
	 **/
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
	// calculation of iterations; value will be bigger than dataSize/elements_per_register
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

	// define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elements_per_register elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {

		// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
		// dataVec = load<Type, regSize>(input, i_cnt);	

		// calculate chunk_idx and chunk_offset for current iteration step
		const int i_cnt_const = i_cnt;
		const int chunk_idx = i_cnt_const / kIterationsPerChunk;
		const int chunk_offset = i_cnt_const % kIterationsPerChunk;

		// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
		dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

		/**
		* iterate over input data / always step by step through the currently 16 (or #elements_per_register) loaded elements
		* @param p current element of input data array
		**/ 	
		int p = 0;
		while (p < elements_per_register) {
		
			// get single value from current dataVec register at position p
			Type inputValue = dataVec.elements[p];

			// compute hash_key of the input value
			Type hash_key = hashx(inputValue,HSIZE);

			// broadcast inputValue into a SIMD register
			fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);

			while (1) {
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				TypeSigned overflow = (hash_key + elements_per_register) - HSIZE;		
				overflow = overflow < 0? 0: overflow;
				Type oferflowUnsigned = (Type)overflow;		

				// throw exception, if calculated overflow is bigger than amount of elements within the register
				// This case can only result from incorrectly configured global parameters.
				if (oferflowUnsigned > elements_per_register) {
					throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elements_per_register - no valid values / range");
				}
					
				// use function createOverflowCorrectionMask() to create overflow correction mask
				fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

				// Load #elements_per_register consecutive elements from hashVec, starting from position hash_key
				fpvec<Type, regSize> nextElements = mask_loadu(oneMask, hashVec, hash_key);

				// compare vector with broadcast value against vector with following elements for equality
				fpvec<Type, regSize> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);

				/**
				* case distinction regarding the content of the mask "compareRes"
				* 
				* CASE (A):
				* inputValue does match one of the keys in nextElements (key match)
				* just increment the associated count entry in countVec
				**/ 
				if ((mask2int(compareRes)) != 0) {    // !=0, because own function returns only 0 if any bit is zero
					// load cout values from the corresponding location                
					fpvec<Type, regSize> nextCounts = mask_loadu(oneMask, countVec, hash_key);
					
					// increment by one at the corresponding location
					nextCounts = mask_add_epi32(nextCounts, compareRes, nextCounts, oneMask);
							
					// selective store of changed value
					mask_storeu_epi32(countVec, hash_key, compareRes,nextCounts);
					p++;
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
					fpvec<Type, regSize> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_correction_mask, zeroMask, nextElements);
					Type innerMask = mask2int(checkForFreeSpace);
					if(innerMask != 0) {                // CASE B1    
						//compute position of the emtpy slot   
						Type pos = ctz_onceBultin(checkForFreeSpace);
						// use 
						hashVec[hash_key+pos] = (uint32_t)inputValue;
						countVec[hash_key+pos]++;
						p++;
						break;
					} 
					else {         			          // CASE B2   
						hash_key += elements_per_register;			
						if(hash_key >= HSIZE){
							hash_key = 0;
						}							
					}
				}
			} 
		}
	}
}   
//// end of LinearProbingFPGA_variant1()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 2 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]	// global defined, not part of paramater list anymore
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])				// global defined, not part of paramater list anymore
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant2(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size) {
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
	// calculation of iterations; value will be bigger than dataSize/elements_per_register
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
//// starting point of the logic of the algorithm

	// define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elements_per_register elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
	
		// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
		// dataVec = load<Type, regSize>(input, i_cnt);	

		// calculate chunk_idx and chunk_offset for current iteration step
		const int i_cnt_const = i_cnt;
		const int chunk_idx = i_cnt_const / kIterationsPerChunk;
		const int chunk_offset = i_cnt_const % kIterationsPerChunk;

		// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
		dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

		/**
		* iterate over input data / always step by step through the currently 16 (or #elements_per_register) loaded elements
		* @param p current element of input data array
		**/ 	
		int p = 0;
		while (p < elements_per_register) {

			// get single value from current dataVec register at position p
			Type inputValue = dataVec.elements[p];

			// compute hash_key of the input value
			Type hash_key = hashx(inputValue,HSIZE);

			// compute the aligned start position within the hashMap based the hash_key
			Type remainder = hash_key % elements_per_register; // should be equal to (hash_key/elements_per_register)*elements_per_register;
			Type aligned_start = hash_key - remainder;

			/**
			* broadcast element p of input[] to vector of type fpvec<uint32_t>
			* broadcastCurrentValue contains sixteen times value of input[i]
			**/
			fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);

			while(1) {
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				TypeSigned overflow = (aligned_start + elements_per_register) - HSIZE;
				overflow = overflow < 0 ? 0 : overflow;
				Type oferflowUnsigned = (Type)overflow;

				// throw exception, if calculated overflow is bigger than amount of elements within the register
				// This case can only result from incorrectly configured global parameters.
				if (oferflowUnsigned > elements_per_register) {
					throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elements_per_register - no valid values / range");
				}

				// use function createOverflowCorrectionMask() to create overflow correction mask
				fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

				// Calculating a cutlow correction mask and a overflow_and_cutlow_mask 
				TypeSigned cutlow = elements_per_register - remainder; // should be in a range from 1 to elements_per_register
				Type cutlowUnsigned = (Type)cutlow;
				fpvec<Type, regSize> cutlow_mask = createCutlowMask<Type, regSize>(cutlowUnsigned);
				
				fpvec<Type, regSize> overflow_and_cutlow_mask = mask_cmpeq_epi32_mask(oneMask, cutlow_mask, overflow_correction_mask);
	
				// Load 16 consecutive elements from hashVec, starting from position hash_key
				fpvec<Type, regSize> nextElements = load_epi32<Type, regSize>(hashVec, aligned_start);

				// compare vector with broadcast value against vector with following elements for equality
				fpvec<Type, regSize> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);

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
					//uint32_t matchPos = ((64/sizeof(uint32_t))-clz_onceBultin(compareRes)); // old variant

					// WE COULD DO THIS LIKE VARIANT ONE.
					// This would mean we wouldn't calculate the match pos since it is clear already.
					// increase the counter in countVec
					countVec[aligned_start+matchPos]++;
					p++;		
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
					fpvec<Type, regSize> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, zeroMask,nextElements);
					Type innerMask = mask2int(checkForFreeSpace);
					if(innerMask != 0) {                // CASE B1    
						//this does not calculate the correct position. we should rather look at trailing zeros.
						Type pos = ctz_onceBultin(checkForFreeSpace);

						hashVec[aligned_start+pos] = (uint32_t)inputValue;
						countVec[aligned_start+pos]++;
						p++;
						break;
					}
					else {                   // CASE B2 
					//aligned_start = (aligned_start+16) % HSIZE;
					// since we now use the overflow mask we can do this to change our position
					// we ALSO need to set the remainder to 0.  
						remainder = 0;
						aligned_start += elements_per_register;			
						if(aligned_start >= HSIZE){
							aligned_start = 0;
						}							
					} 
				}  
			}
		} 
	}   
}  
//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 3 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]	// global defined, not part of paramater list anymore
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])				// global defined, not part of paramater list anymore
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant3(uint32_t* input, uint32_t* hashVec, uint32_t* countVec, size_t size) {
////////////////////////////////////////////////////////////////////////////////
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
	// calculation of iterations; value will be bigger than dataSize/elements_per_register
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

    // define dataVec register
	fpvec<Type, regSize> iValues;

	// iterate over input data with a SIMD register size of regSize bytes (elements_per_register elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
		
		// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
		// dataVec = load<Type, regSize>(input, i_cnt);	

		// calculate chunk_idx and chunk_offset for current iteration step
		const int i_cnt_const = i_cnt;
		const int chunk_idx = i_cnt_const / kIterationsPerChunk;
		const int chunk_offset = i_cnt_const % kIterationsPerChunk;

		// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
		iValues = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

		/**
		* iterate over input data / always step by step through the currently 16 (or #elements_per_register) loaded elements
		* @param p current element of input data array
		**/ 	

		// remove first while loop
		// Due to the parallel processing of the entire register (16 elements), 
		// the outer while loop is no longer necessary, since the elements are not processed individually.
		// int p = 0;
		// while (p < elements_per_register) {
		// 		load #inner_elements_per_register input values --> use the #elements_per_register elements of dataVec	/// !! change compared to the serial implementation !!
		// 		old: fpvec<Type, regSize> iValues = dataVec;
		// 		commented out, since unnecessary double assignment; direct assignment on line 176

		//iterate over the input values
		int k=0;
		while (k<elements_per_register) {

			// broadcast single value from input at postion k into a new SIMD register
			fpvec<Type, regSize> idx = set1<Type, regSize>((Type)k);
			fpvec<Type, regSize> broadcastCurrentValue = permutexvar_epi32(idx,iValues);

			Type inputValue = (Type)broadcastCurrentValue.elements[0];
			Type hash_key = hashx(inputValue,HSIZE);

			// compute the aligned start position within the hashMap based the hash_key
			Type remainder = hash_key % elements_per_register; // should be equal to (hash_key/elements_per_register)*elements_per_register;
			Type aligned_start = hash_key - remainder;
			
			while (1) {
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				TypeSigned overflow = (aligned_start + elements_per_register) - HSIZE;
				overflow = overflow < 0 ? 0 : overflow;
				Type oferflowUnsigned = (Type)overflow;

				// throw exception, if calculated overflow is bigger than amount of elements within the register
				// This case can only result from incorrectly configured global parameters.
				if (oferflowUnsigned > elements_per_register) {
					throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elements_per_register - no valid values / range");
				}

				// use function createOverflowCorrectionMask() to create overflow correction mask
				fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

				// Calculating a cutlow correction mask and a overflow_and_cutlow_mask 
				TypeSigned cutlow = elements_per_register - remainder; // should be in a range from 1 to elements_per_register
				Type cutlowUnsigned = (Type)cutlow;
				fpvec<Type, regSize> cutlow_mask = createCutlowMask<Type, regSize>(cutlowUnsigned);
					
				fpvec<Type, regSize> overflow_and_cutlow_mask = mask_cmpeq_epi32_mask(oneMask, cutlow_mask, overflow_correction_mask);

				// Load 16 consecutive elements from hashVec, starting from position hash_key
				fpvec<Type, regSize> nextElements = load_epi32<Type, regSize>(hashVec, aligned_start);
					
				// compare vector with broadcast value against vector with following elements for equality
				fpvec<Type, regSize> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
		
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
					
					//WE COULD DO THIS LIKE VARIANT ONE.
					//  This would mean we wouldn't calculate the match pos since it is clear already.                
					// increase the counter in countVec
					countVec[aligned_start+matchPos]++;
					k++;
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
					fpvec<Type, regSize> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, zeroMask, nextElements);
					Type innerMask = mask2int(checkForFreeSpace);
					if(innerMask != 0) {                // CASE B1    
						//this does not calculate the correct position. we should rather look at trailing zeros.
						Type pos = ctz_onceBultin(checkForFreeSpace);
								
						hashVec[aligned_start+pos] = (Type)inputValue;
						countVec[aligned_start+pos]++;
						k++;
						break;
					}   
					else {    			               // CASE B2                    
						//aligned_start = (aligned_start+16) % HSIZE;
						// since we now use the overflow mask we can do this to change our position
						// we ALSO need to set the remainder to 0.
						remainder = 0;
						aligned_start += elements_per_register;
						if(aligned_start >= HSIZE){
							aligned_start = 0;
						}
					}
				}
			}
		}
		// delete following line with deactivating outer while-loop
		// p+=elements_per_register;		
	}	
}
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 4 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]	// global defined, not part of paramater list anymore
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])				// global defined, not part of paramater list anymore
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant4(uint32_t* input, uint32_t* hashVec, uint32_t* countVec, size_t size) {
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
	// calculation of iterations; value will be bigger than dataSize/elements_per_register
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

	//// declare the basic hash- and count-map structure for this approach and some function intern variables
    fpvec<Type, regSize>* hash_map;
    fpvec<Type, regSize>* count_map;

    // use a vector with elements of type <fpvec<uint32_t> as structure "around" the registers
    hash_map = new fpvec<Type, regSize>[m_HSIZE_v_v4_2048bit];
    count_map = new fpvec<Type, regSize>[m_HSIZE_v_v4_2048bit];

    // loading data. On the first exec this should result in only 0 vals.   
    for(size_t i = 0; i < m_HSIZE_v_v4_2048bit; i++){
        size_t h = i * m_elements_per_vector_v4_2048bit;

       	hash_map[i] = load_epi32<Type, regSize>(hashVec, h);
    	count_map[i] = load_epi32<Type, regSize>(countVec, h);
	}

	/**
	 * calculate overflow in last register of hash_map and count_map, to prevent errors from storing elements in hash_map[m_HSIZE_v_v4_2048bit-1] in positions that are >HSIZE 
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

    // creating writing masks
	/** CREATING WRITING MASKS
	 * 
	 * Following line isn't needed anymore. Instead of zero_cvtu32_mask, please use zeroMask as mask with all 0 and elements_per_register elements!
	 * fpvec<uint32_t> zero_cvtu32_mask = cvtu32_mask16((uint32_t)0);	
	 *
	 *	old code for creating writing masks:
     *	std::array<fpvec<uint32_t>, 16> masks {};
     *	for(uint32_t i = 1; i <= 16; i++){
     *		masks[i-1] = cvtu32_mask16((uint32_t)(1 << (i-1)));
	 *	}
	 *
	 * new solution is working with (variable) regSize and elements_per_register per register (e.g. 256 byte and 64 elements per register)
	 * It generates a matrix of the required size according to the parameters used.  
	*/
    std::array<fpvec<Type, regSize>, (regSize/sizeof(Type))> masks {};
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
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {

		// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
		// dataVec = load<Type, regSize>(input, i_cnt);	

		// calculate chunk_idx and chunk_offset for current iteration step
		const int i_cnt_const = i_cnt;
		const int chunk_idx = i_cnt_const / kIterationsPerChunk;
		const int chunk_offset = i_cnt_const % kIterationsPerChunk;

		// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
		dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

	    /**
		 * iterate over input data / always step by step through the currently 16 (or #elements_per_register) loaded elements
		 * @param p current element of input data array
		**/ 	
    	int p = 0;
		while (p < elements_per_register) {
			Type inputValue = dataVec.elements[p];
			Type hash_key = hashx(inputValue,m_HSIZE_v_v4_2048bit);
			fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);

			while(1) {

				// compare vector with broadcast value against vector with following elements for equality
				fpvec<Type, regSize> compareRes = cmpeq_epi32_mask(broadcastCurrentValue, hash_map[hash_key]);

				// found match
				if (mask2int(compareRes) != 0) {
					count_map[hash_key] = mask_add_epi32(count_map[hash_key], compareRes, count_map[hash_key], oneMask);
					p++;
					break;
				} else { // no match found
					// deterime free position within register
					fpvec<Type, regSize> checkForFreeSpace = cmpeq_epi32_mask(zeroMask, hash_map[hash_key]);

					if(mask2int(checkForFreeSpace) != 0) {                // CASE B1   
						Type pos = ctz_onceBultin(checkForFreeSpace);
						//store key
						hash_map[hash_key] = mask_set1<Type, regSize>(hash_map[hash_key], masks[pos], inputValue);
						//set count to one
						count_map[hash_key] = mask_set1<Type, regSize>(count_map[hash_key], masks[pos], (Type)1);
						p++;
						break;
					}   else    { // CASE B2
						hash_key = (hash_key + 1) % m_HSIZE_v_v4_2048bit;
					}
				}
			}
		}
	}
	// #######################################
	// #### END OF FPGA parallelized part ####
	// #######################################

    //store data from hash_ma
    for(size_t i = 0; i < m_HSIZE_v_v4_2048bit; i++){
		size_t h = i * m_elements_per_vector_v4_2048bit;
				
        store_epi32(hashVec, h, hash_map[i]);
        store_epi32(countVec, h, count_map[i]);
    }
}
//// end of LinearProbingFPGA_variant4()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////