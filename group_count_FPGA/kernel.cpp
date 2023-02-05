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

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "primitives.hpp"
#include "kernel.hpp"
#include "helper_kernel.cpp"

#include "lib/lib.hpp"

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
class LinearProbingFPGA_variant1;
class LinearProbingFPGA_variant2;
class LinearProbingFPGA_variant3;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 1 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant1(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, long *out_d, uint64_t dataSize, uint64_t HSIZE, size_t size) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config) & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(int) == 0);
	static_assert(kDDRInterleavedChunkSize % sizeof(int) == 0);

	constexpr size_t kValuesPerInterleavedChunk = kDDRInterleavedChunkSize / sizeof(int);
	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(int);
	static_assert(kValuesPerInterleavedChunk % kValuesPerLSU == 0);

	constexpr size_t kNumLSUs = kDDRChannels;
	constexpr size_t kIterationsPerChunk = kValuesPerInterleavedChunk / kValuesPerLSU;
	
	// ensure size is nice
	assert(size % kValuesPerInterleavedChunk == 0);
	assert(size % kNumLSUs == 0);

	size_t total_chunks = size / kValuesPerInterleavedChunk;
	size_t chunks_per_lsu = total_chunks / kNumLSUs;
	size_t iterations = chunks_per_lsu * kIterationsPerChunk; 
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

// recalculate iterations, because we must ierate through all data lines of input array.
// the input array contains dataSize lines 
// per cycle we can load 16 elements
// !! That means dataSize must be a multiple of 16 !! 
	iterations =  (dataSize / 16);
	assert(dataSize % 16 == 0);

	q.submit([&](handler& h) {
		h.single_task<kernel>([=]() [[intel::kernel_args_restrict]] {

		device_ptr<uint32_t> input(arr_d);
		device_ptr<uint32_t> hashVec(hashVec_d);
		device_ptr<uint32_t> countVec(countVec_d);
		device_ptr<long> out(out_d);

		////////////////////////////////////////////////////////////////////////////////
		//// declare some basic masks and arrays
		uint32_t one = 1;
		uint32_t zero = 0;
		fpvec<uint32_t> oneMask = set1(one);
		fpvec<uint32_t> zeroMask = set1(zero);
		////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////

		out[0] = 0;

			// define two registers
			fpvec<uint32_t> dataVec;
// not used fpvec<uint32_t> resVec;

			// iterate over input data with a SIMD registers size of 512-bit (16 elements)
			#pragma unroll 1
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
				// Load complete CL (register) in one clock cycle
				dataVec = load<uint32_t>(input, i_cnt);

				/**
				* iterate over input data / always step by step through the currently 16 loaded elements
				* @param p current element of input data array
				**/ 	
				int p = 0;
				while (p < 16) {
						// get single value from input at position p
						uint32_t inputValue = input[p];

						// compute hash_key of the input value
						uint32_t hash_key = hashx(inputValue,HSIZE);

						// broadcast inputValue into a SIMD register
						fpvec<uint32_t> broadcastCurrentValue = set1(inputValue);

						while (1) {
							// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
							int32_t overflow = (hash_key + 16) - HSIZE;
							overflow = overflow < 0? 0: overflow;
							uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
							fpvec<uint32_t> overflow_correction_mask = cvtu32_mask16(overflow_correction_mask_i);

							// Load 16 consecutive elements from hashVec, starting from position hash_key
							fpvec<uint32_t> nextElements = mask_loadu(oneMask, hashVec, hash_key, HSIZE);

							// compare vector with broadcast value against vector with following elements for equality
							fpvec<uint32_t> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);

							/**
							* case distinction regarding the content of the mask "compareRes"
							* 
							* CASE (A):
							* inputValue does match one of the keys in nextElements (key match)
							* just increment the associated count entry in countVec
							**/ 
							if ((mask2int(compareRes)) != 0) {    // !=0, because own function returns only 0 if any bit is zero
								// load cout values from the corresponding location                
								fpvec<uint32_t> nextCounts = mask_loadu(oneMask, countVec, hash_key, HSIZE);
							
								// increment by one at the corresponding location
								nextCounts = mask_add_epi32(nextCounts, compareRes, nextCounts, oneMask);
							
								// selective store of changed value
								mask_storeu_epi32(countVec, hash_key, HSIZE, compareRes,nextCounts);
// out[0]++; only for testing
								out[0]++;
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
								fpvec<uint32_t> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_correction_mask, zeroMask, nextElements);
								uint32_t innerMask = mask2int(checkForFreeSpace);
								if(innerMask != 0) {                // CASE B1    
									//compute position of the emtpy slot   
									uint32_t pos = ctz_onceBultin(checkForFreeSpace);

									// use 
									hashVec[hash_key+pos] = (uint32_t)inputValue;
									countVec[hash_key+pos]++;
// out[0]++; only for testing									
									out[0]++;
									p++;
									break;
								} 
								else {         			          // CASE B2   
									hash_key += 16;
									if(hash_key >= HSIZE){
										hash_key = 0;
									}
								}
							}
						} 
				}	
			}
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant1()
////////////////////////////////////////////////////////////////////////////////


/**
 * Variant 2 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
/* 
void LinearProbingFPGA_variant2(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, long *out_v2_d, uint64_t dataSize, uint64_t HSIZE, size_t size) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config) & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(int) == 0);
	static_assert(kDDRInterleavedChunkSize % sizeof(int) == 0);

	constexpr size_t kValuesPerInterleavedChunk = kDDRInterleavedChunkSize / sizeof(int);
	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(int);
	static_assert(kValuesPerInterleavedChunk % kValuesPerLSU == 0);

	constexpr size_t kNumLSUs = kDDRChannels;
	constexpr size_t kIterationsPerChunk = kValuesPerInterleavedChunk / kValuesPerLSU;
	
	// ensure size is nice
	assert(size % kValuesPerInterleavedChunk == 0);
	assert(size % kNumLSUs == 0);

	size_t total_chunks = size / kValuesPerInterleavedChunk;
	size_t chunks_per_lsu = total_chunks / kNumLSUs;
	size_t iterations = chunks_per_lsu * kIterationsPerChunk; 
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm




}  
//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////
*/
/**
 * Variant 3 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param arr the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
 /*
void LinearProbingFPGA_variant3(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, long *out_v3_d, uint64_t dataSize, uint64_t HSIZE, size_t size) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config) & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(int) == 0);
	static_assert(kDDRInterleavedChunkSize % sizeof(int) == 0);

	constexpr size_t kValuesPerInterleavedChunk = kDDRInterleavedChunkSize / sizeof(int);
	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(int);
	static_assert(kValuesPerInterleavedChunk % kValuesPerLSU == 0);

	constexpr size_t kNumLSUs = kDDRChannels;
	constexpr size_t kIterationsPerChunk = kValuesPerInterleavedChunk / kValuesPerLSU;
	
	// ensure size is nice
	assert(size % kValuesPerInterleavedChunk == 0);
	assert(size % kNumLSUs == 0);

	size_t total_chunks = size / kValuesPerInterleavedChunk;
	size_t chunks_per_lsu = total_chunks / kNumLSUs;
	size_t iterations = chunks_per_lsu * kIterationsPerChunk; 
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm


  
}
*/
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////