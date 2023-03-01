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
#include "global_settings.hpp"
#include "helper_kernel.hpp"
#include "primitives.hpp"

#include "lib/lib.hpp"
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	OVERVIEW about functions in kernel.cpp
//
//	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
// 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
//	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
//	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
// 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 
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
// constexpr size_t kPCIeWidth = PCIE_WIDTH;	// currently not used
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// declaration of the classes
class kernelV1;
class kernelV2;
class kernelV3;
class kernelV4;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 1 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant1(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         
	
	// recalculate iterations, because we must ierate through all data lines of input array
	// the input array contains dataSize lines 
	// per cycle we can load #(regSize/sizeof(Type)) elements
	// !! That means dataSize must be a multiple of (regSize/sizeof(Type)) !! 
	const size_t iterations =  loops;

	// ensure dataSize is nice
	assert(dataSize % elementCount == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);

	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	// NOTE: 	Due to current data loading approach, regSize must be 256 byte, so that
	//			every register has a overall size of 2048 bit so that it can be loaded in one cycle using the 4 memory controllers
	assert(regSize == 256);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	q.submit([&](handler& h) {
		h.single_task<kernelV1>([=]() [[intel::kernel_args_restrict]] {

		device_ptr<Type> input(arr_d);
		device_ptr<Type> hashVec(hashVec_d);
		device_ptr<Type> countVec(countVec_d);

		////////////////////////////////////////////////////////////////////////////////
		//// declare some basic masks and arrays
		
		Type one = 1;
		Type zero = 0;
		fpvec<Type, regSize> oneMask = set1<Type, regSize>(one);
		fpvec<Type, regSize> zeroMask = set1<Type, regSize>(zero);
		////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////

		// define dataVec register
		fpvec<Type, regSize> dataVec;

		// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
//#pragma nounroll
		for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {

			// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
			// dataVec = load<Type, regSize>(input, i_cnt);	

			// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
			dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, i_cnt, kNumLSUs, kValuesPerLSU, elementCount);

			/**
			* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
			* @param p current element of input data array
			**/ 	
			int p = 0;
//#pragma nounroll
			while (p < elementCount) {
			
				// get single value from current dataVec register at position p
				Type inputValue = dataVec.elements[p];
			
				// compute hash_key of the input value
				Type hash_key = hashx(inputValue,HSIZE);

				// broadcast inputValue into a SIMD register
				fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);

				while (1) {
					// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
					TypeSigned overflow = (hash_key + elementCount) - HSIZE;		
					overflow = overflow < 0? 0: overflow;
					Type oferflowUnsigned = (Type)overflow;		
			
					// use function createOverflowCorrectionMask() to create overflow correction mask
					fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

					// Load 16 consecutive elements from hashVec, starting from position hash_key
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
							hash_key +=elementCount;
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
 */
void LinearProbingFPGA_variant2(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         
	
	// recalculate iterations, because we must ierate through all data lines of input array
	// the input array contains dataSize lines 
	// per cycle we can load #(regSize/sizeof(Type)) elements
	// !! That means dataSize must be a multiple of (regSize/sizeof(Type)) !! 
	const size_t iterations =  loops;

	// ensure dataSize is nice
	assert(dataSize % elementCount == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);

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
			device_ptr<Type> hashVec(hashVec_d);
			device_ptr<Type> countVec(countVec_d);

			////////////////////////////////////////////////////////////////////////////////
			//// declare some basic masks and arrays
			Type one = 1;
			Type zero = 0;
			fpvec<Type, regSize> oneMask = set1<Type, regSize>(one);
			fpvec<Type, regSize> zeroMask = set1<Type, regSize>(zero);
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			// define dataVec register
			fpvec<Type, regSize> dataVec;

			// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
			#pragma nounroll
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
				
				// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
				// dataVec = load<Type, regSize>(input, i_cnt);	

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, i_cnt, kNumLSUs, kValuesPerLSU, elementCount);

				/**
				* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
				* @param p current element of input data array
				**/ 	
				int p = 0;
				#pragma nounroll
				while (p < elementCount) {
					// get single value from current dataVec register at position p
					Type inputValue = dataVec.elements[p];

					// compute hash_key of the input value
					Type hash_key = hashx(inputValue,HSIZE);

					// compute the aligned start position within the hashMap based the hash_key
					Type remainder = hash_key % elementCount; // should be equal to (hash_key/elementCount)*elementCount;
					Type aligned_start = hash_key - remainder;
					
					/**
					* broadcast element p of input[] to vector of type fpvec<uint32_t>
					* broadcastCurrentValue contains sixteen times value of input[i]
					**/
					fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);

					while(1) {
						// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
						TypeSigned overflow = (aligned_start + elementCount) - HSIZE;
						overflow = overflow < 0 ? 0 : overflow;
						Type oferflowUnsigned = (Type)overflow;

						// use function createOverflowCorrectionMask() to create overflow correction mask
						fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

						// Calculating a cutlow correction mask and a overflow_and_cutlow_mask 
						TypeSigned cutlow = elementCount - remainder; // should be in a range from 1 to elementCount
						Type cutlowUnsigned = (Type)cutlow;
						fpvec<Type, regSize> cutlow_mask = createCutlowMask<Type, regSize>(cutlowUnsigned);
						
						fpvec<Type, regSize> overflow_and_cutlow_mask = mask_cmpeq_epi32_mask(oneMask, cutlow_mask, overflow_correction_mask);

						// Load 16 consecutive elements from hashVec, starting from position hash_key
						fpvec<Type, regSize> nextElements = load_epi32(oneMask, hashVec, aligned_start);

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
								aligned_start +=elementCount;
								if(aligned_start >= HSIZE){
									aligned_start = 0;
								}
							} 
						}  
					}
						 
				}	
			}
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



/**
 * Variant 3 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant3(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         
	
	// recalculate iterations, because we must ierate through all data lines of input array
	// the input array contains dataSize lines 
	// per cycle we can load #(regSize/sizeof(Type)) elements
	// !! That means dataSize must be a multiple of (regSize/sizeof(Type)) !! 
	const size_t iterations =  loops;

	// ensure dataSize is nice
	assert(dataSize % elementCount == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);

	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	// NOTE: 	Due to current data loading approach, regSize must be 256 byte, so that
	//			every register has a overall size of 2048 bit so that it can be loaded in one cycle using the 4 memory controllers
	assert(regSize == 256);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	q.submit([&](handler& h) {
		h.single_task<kernelV3>([=]() [[intel::kernel_args_restrict]] {

			device_ptr<Type> input(arr_d);
			device_ptr<Type> hashVec(hashVec_d);
			device_ptr<Type> countVec(countVec_d);

			////////////////////////////////////////////////////////////////////////////////
			//// declare some basic masks and arrays
			Type one = 1;
			Type zero = 0;
			fpvec<Type, regSize> oneMask = set1<Type, regSize>(one);
			fpvec<Type, regSize> zeroMask = set1<Type, regSize>(zero);
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			// define dataVec register
			fpvec<Type, regSize> dataVec;

			// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
			#pragma nounroll
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
				
				// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
				// dataVec = load<Type, regSize>(input, i_cnt);	

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, i_cnt, kNumLSUs, kValuesPerLSU, elementCount);

				/**
				* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
				* @param p current element of input data array
				**/ 	
				int p = 0;
				#pragma nounroll
				while (p < elementCount) {

					// load #elementCount input values --> use the #elementCount elements of dataVec	/// !! change compared to the serial implementation !!
					fpvec<Type, regSize> iValues = dataVec;

					//iterate over the input values
					int i=0;
					while (i<elementCount) {

						// broadcast single value from input at postion i into a new SIMD register
						fpvec<Type, regSize> idx = set1<Type, regSize>((Type)i);
						fpvec<Type, regSize> broadcastCurrentValue = permutexvar_epi32(idx,iValues);

						Type inputValue = (Type)broadcastCurrentValue.elements[0];
						Type hash_key = hashx(inputValue,HSIZE);

						// compute the aligned start position within the hashMap based the hash_key
						Type remainder = hash_key % elementCount; // should be equal to (hash_key/elementCount)*elementCount;
						Type aligned_start = hash_key - remainder;
					
						while (1) {
							// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
							TypeSigned overflow = (aligned_start + elementCount) - HSIZE;
							overflow = overflow < 0 ? 0 : overflow;
							Type oferflowUnsigned = (Type)overflow;

							// use function createOverflowCorrectionMask() to create overflow correction mask
							fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

							// Calculating a cutlow correction mask and a overflow_and_cutlow_mask 
							TypeSigned cutlow = elementCount - remainder; // should be in a range from 1 to elementCount
							Type cutlowUnsigned = (Type)cutlow;
							fpvec<Type, regSize> cutlow_mask = createCutlowMask<Type, regSize>(cutlowUnsigned);
							
							fpvec<Type, regSize> overflow_and_cutlow_mask = mask_cmpeq_epi32_mask(oneMask, cutlow_mask, overflow_correction_mask);

							// Load 16 consecutive elements from hashVec, starting from position hash_key
							fpvec<Type, regSize> nextElements = load_epi32(oneMask, hashVec, aligned_start);
						
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
								
								// WE COULD DO THIS LIKE VARIANT ONE.
								// This would mean we wouldn't calculate the match pos since it is clear already.                
								// increase the counter in countVec
								countVec[aligned_start+matchPos]++;			
								i++;
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
									i++;
									break;
								}   
								else {    			               // CASE B2                    
									//aligned_start = (aligned_start+16) % HSIZE;
									// since we now use the overflow mask we can do this to change our position
									// we ALSO need to set the remainder to 0.
									remainder = 0;
									aligned_start +=elementCount;
									if(aligned_start >= HSIZE){
										aligned_start = 0;
									}
								}
							}
						}
					}
					p+=elementCount;	
				}	
			}
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////



/**
 * Variant 4 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 * @param hash_map_d vector of #m_HSIZE_v registers of type fpvec<Type, regSize> -> used to realize hash_map on FPGA
 * @param count_map_d vector of #m_HSIZE_v registers of type fpvec<Type, regSize> -> used to realize count_map on FPGA
 * @param m_elements_per_vector = elementCount
 * @param m_HSIZE_v = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
 * @param m_HSIZE = HSIZE
 */
void LinearProbingFPGA_variant4(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE, fpvec<Type, regSize> *hash_map_d, fpvec<Type, regSize> *count_map_d, size_t m_elements_per_vector, size_t m_HSIZE_v, size_t m_HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         
	
	// recalculate iterations, because we must ierate through all data lines of input array
	// the input array contains dataSize lines 
	// per cycle we can load #(regSize/sizeof(Type)) elements
	// !! That means dataSize must be a multiple of (regSize/sizeof(Type)) !! 
	const size_t iterations =  loops;

	// ensure dataSize is nice
	assert(dataSize % elementCount == 0);
	assert(dataSize % kValuesPerLSU == 0);
	assert(dataSize % kNumLSUs == 0);

	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	// NOTE: 	Due to current data loading approach, regSize must be 256 byte, so that
	//			every register has a overall size of 2048 bit so that it can be loaded in one cycle using the 4 memory controllers
	assert(regSize == 256);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	q.submit([&](handler& h) {
		h.single_task<kernelV4>([=]() [[intel::kernel_args_restrict]] {

			device_ptr<Type> input(arr_d);
			device_ptr<Type> hashVec(hashVec_d);
			device_ptr<Type> countVec(countVec_d);
			// use a vector with elements of type <fpvec<uint32_t> as structure "around" the registers (-> SoAoV)
			device_ptr<fpvec<Type, regSize>> hash_map(hash_map_d);
			device_ptr<fpvec<Type, regSize>> count_map(count_map_d);

			////////////////////////////////////////////////////////////////////////////////
			//// declare some basic masks and arrays
			Type one = 1;
			Type zero = 0;
			fpvec<Type, regSize> oneMask = set1<Type, regSize>(one);
			fpvec<Type, regSize> zeroMask = set1<Type, regSize>(zero);
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			// loading data. On the first exec this should result in only 0 vals.   
			for(size_t i = 0; i < m_HSIZE_v; i++){
				size_t h = i * m_elements_per_vector;

				hash_map[i] = load_epi32(oneMask, hashVec, h);
				count_map[i] = load_epi32(oneMask, countVec, h);
			}

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
			Type oferflowUnsigned = (m_HSIZE_v * m_elements_per_vector) - HSIZE;
			if (oferflowUnsigned > 0) {
				overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);
				value_bigger_distinctValues = (Type)(distinctValues+7); 	
				value_bigger_distinctValues_mask = set1<Type, regSize>(value_bigger_distinctValues);

				hash_map[m_HSIZE_v-1] = mask_set1(value_bigger_distinctValues_mask, overflow_correction_mask, zero);
				count_map[m_HSIZE_v-1] = set1<Type, regSize>(zero);
			}

			/** CREATING WRITING MASKS
			 * 
			 * Following line isn't needed anymore. Instead of zero_cvtu32_mask, please use zeroMask as mask with all 0 and elementCount elements!
			 * fpvec<uint32_t> zero_cvtu32_mask = cvtu32_mask16((uint32_t)0);	
			 *
			 *	old code for creating writing masks:
			 *	std::array<fpvec<uint32_t>, 16> masks {};
			 *	for(uint32_t i = 1; i <= 16; i++){
			 *		masks[i-1] = cvtu32_mask16((uint32_t)(1 << (i-1)));
			 *	}
			 *
			 * new solution is working with (variable) regSize and elementCount per register (e.g. 256 byte and 64 elements per register)
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

			// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
			#pragma nounroll
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
				// old load-operation; works with regSize of 64, 128, 256 byte, but isn't optimized regarding parallel load by 4 memory controller
				// dataVec = load<Type, regSize>(input, i_cnt);	

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				dataVec = maxLoad_per_clock_cycle<Type, regSize>(input, i_cnt, kNumLSUs, kValuesPerLSU, elementCount);

				/**
				 * iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
				 * @param p current element of input data array
				**/ 	
				int p = 0;
				while (p < elementCount) {
					Type inputValue = dataVec.elements[p];
					Type hash_key = hashx(inputValue,m_HSIZE_v);
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
								hash_key = (hash_key + 1) % m_HSIZE_v;
							}
						}
					}
				}				
			}
			// #######################################
			// #### END OF FPGA parallelized part ####
			// #######################################

			//store data from hash_ma
			for(size_t i = 0; i < m_HSIZE_v; i++){
				size_t h = i * m_elements_per_vector;
						
				store_epi32(hashVec, h, hash_map[i]);
				store_epi32(countVec, h, count_map[i]);
			}
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant4()
////////////////////////////////////////////////////////////////////////////////