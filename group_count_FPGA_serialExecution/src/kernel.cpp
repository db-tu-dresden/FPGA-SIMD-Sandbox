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
#include "global_settings.hpp"
#include "helper_kernel.hpp"
#include "primitives.hpp"

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
	fpvec<uint64_t,512> zeroMask_64bit_64elements = set1<uint64_t, 512>((uint64_t)0);
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
void LinearProbingFPGA_variant1(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE) {
	/** 
	 * define example register
	 * fpvec<uint32_t> testReg;
	 * 
	 * example function call from primitives.hpp
	 * testReg = cvtu32_mask16(n);
	 **/
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         

	const size_t iterations = loops;

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

	// define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
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

				// throw exception, if calculated overflow is bigger than amount of elements within the register
				// This case can only result from incorrectly configured global parameters.
				if (oferflowUnsigned > elementCount) {
					throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elementCount - no valid values / range");
				}
					
				// use function createOverflowCorrectionMask() to create overflow correction mask
				fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);

				// Load #elementCount consecutive elements from hashVec, starting from position hash_key
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
						hash_key += elementCount;			
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
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant2(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         

	const size_t iterations = loops;

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
//// starting point of the logic of the algorithm

	// define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
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

				// throw exception, if calculated overflow is bigger than amount of elements within the register
				// This case can only result from incorrectly configured global parameters.
				if (oferflowUnsigned > elementCount) {
					throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elementCount - no valid values / range");
				}

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
						aligned_start += elementCount;			
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
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant3(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         

	const size_t iterations = loops;

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

    // define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
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

			// load (regSize/sizeof(Type)) input values --> use the 16 elements of dataVec	/// !! change compared to the serial implementation !!
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

					// throw exception, if calculated overflow is bigger than amount of elements within the register
					// This case can only result from incorrectly configured global parameters.
					if (oferflowUnsigned > elementCount) {
						throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elementCount - no valid values / range");
					}

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
						
						//WE COULD DO THIS LIKE VARIANT ONE.
						//  This would mean we wouldn't calculate the match pos since it is clear already.                
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
							aligned_start += elementCount;
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
}
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 4 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant4(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         

	const size_t iterations = loops;

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

	//// declare the basic hash- and count-map structure for this approach and some function intern variables
    fpvec<Type, regSize>* hash_map;
    fpvec<Type, regSize>* count_map;

	const size_t m_elements_per_vector = elementCount; 			// should be equivalent to (regSize)/sizeof(Type);		
	const size_t m_HSIZE_v = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
	const size_t m_HSIZE = HSIZE;
 

    // use a vector with elements of type <fpvec<uint32_t> as structure "around" the registers
    hash_map = new fpvec<Type, regSize>[m_HSIZE_v];
    count_map = new fpvec<Type, regSize>[m_HSIZE_v];

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

    // creating writing masks
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
}
//// end of LinearProbingFPGA_variant4()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 5 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant5(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global board settings (regarding DDR4 config), global parameters & calculate iterations parameter
	static_assert(kDDRWidth % sizeof(Type) == 0);							

	constexpr size_t kValuesPerLSU = kDDRWidth / sizeof(Type);				
	constexpr size_t kNumLSUs = kDDRChannels;         

	const size_t iterations = loops;

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

	Type *buffer = reinterpret_cast< Type* >( _mm_malloc( elementCount * sizeof(Type), regSize ) );

	size_t p = 0;
	
	// AVX512-implementation of this LinearProbing-algorithm_v5 (SoA_conflict_v1) was actually using this while-loop.
	// We replaced this solution through our for-loop for the loading cycles similar to the previous versions.
	// while(p + elementCount < dataSize){

	// #########################################
	// #### START OF FPGA parallelized part ####
	// #########################################
	// define dataVec register
	fpvec<Type, regSize> input_value;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {

		// load the to aggregate data
		// fpvec<Type, regSize> input_value = load_epi32(oneMask, input, p, HSIZE);		// not used anymore

		// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
		input_value = maxLoad_per_clock_cycle<Type, regSize>(input, i_cnt, kNumLSUs, kValuesPerLSU, elementCount);	

		// how much the given count should be increased for the given input.
		fpvec<Type, regSize> input_add = set1<Type, regSize>(one);

		// search for conflicts
		fpvec<Type, regSize> conflicts = conflict_epi32(input_value);
		// masked to indicate were there is a conflict in the input_values and were not.
		fpvec<Type, regSize> no_conflicts_mask = cmpeq_epi32_mask(zeroMask, conflicts);
		fpvec<Type, regSize> negativ_no_conflicts_mask = knot(no_conflicts_mask);

		// we need to store the conflicts so we can interprete them as masks. and access them.
		// we are only interested in the enties that are not zero. That means the conflict cases.
		mask_compressstoreu_epi32(buffer, negativ_no_conflicts_mask, conflicts);
		size_t conflict_count = popcount_builtin(negativ_no_conflicts_mask);
		// add at all the places where the conflict masks indicates that there is an overlap
		for(size_t i = 0; i < conflict_count; i++){
			fpvec<Type, regSize> tmp_buffer_mask = setX_singleValue<Type, regSize>(buffer[i]);
			input_add = mask_add_epi32<Type, regSize>(input_add, tmp_buffer_mask, input_add, oneMask);
		}

		// we override the value and what to add with zero in the positions where we have a conflict.
		// NOTE: This steps might not be necessary.
		input_value = mask_set1(input_value, negativ_no_conflicts_mask, zero);
		input_add = mask_set1(input_add, negativ_no_conflicts_mask, zero);

		// now we can calculate the hashes.
		// for this we can store the input_value hash it and load it
		// OR we use the input and hash it save it in to buffer and than make a maskz load for the hashed data
		// OR we have a simdifyed Hash Algorithm! For the most cases we would need an avx... mod. 
		// _mm512_store_epi32(buffer, input_value);
		for(size_t i = 0; i < elementCount; i++){
			// old : buffer[i] = hashx(input[p + i], HSIZE);
			// we don't need this offset-calculation (p+i), because we iterate through our data-register (input_value), which
			// will be loaded with new data in every data-loading-iteration. So we just have to iterate through the elements within this register. 
			buffer[i] = hashx(input_value.elements[i], HSIZE);
		}

		fpvec<Type, regSize> hash_map_position = mask_loadu(no_conflicts_mask, buffer, (Type)0); // these are the hash values

		do{
			// now we can gather the data from the different positions where we have no conflicts.
			fpvec<Type, regSize> hash_map_value = mask_i32gather_epi32(zeroMask, no_conflicts_mask, hash_map_position, hashVec, 4);
			// with these we can calculate the different possible hits. Real hits and empty positions.
			fpvec<Type, regSize> foundPos = mask_cmpeq_epi32_mask(no_conflicts_mask, input_value, hash_map_value);
			fpvec<Type, regSize> foundEmpty = mask_cmpeq_epi32_mask(no_conflicts_mask, zeroMask, hash_map_value);

			/**
			 * convert variable HSIZE to 32-bit
			 * we don't need HSIZE as 64-bit integer, but this datatype cost much ressources on FPGA
			 * @ TODO change datatype of HSIZE global to "TYPE"
			 * @ TODO delete this conversion and use optimized HSIZE variable, when optimization above is done.  
			*/ 
			Type tmp_HSIZE = (Type)HSIZE;

			if(mask2int(foundPos) != 0){		//A
				// Now we have to gather the count. IMPORTANT! the count is a 32bit integer. 
				// FOR NOW THIS IS CORRECT BUT MIGHT CHANGE LATER!
				// For 64bit integers we would need to find a different solution!
				fpvec<Type, regSize> hash_map_value = mask_i32gather_epi32(zeroMask, foundPos, hash_map_position, countVec, 4);
				// on this count we can know add the pre calculated values. and scatter it back to their positions
				hash_map_value = maskz_add_epi32(foundPos, hash_map_value, input_add);
				mask_i32scatter_epi32<Type, regSize>(countVec, foundPos, hash_map_position, hash_map_value, 4, tmp_HSIZE);
					
				// finaly we remove the entries we just saved from the no_conflicts_mask such that the work to be done shrinkes.
				no_conflicts_mask = kAndn(foundPos, no_conflicts_mask);
			}

			if(mask2int(foundEmpty) != 0){		//B1
				// now we have to check for conflicts to prevent two different entries to write to the same position.
				fpvec<uint64_t, 512> saveConflicts = maskz_conflict_ret_uint64_64elements(foundEmpty, hash_map_position);

				// with the adjusted function maskz_conflict_ret_uint64_64elements, we don't need the procedure of register_and(saveConflicts, empty); anymore
				// fpvec<uint64_t, 512> empty = set1<uint64_t, 512>(mask2int_uint64_t(foundEmpty));
				// saveConflicts = register_and(saveConflicts, empty);
				
				fpvec<Type, regSize> to_save_data = cmpeq_epi64_reg_return_uint32_mask<Type, regSize>(zeroMask_64bit_64elements, saveConflicts);

				to_save_data = kAnd(to_save_data, foundEmpty);

				// with the cleaned mask we can now save the data.
				mask_i32scatter_epi32<Type, regSize>(hashVec, to_save_data, hash_map_position, input_value, 4, tmp_HSIZE);
				mask_i32scatter_epi32<Type, regSize>(countVec, to_save_data, hash_map_position, input_add, 4, tmp_HSIZE);

				//and again we need to remove the data from the todo list
				no_conflicts_mask = kAndn(to_save_data, no_conflicts_mask);
			}

			// afterwards we add one on the current positions of the still to be handled values.
			hash_map_position = maskz_add_epi32(no_conflicts_mask, hash_map_position, oneMask);

			// Since there isn't a modulo operation we have to check if the values are bigger or equal the HSIZE AND IF we have to set them to zero
			fpvec<Type, regSize> tmp_HSIZE_mask = set1<Type, regSize>(tmp_HSIZE);
			fpvec<Type, regSize> tobig = mask_cmp_epi32_mask_NLT(no_conflicts_mask, hash_map_position, tmp_HSIZE_mask);
			hash_map_position = mask_set1(hash_map_position, tobig, (Type)0);

			// we repeat this for one vector as long as their is still a value to be saved.
		}while(mask2int(no_conflicts_mask) !=0);
		p += elementCount;
	}	
	// #######################################
	// #### END OF FPGA parallelized part ####
	// #######################################

	//scalar remainder
    while(p < dataSize){
        int error = 0;
        // get the possible possition of the element.
        Type hash_key = hashx(input[p], HSIZE);
        
        while(1){
            // get the value of this position
            Type value = hashVec[hash_key];
            
            // Check if it is the correct spot
            if(value == input[p]){
                countVec[hash_key]++;
                break;
            
            // Check if the spot is empty
            }else if(value == EMPTY_SPOT){
                hashVec[hash_key] = input[p];
                countVec[hash_key] = 1;
                break;
            
            }
            else{
                //go to the next spot
                hash_key = (hash_key + 1) % HSIZE;
                //we assume that the hash_table is big enough
            }
        }
        p++;
    }	
    // multiple improvements are possible:
    // 1.   we could increase the performance of the worst case first write.
    // 2.   we could absorbe the scalar remainder with overflow masks
    // these would probably have a negative impact on  the overall performance.
}
//// end of LinearProbingFPGA_variant5()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////