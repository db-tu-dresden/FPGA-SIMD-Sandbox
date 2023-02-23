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

////////////////////////////////////////////////////////////////////////////////
//// declaration of the classes
class kernelV1;
class kernelV2;
class kernelV3;

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
//// Check global parameters & calculate iterations parameter
	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	assert((regSize == 64) || (regSize == 128) || (regSize == 256));
	size_t iterations =  loops;
	assert(dataSize % elementCount == 0);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	// define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
/*std::cout<<"content of hashVec | countVec: "<<std::endl;
for (int i=0; i<HSIZE; i++) {
	std::cout << hashVec[i] << "  |  " << countVec[i]<<std::endl;
} 	
std::cout<<" "<<std::endl;
		
std::cout<<"================================================"<<std::endl;
std::cout<<"===ATTENTION for-loop for iterations: i_cnt "<<i_cnt<<std::endl;		
*/		// Load complete CL (register) in one clock cycle
		dataVec = load<Type, regSize>(input, i_cnt);
/*
std::cout<<"dataVec_Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << dataVec.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
*/
		/**
		* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
		* @param p current element of input data array
		**/ 	
		int p = 0;
		while (p < elementCount) {
//std::cout<<"===ATTENTION while-loop for elements of register: p "<<p<<std::endl;			
			// get single value from current dataVec register at position p
			Type inputValue = dataVec.elements[p];
//std::cout<<"inputValue "<<inputValue<<std::endl;			
			// compute hash_key of the input value
			Type hash_key = hashx(inputValue,HSIZE);
//std::cout<<"hash_key "<<hash_key<<std::endl;
			// broadcast inputValue into a SIMD register
			fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);
/*
std::cout<<"broadcastCurrentValue_Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << broadcastCurrentValue.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
*/			while (1) {
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				int32_t overflow = (hash_key + 16) - HSIZE;
//std::cout<<"overflow "<<overflow<<std::endl;					
				overflow = overflow < 0? 0: overflow;
//std::cout<<"overflow "<<overflow<<std::endl;					
				uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
//std::cout<<"overflow_correction_mask_i "<<overflow_correction_mask_i<<std::endl;				 
			//	fpvec<Type, regSize> overflow_correction_mask = cvtu32_mask16<Type, regSize>(overflow_correction_mask_i);
/*
std::cout<<"overflow_correction_mask OLD: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << overflow_correction_mask.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
*/
///////////////////////////////////////////////////////////////////////
				TypeSigned overflow_NEW = (hash_key + elementCount) - HSIZE;
//std::cout<<"overflow_NEW "<<overflow_NEW<<std::endl;					
				overflow_NEW = overflow_NEW < 0? 0: overflow_NEW;
//std::cout<<"overflow_NEW "<<overflow_NEW<<std::endl;
				Type oferflowUnsigned_NEW = (Type)overflow_NEW;		
//std::cout<<"oferflowUnsigned_NEW "<<oferflowUnsigned_NEW<<std::endl;	
				
				if (oferflowUnsigned_NEW > elementCount) {
					throw std::out_of_range("Value of oferflowUnsigned_NEW is bigger than elementCount - no valid range");
				}
				

				fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned_NEW);
/*
std::cout<<"overflow_correction_mask NEW: "<<std::endl;				
for (int i=0; i<elementCount; i++) {
	std::cout << overflow_correction_mask_NEW.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
*/


/////////////////////////////////////////////////////////////////////
// std::cout<<" "<<std::endl;
// std::cout<<"inputValue "<<inputValue<<std::endl;	
// std::cout<<"hash_key "<<hash_key<<std::endl;		

				// Load 16 consecutive elements from hashVec, starting from position hash_key
				fpvec<Type, regSize> nextElements = mask_loadu(oneMask, hashVec, hash_key, HSIZE);
/*
std::cout << "nextElements ";
for (int i=0; i<elementCount; i++) {
		std::cout << nextElements.elements[i] << " ";
	} 	
std::cout<<" "<<std::endl;
*/
				// compare vector with broadcast value against vector with following elements for equality
				fpvec<Type, regSize> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
/*			
std::cout << "compareRes ";
for (int i=0; i<elementCount; i++) {
		std::cout << compareRes.elements[i] << " ";
	} 	
	std::cout<<" "<<std::endl;
*/
				/**
				* case distinction regarding the content of the mask "compareRes"
				* 
				* CASE (A):
				* inputValue does match one of the keys in nextElements (key match)
				* just increment the associated count entry in countVec
				**/ 
				if ((mask2int(compareRes)) != 0) {    // !=0, because own function returns only 0 if any bit is zero
					// load cout values from the corresponding location                
					fpvec<Type, regSize> nextCounts = mask_loadu(oneMask, countVec, hash_key, HSIZE);
					
					// increment by one at the corresponding location
					nextCounts = mask_add_epi32(nextCounts, compareRes, nextCounts, oneMask);
						
					// selective store of changed value
					mask_storeu_epi32(countVec, hash_key, HSIZE, compareRes,nextCounts);
					p++;
//std::cout<<"say hello1 "<<std::endl;
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
/*
std::cout << "checkForFreeSpace ";
for (int i=0; i<elementCount; i++) {
		std::cout << checkForFreeSpace.elements[i] << " ";
	} 	
std::cout<<" "<<std::endl;		
*/			
					Type innerMask = mask2int(checkForFreeSpace);
					if(innerMask != 0) {                // CASE B1    
						//compute position of the emtpy slot   
						Type pos = ctz_onceBultin(checkForFreeSpace);
						// use 
						hashVec[hash_key+pos] = (uint32_t)inputValue;
						countVec[hash_key+pos]++;
// std::cout<<"hash_key "<<hash_key<<std::endl;		
// std::cout<<"pos "<<pos<<std::endl;		
// std::cout<<"inputValue "<<inputValue<<std::endl;		
						p++;
//std::cout<<"say hello2 "<<std::endl;
						break;
					} 
					else {         			          // CASE B2   
						hash_key += elementCount;
// std::cout<<"B2 hash_key before "<<hash_key<<std::endl;	
						
						if(hash_key >= HSIZE){
							hash_key = 0;
						}
// std::cout<<"B2 hash_key after "<<hash_key<<std::endl;							
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
//// Check global parameters & calculate iterations parameter
	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	assert((regSize == 64) || (regSize == 128) || (regSize == 256));
	size_t iterations =  loops;
	assert(dataSize % elementCount == 0);
////////////////////////////////////////////////////////////////////////////////

	// define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
/*std::cout<<"content of hashVec | countVec: "<<std::endl;
for (int i=0; i<HSIZE; i++) {
	std::cout << hashVec[i] << "  |  " << countVec[i]<<std::endl;
} 	
std::cout<<" "<<std::endl;
		
std::cout<<"================================================"<<std::endl;
std::cout<<"===ATTENTION for-loop for iterations: i_cnt "<<i_cnt<<std::endl;*/		
		// Load complete CL (register) in one clock cycle
		dataVec = load<Type, regSize>(input, i_cnt);

/*std::cout<<"dataVec_Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << dataVec.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;*/
		/**
		* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
		* @param p current element of input data array
		**/ 	
		int p = 0;
		while (p < elementCount) {
//std::cout<<"===ATTENTION while-loop for elements of register: p "<<p<<std::endl;			
			// get single value from current dataVec register at position p
			Type inputValue = dataVec.elements[p];
//std::cout<<"inputValue "<<inputValue<<std::endl;	
			// compute hash_key of the input value
			Type hash_key = hashx(inputValue,HSIZE);
//std::cout<<"hash_key "<<hash_key<<std::endl;
			// compute the aligned start position within the hashMap based the hash_key

/*			
			Type aligned_start = (hash_key/16)*16;
			Type remainder = hash_key - aligned_start; // should be equal to hash_key % elementCount
std::cout<<"aligned_start "<<aligned_start<<std::endl;
std::cout<<"remainder "<<remainder<<std::endl;	
*/
////// NEW //////
Type remainder = hash_key % elementCount; // should be equal to (hash_key/elementCount)*elementCount;
Type aligned_start = hash_key - remainder;
//std::cout<<"aligned_start_new  "<<aligned_start_new <<std::endl;
//std::cout<<"remainder_new  "<<remainder_new <<std::endl;
////////////////


			/**
			* broadcast element p of input[] to vector of type fpvec<uint32_t>
			* broadcastCurrentValue contains sixteen times value of input[i]
			**/
			fpvec<Type, regSize> broadcastCurrentValue = set1<Type, regSize>(inputValue);
/*std::cout<<"broadcastCurrentValue_Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << broadcastCurrentValue.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;	
*/			while(1) {

////// OLD //////////////////////	
/*
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				int32_t overflow = (aligned_start + 16) - HSIZE;
std::cout<<"overflow "<<overflow<<std::endl;						
				overflow = overflow < 0? 0: overflow;
std::cout<<"overflow "<<overflow<<std::endl;						
				Type overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
std::cout<<"overflow_correction_mask_i "<<overflow_correction_mask_i<<std::endl;					
				fpvec<Type, regSize> overflow_correction_mask = cvtu32_mask16<Type, regSize>(overflow_correction_mask_i);
std::cout<<"overflow_correction_mask Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << overflow_correction_mask.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;	*/
////////////////////////////////////////////
////// NEW //////
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				TypeSigned overflow = (aligned_start + elementCount) - HSIZE;
//std::cout<<"overflow_new "<<overflow_new<<std::endl;				
				overflow = overflow < 0 ? 0 : overflow;
				Type oferflowUnsigned = (Type)overflow;

				// throw exception, if calculated overflow is bigger than amount of elements within the register
				// This case can only result from incorrectly configured global parameters.
				if (oferflowUnsigned > elementCount) {
					throw std::out_of_range("Value of oferflowUnsigned is bigger than value of elementCount - no valid values / range");
				}

				// use function createOverflowCorrectionMask() to create overflow correction mask
				fpvec<Type, regSize> overflow_correction_mask = createOverflowCorrectionMask<Type, regSize>(oferflowUnsigned);
/*std::cout<<"overflow_new "<<overflow_new<<std::endl;	
std::cout<<"oferflowUnsigned_new "<<oferflowUnsigned_new<<std::endl;		
std::cout<<"Register overflow_correction_mask_new : "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << overflow_correction_mask_new.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
std::cout<<" "<<std::endl;	
std::cout<<" "<<std::endl;					
////////////////
std::cout << "============= used variables 2nd party =========" << std::endl;
std::cout<<"inputValue "<<inputValue<<std::endl;	
std::cout<<"hash_key "<<hash_key<<std::endl;	
std::cout<<"aligned_start "<<aligned_start<<std::endl;
std::cout<<"remainder "<<remainder<<std::endl;	
std::cout << "============= BEGIN 2nd party =========" << std::endl;*/

////// OLD //////////////////////
/*
				int32_t cutlow = 16 - remainder; // should be in a range from 1-(regSize/sizeof(Type))
std::cout<<"cutlow "<<cutlow<<std::endl;					
				Type cutlow_mask_i = (1 << cutlow) -1;
std::cout<<"cutlow_mask_i "<<cutlow_mask_i<<std::endl;					
				cutlow_mask_i <<= remainder;
std::cout<<"cutlow_mask_i after <<= remainder  "<<cutlow_mask_i<<std::endl;					

				Type combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
std::cout<<"combined_mask_i "<<combined_mask_i<<std::endl;				
				fpvec<Type, regSize> overflow_and_cutlow_mask = cvtu32_mask16<Type, regSize>(combined_mask_i);
std::cout<<"overflow_and_cutlow_mask Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << overflow_and_cutlow_mask.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;	*/
////////////////////////////////////////////
////// NEW //////////////////////

TypeSigned cutlow = elementCount - remainder; // should be in a range from 1 to elementCount
//std::cout << "cutlow NEW " << cutlow_NEW << std::endl;
Type cutlowUnsigned = (Type)cutlow;
//std::cout << "cutlowUnsigned NEW " << cutlowUnsigned_NEW<< std::endl;

fpvec<Type, regSize> cutlow_mask = createCutlowMask<Type, regSize>(cutlowUnsigned);
/*std::cout << "cutlow_mask_NEW ";
for (int i = 0; i < elementCount; i++) {
	std::cout << cutlow_mask_NEW.elements[i] << " ";
}
std::cout << " " << std::endl;
*/
fpvec<Type, regSize> overflow_and_cutlow_mask = mask_cmpeq_epi32_mask(oneMask, cutlow_mask, overflow_correction_mask);
/*std::cout << "overflow_and_cutlow_mask_NEW NEW ";
for (int i = 0; i < elementCount; i++) {
	std::cout << overflow_and_cutlow_mask_NEW.elements[i] << " ";
}
std::cout << " " << std::endl;
*/

////////////////////////////////
////////////////////////////////

				// Load 16 consecutive elements from hashVec, starting from position hash_key
				fpvec<Type, regSize> nextElements = load_epi32(oneMask, hashVec, aligned_start, HSIZE);
/*std::cout << "nextElements ";
for (int i=0; i<elementCount; i++) {
		std::cout << nextElements.elements[i] << " ";
	} 	
std::cout<<" "<<std::endl;
*/				// compare vector with broadcast value against vector with following elements for equality
				fpvec<Type, regSize> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
/*std::cout << "compareRes ";
for (int i=0; i<elementCount; i++) {
		std::cout << compareRes.elements[i] << " ";
	} 	
	std::cout<<" "<<std::endl;			
*/				/**
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
//std::cout<<"say hello1 "<<std::endl;					
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
/*
std::cout << "checkForFreeSpace ";
for (int i=0; i<elementCount; i++) {
		std::cout << checkForFreeSpace.elements[i] << " ";
	} 	
std::cout<<" "<<std::endl;		
*/
					Type innerMask = mask2int(checkForFreeSpace);
					if(innerMask != 0) {                // CASE B1    
						//this does not calculate the correct position. we should rather look at trailing zeros.
						Type pos = ctz_onceBultin(checkForFreeSpace);

						hashVec[aligned_start+pos] = (uint32_t)inputValue;
						countVec[aligned_start+pos]++;
//std::cout<<"hash_key "<<hash_key<<std::endl;		
//std::cout<<"pos "<<pos<<std::endl;		
//std::cout<<"inputValue "<<inputValue<<std::endl;		
						p++;
//std::cout<<"say hello2 "<<std::endl;
						break;
					}
					else {                   // CASE B2 
					//aligned_start = (aligned_start+16) % HSIZE;
	// since we now use the overflow mask we can do this to change our position
	// we ALSO need to set the remainder to 0.  
						remainder = 0;
						aligned_start += elementCount;
//std::cout<<"B2 hash_key before "<<hash_key<<std::endl;						
						if(aligned_start >= HSIZE){
							aligned_start = 0;
						}
//std::cout<<"B2 hash_key after "<<hash_key<<std::endl;							
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
 * Variant 3 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param input the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant3(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
////////////////////////////////////////////////////////////////////////////////
//// Check global parameters & calculate iterations parameter
	// ensure global defined regSize is nice
    // old:  assert((regSize == 64) || (regSize == 128) || (regSize == 192) || (regSize == 256));
	assert((regSize == 64) || (regSize == 128) || (regSize == 256));
	size_t iterations =  loops;
	assert(dataSize % elementCount == 0);
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

    // define dataVec register
	fpvec<Type, regSize> dataVec;

	// iterate over input data with a SIMD register size of regSize bytes (elementCount elements)
	for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
		// Load complete CL (register) in one clock cycle
		dataVec = load<Type, regSize>(input, i_cnt);

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
				Type aligned_start = (hash_key/elementCount)*elementCount;
				Type remainder = hash_key - aligned_start; // should be equal to hash_key % 16
			
				while (1) {
					Type overflow = (aligned_start + elementCount) - HSIZE;
					overflow = overflow < 0? 0: overflow;
					Type overflow_correction_mask_i = (1 << ((Type)(elementCount-overflow))) - 1; 
					fpvec<Type, regSize> overflow_correction_mask = cvtu32_mask16<Type, regSize>(overflow_correction_mask_i);

					Type cutlow = elementCount - remainder; // should be in a range from 1 - (regSize/sizeof(Type))
					Type cutlow_mask_i = (1 << cutlow) -1;
					cutlow_mask_i <<= remainder;

					Type combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
					fpvec<Type, regSize> overflow_and_cutlow_mask = cvtu32_mask16<Type, regSize>(combined_mask_i);

					// Load 16 consecutive elements from hashVec, starting from position hash_key
					fpvec<Type, regSize> nextElements = load_epi32(oneMask, hashVec, aligned_start, HSIZE);
					
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
 * Variant 4 of a AVX512-based group_count implementation.
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

    fpvec<Type, 64> oneMask_TMP = set1<Type, 64>(one);
    fpvec<Type, 64> zeroMask_TMP = set1<Type, 64>(zero);
/*
std::cout<<"input"<<std::endl;
	for(size_t i=0; i<dataSize; i++) {
		std::cout<<input[i] << "  |  "<<input[i]<<std::endl;
	}
*/
	
	//// declare the basic hash- and count-map structure for this approach an some function intern variables
    fpvec<Type, 64>* hash_map;
    fpvec<Type, 64>* count_map;

	const size_t m_elements_per_vector = (64) / sizeof(Type);					/////// CHANGE TO = elementCount !!
std::cout<< "result m_elements_per_vector: "<<m_elements_per_vector<<std::endl;
	const size_t m_HSIZE_v = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
std::cout<< "result m_HSIZE_v: "<<m_HSIZE_v<<std::endl;
	const size_t m_HSIZE = HSIZE;
 

    // use a vector with elements of type <fpvec<uint32_t> as structure "around" the registers
    hash_map = new fpvec<Type, 64>[m_HSIZE_v];
    count_map = new fpvec<Type, 64>[m_HSIZE_v];

    // loading data. On the first exec this should result in only 0 vals.   
    for(size_t i = 0; i < m_HSIZE_v; i++){
        size_t h = i * m_elements_per_vector;

       	hash_map[i] = load_epi32(oneMask_TMP, hashVec, h, m_HSIZE);
    	count_map[i] = load_epi32(oneMask_TMP, countVec, h, m_HSIZE);
	}

	/**
	 * calculate overflow in last register of hash_map and count_map, to prevent errors from storing elements in hash_map[m_HSIZE_v-1] which position within end-result is >HSIZE 
	 *	
	 * due to this approach, the hash_map and count_map can have overall more slots than the value of HSIZE
	 * set value of positions of the last register that "overflows" to a value that is bigger than distinctValues
	 * These values can't be part of input data array (because positions bigger than HSIZE will not be stored), but will be will be handled as "no match, but position already filled" within the algorithm.
	 * Since only HSIZE values are stored at the end (back to hashVec and countVec), these values are simply dropped at the end.
	 */
	// define variables and register for overflow calculation
	fpvec<Type, 64> overflow_correction_mask;
	Type value_bigger_distinctValues;
	fpvec<Type, 64> value_bigger_distinctValues_mask;

	// caculate overflow and mark positions in last register that will be overflow the value of HSIZE
	Type oferflowUnsigned = (m_HSIZE_v * m_elements_per_vector) - HSIZE;
	if (oferflowUnsigned > 0) {
		overflow_correction_mask = createOverflowCorrectionMask<Type, 64>(oferflowUnsigned);
		value_bigger_distinctValues = (Type)(distinctValues+7); 	
		value_bigger_distinctValues_mask = set1<Type, 64>(value_bigger_distinctValues);

		hash_map[m_HSIZE_v-1] = mask_set1(value_bigger_distinctValues_mask, overflow_correction_mask, zero);
		count_map[m_HSIZE_v-1] = set1<Type, 64>(zero);
	}
std::cout<< "oferflowUnsigned: "<<oferflowUnsigned<<std::endl;
std::cout<< "overflow_correction_mask: "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << overflow_correction_mask.elements[i] << " ";
}  std::cout<<std::endl;		
std::cout<< "value_bigger_distinctValues: "<<value_bigger_distinctValues<<std::endl;
std::cout<< "value_bigger_distinctValues_mask: "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << value_bigger_distinctValues_mask.elements[i] << " ";
}  std::cout<<std::endl;	

std::cout<< "hash_map[m_HSIZE_v-1]: "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << hash_map[m_HSIZE_v-1].elements[i] << " ";
}  std::cout<<std::endl;	


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
    std::array<fpvec<Type, 64>, 16> masks {};
	masks = cvtu32_create_writeMask_Matrix<Type, 64>();

///    
// ! ATTENTION - changed indizes (compared to the first AVX512 implementation) !
// mask with only 0 => zero_cvtu32_mask
// masks = array of 16 masks respectively fpvec<uint32_t> with one 1 at unique positions 
///	

    int p = 0;
    while (p < dataSize) {
        Type inputValue = input[p];
		Type hash_key = hashx(inputValue,m_HSIZE_v);

        fpvec<Type, 64> broadcastCurrentValue = set1<Type, 64>(inputValue);
/*
std::cout<< "=============== NEW ROUND : p = "<<p<<std::endl;		
std::cout<< "inputValue "<<inputValue<<std::endl;	
std::cout<< "hash_key "<<hash_key<<std::endl;	

std::cout<< "broadcastCurrentValue register: "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << broadcastCurrentValue.elements[i] << " ";
}  std::cout<<std::endl;
*/
        while(1) {
/*
std::cout<< "hash_map[hash_key] - BEFORE:"<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << hash_map[hash_key].elements[i] << " ";
}  std::cout<<std::endl;	
std::cout<< "count_map[hash_key] - BEFORE:"<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << count_map[hash_key].elements[i] << " ";
}  std::cout<<std::endl;		
*/
            // compare vector with broadcast value against vector with following elements for equality
            fpvec<Type, 64> compareRes = cmpeq_epi32_mask(broadcastCurrentValue, hash_map[hash_key]);
/*
std::cout<< "compareRes register: "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << compareRes.elements[i] << " ";
}  std::cout<<std::endl;
*/
            // found match
            if (mask2int(compareRes) != 0) {
// std::cout<< "found match ! "<<std::endl;				
                count_map[hash_key] = mask_add_epi32(count_map[hash_key], compareRes, count_map[hash_key], oneMask_TMP);

                p++;
                break;
            } else { // no match found
// std::cout<< "no match found "<<std::endl;					
                // deterime free position within register
                fpvec<Type, 64> checkForFreeSpace = cmpeq_epi32_mask(zeroMask_TMP, hash_map[hash_key]);
/*std::cout<< "checkForFreeSpace register: "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << checkForFreeSpace.elements[i] << " ";
}  std::cout<<std::endl;	*/			
                if(mask2int(checkForFreeSpace) != 0) {                // CASE B1   
//std::cout<< "IF-ZWEIG mask2int(checkForFreeSpace) "<< mask2int(checkForFreeSpace) <<std::endl;				

// old : uint32_t pos = __builtin_ctz(checkForFreeSpace) + 1;
// --> omit +1, because masks with only 0 at every position is outsourced to zero_cvtu32_mask --> zeroMask is used instead                
                    Type pos = ctz_onceBultin(checkForFreeSpace);
// std::cout<< "pos "<< pos <<std::endl;	
                    //store key
                    hash_map[hash_key] = mask_set1<Type, 64>(hash_map[hash_key], masks[pos], inputValue);
                    //set count to one
                    count_map[hash_key] = mask_set1<Type, 64>(count_map[hash_key], masks[pos], (Type)1);
/*
std::cout<< "hash_map[hash_key] "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << hash_map[hash_key].elements[i] << " ";
}  std::cout<<std::endl;	
std::cout<< "count_map[hash_key] "<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << count_map[hash_key].elements[i] << " ";
}  std::cout<<std::endl;						
*/
                    p++;
                    break;
                }   else    { // CASE B2
// std::cout<< "ELSE-ZWEIG mask2int(checkForFreeSpace) "<< mask2int(checkForFreeSpace) <<std::endl;				
                    hash_key = (hash_key + 1) % m_HSIZE_v;
                }
            }
        }
    }
/*
std::cout<< "hash_map[11] - BEFORE:"<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << hash_map[11].elements[i] << " ";
}  std::cout<<std::endl;	
std::cout<< "count_map[11] - BEFORE:"<<std::endl;
for (int i=0; i<(64/sizeof(Type)); i++) {
		std::cout << count_map[11].elements[i] << " ";
}  std::cout<<std::endl;	
*/
 std::cout<<"============================================================="<<std::endl;	
 std::cout<<"============== BEFORE STORE ================================="<<std::endl;	

for (int i=0; i<m_HSIZE_v; i++) {
	std::cout<<"Current line - hash_key = "<<i<<std::endl;	
	std::cout<< "hash_map["<<i<<"]: ";
	for (int j=0; j<(64/sizeof(Type)); j++) {
		std::cout << hash_map[i].elements[j] << " ";
	}  std::cout<<std::endl;	
	std::cout<< "count_map["<<i<<"]: ";
	for (int j=0; j<(64/sizeof(Type)); j++) {
		std::cout << count_map[i].elements[j] << " ";
	}  std::cout<<std::endl;	
}

    //store data
    for(size_t i = 0; i < m_HSIZE_v; i++){
		size_t h = i * m_elements_per_vector;
				
        store_epi32(hashVec, h, hash_map[i]);
        store_epi32(countVec, h, count_map[i]);
    }

/*	std::cout<<"hashVec  |  countVec"<<std::endl;
	for(size_t i=0; i<HSIZE; i++) {
		std::cout<<hashVec[i] << "  |  "<<countVec[i]<<std::endl;
	}
*/
}
//// end of LinearProbingFPGA_variant4()
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////