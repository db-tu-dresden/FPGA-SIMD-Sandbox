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
/*std::cout<<"broadcastCurrentValue_Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << broadcastCurrentValue.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
*/			while (1) {
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				Type  overflow = (hash_key + elementCount) - HSIZE;
//std::cout<<"overflow "<<overflow<<std::endl;					
				overflow = overflow < 0? 0: overflow;
//std::cout<<"overflow "<<overflow<<std::endl;					
				Type overflow_correction_mask_i = (1 << (int32_t)(elementCount-overflow)) - 1;
//std::cout<<"overflow_correction_mask_i "<<overflow_correction_mask_i<<std::endl;				 
				fpvec<Type, regSize> overflow_correction_mask = cvtu32_mask16<Type, regSize>(overflow_correction_mask_i);
/**
for (int i=0; i<elementCount; i++) {
	std::cout << overflow_correction_mask.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
std::cout<<"inputValue "<<inputValue<<std::endl;	
std::cout<<"hash_key "<<hash_key<<std::endl;		
*/
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
std::cout<<"content of hashVec | countVec: "<<std::endl;
for (int i=0; i<HSIZE; i++) {
	std::cout << hashVec[i] << "  |  " << countVec[i]<<std::endl;
} 	
std::cout<<" "<<std::endl;
		
std::cout<<"================================================"<<std::endl;
std::cout<<"===ATTENTION for-loop for iterations: i_cnt "<<i_cnt<<std::endl;		
		// Load complete CL (register) in one clock cycle
		dataVec = load<Type, regSize>(input, i_cnt);

std::cout<<"dataVec_Register: "<<std::endl;
for (int i=0; i<elementCount; i++) {
	std::cout << dataVec.elements[i] << " ";
} 	
std::cout<<" "<<std::endl;
		/**
		* iterate over input data / always step by step through the currently 16 (or #elementCount) loaded elements
		* @param p current element of input data array
		**/ 	
		int p = 0;
		while (p < elementCount) {
std::cout<<"===ATTENTION while-loop for elements of register: p "<<p<<std::endl;			
			// get single value from current dataVec register at position p
			Type inputValue = dataVec.elements[p];
std::cout<<"inputValue "<<inputValue<<std::endl;	
			// compute hash_key of the input value
			Type hash_key = hashx(inputValue,HSIZE);
std::cout<<"hash_key "<<hash_key<<std::endl;
			// compute the aligned start position within the hashMap based the hash_key
			Type aligned_start = (hash_key/elementCount)*elementCount;
			Type remainder = hash_key - aligned_start; // should be equal to hash_key % elementCount
std::cout<<"remainder "<<remainder<<std::endl;				
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
				// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
				Type overflow = (aligned_start + 64) - HSIZE;
std::cout<<"overflow "<<overflow<<std::endl;						
				overflow = overflow < 0? 0: overflow;
std::cout<<"overflow "<<overflow<<std::endl;						
				Type overflow_correction_mask_i = (1 << (64-overflow)) - 1; 
std::cout<<"overflow_correction_mask_i "<<overflow_correction_mask_i<<std::endl;					
				fpvec<Type, regSize> overflow_correction_mask = cvtu32_mask16<Type, regSize>(overflow_correction_mask_i);

				int32_t cutlow = 64 - remainder; // should be in a range from 1-(regSize/sizeof(Type))
std::cout<<"cutlow "<<cutlow<<std::endl;					
				Type cutlow_mask_i = (1 << cutlow) -1;
std::cout<<"cutlow_mask_i "<<cutlow_mask_i<<std::endl;					
				cutlow_mask_i <<= remainder;
std::cout<<"cutlow_mask_i "<<cutlow_mask_i<<std::endl;					

				Type combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
std::cout<<"combined_mask_i "<<combined_mask_i<<std::endl;				
				fpvec<Type, regSize> overflow_and_cutlow_mask = cvtu32_mask16<Type, regSize>(combined_mask_i);
/*
std::cout<<" "<<std::endl;
std::cout<<"inputValue "<<inputValue<<std::endl;	
std::cout<<"hash_key "<<hash_key<<std::endl;	
*/
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

void myTest() {

	Type inputValue = 37;
std::cout<<"inputValue "<<inputValue<<std::endl;			
	// compute hash_key of the input value
	Type hash_key = hashx(inputValue,HSIZE);
std::cout<<"hash_key "<<hash_key<<std::endl;

	Type inputValue2 = 47;
std::cout<<"inputValue2 "<<inputValue2<<std::endl;			
	// compute hash_key of the input value
	Type hash_key2 = hashx(inputValue2,HSIZE);
std::cout<<"hash_key2 "<<hash_key2<<std::endl;

	Type inputValue3 = 128;
std::cout<<"inputValue3 "<<inputValue3<<std::endl;			
	// compute hash_key of the input value
	Type hash_key3 = hashx(inputValue3,HSIZE);
std::cout<<"hash_key3 "<<hash_key3<<std::endl;

	
}