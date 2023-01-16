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

#include "primitives.hpp"
#include "kernels.hpp"
#include "helper_kernel.cpp"

class kernels;

/**
 * declare some (global) basic masks and arrays
 */ 
uint32_t one = 1;
uint32_t zero = 0;
fpvec<uint32_t> oneMask = set1(one);
fpvec<uint32_t> zeroMask = set1(zero);
fpvec<uint32_t> zeroM512iArray = set1(zero);
fpvec<uint32_t> oneM512iArray = set1(one);

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

	/**
	 * iterate over input data
	 * @param p current element of input data array
	 **/ 
	int p = 0;
	while (p < dataSize) {
		// get single value from input at position p
		uint32_t inputValue = input[p];

		// compute hash_key of the input value
		uint32_t hash_key = hashx(inputValue,HSIZE);
std::cout<<"p: "<<p<<" |  inputValue: "<<inputValue<<" |  hash_key: "<<hash_key<<std::endl;
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


for (int i=0; i<(64/sizeof(uint32_t)); i++) {
		std::cout << nextCounts.elements[i] << " ";
	}
	std::cout << " " << std::endl;						


			// increment by one at the corresponding location
			nextCounts = mask_add_epi32(nextCounts, compareRes, nextCounts, oneM512iArray);


for (int i=0; i<(64/sizeof(uint32_t)); i++) {
		std::cout << nextCounts.elements[i] << " ";
	}
	std::cout << " " << std::endl;			
	std::cout << " " << std::endl;		
	std::cout << " " << std::endl;				


			// selective store of changed value
			mask_storeu_epi32(countVec, hash_key, HSIZE, compareRes,nextCounts);
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
				//fpvec<uint32_t> mask1 = knot(checkForFreeSpace); // old, not used anymore
				//compute position of the emtpy slot   
				//uint32_t pos = (32-clz_onceBultin(mask1))%16;    // old, not used anymore
				uint32_t pos = ctz_onceBultin(checkForFreeSpace);
//std::cout<<pos<<std::endl;
				// use 

				hashVec[hash_key+pos] = (uint32_t)inputValue;
				countVec[hash_key+pos]++;
std::cout<<"hashVec[hash_key+pos]: "<<hashVec[hash_key+pos]<<" |  countVec[hash_key+pos]: "<<countVec[hash_key+pos]<<std::endl;
std::cout<<"p: "<<p<<" |  hash_key: "<<hash_key<<" |  pos: "<<pos<<std::endl;
				p++;
				break;
			} else    {                   // CASE B2   
				hash_key += 16;
				if(hash_key >= HSIZE){
					hash_key = 0;
				}
			}
		}
		} 
	}
}   

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
	/**
	 * iterate over input data
	 * @param p current element of input data array
	 **/ 
	int p = 0;
	while (p < dataSize) {
		// get single value from input at position p
		uint32_t inputValue = input[p];

		// compute hash_key of the input value
		uint32_t hash_key = hashx(inputValue,HSIZE);

		// compute the aligned start position within the hashMap based the hash_key
		uint32_t aligned_start = (hash_key/16)*16;
		uint32_t remainder = hash_key - aligned_start; // should be equal to hash_key % 16
		
		/**
		 * broadcast element p of input[] to vector of type fpvec<uint32_t>
		 * broadcastCurrentValue contains sixteen times value of input[i]
		**/
		fpvec<uint32_t> broadcastCurrentValue = set1(inputValue);

		while(1) {
			// Calculating an overflow correction mask, to prevent errors form comparrisons of overflow values.
			int32_t overflow = (aligned_start + 16) - HSIZE;
			overflow = overflow < 0? 0: overflow;
			uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
			fpvec<uint32_t> overflow_correction_mask = cvtu32_mask16(overflow_correction_mask_i);

			int32_t cutlow = 16 - remainder; // should be in a range from 1-16
			uint32_t cutlow_mask_i = (1 << cutlow) -1;
			cutlow_mask_i <<= remainder;

			uint32_t combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
			fpvec<uint32_t> overflow_and_cutlow_mask = cvtu32_mask16(combined_mask_i);

			// Load 16 consecutive elements from hashVec, starting from position hash_key
			fpvec<uint32_t> nextElements = load_epi32(oneMask, hashVec, aligned_start, HSIZE);

			// compare vector with broadcast value against vector with following elements for equality
			fpvec<uint32_t> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
		
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
				uint32_t matchPos = ctz_onceBultin(compareRes);
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
				fpvec<uint32_t> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, zeroMask,nextElements);
				uint32_t innerMask = mask2int(checkForFreeSpace);
				if(innerMask != 0) {                // CASE B1    
					// fpvec<uint32_t> mask1 = knot(checkForFreeSpace);   
					// uint32_t pos = (32-clz_onceBultin(mask1))%16;
					
					//this does not calculate the correct position. we should rather look at trailing zeros.
                    uint32_t pos = ctz_onceBultin(checkForFreeSpace);

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
                    aligned_start += 16;
                    if(aligned_start >= HSIZE){
                        aligned_start = 0;
                    }
				} 
			}  
		}
  	}    
}  

/**
 * Variant 3 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param arr the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant3(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    /**
     * iterate over input data
     * @param p current element of input data array
     **/ 
    int p = 0;
    while (p < dataSize) {

        // load 16 input values
        fpvec<uint32_t> iValues = load_epi32(oneMask, input, p, HSIZE);

        //iterate over the input values
        int i=0;
        while (i<16) {

            // broadcast single value from input at postion i into a new SIMD register
            fpvec<uint32_t> idx = set1((uint32_t)i);
	  		fpvec<uint32_t> broadcastCurrentValue = permutexvar_epi32(idx,iValues);

            uint32_t inputValue = (uint32_t)broadcastCurrentValue.elements[0];
            uint32_t hash_key = hashx(inputValue,HSIZE);

            // compute the aligned start position within the hashMap based the hash_key
            uint32_t aligned_start = (hash_key/16)*16;
            uint32_t remainder = hash_key - aligned_start; // should be equal to hash_key % 16
         
            while (1) {
                int32_t overflow = (aligned_start + 16) - HSIZE;
                overflow = overflow < 0? 0: overflow;
                uint32_t overflow_correction_mask_i = (1 << (16-overflow)) - 1; 
                fpvec<uint32_t> overflow_correction_mask = cvtu32_mask16(overflow_correction_mask_i);

                int32_t cutlow = 16 - remainder; // should be in a range from 1-16
                uint32_t cutlow_mask_i = (1 << cutlow) -1;
                cutlow_mask_i <<= remainder;

                uint32_t combined_mask_i = cutlow_mask_i & overflow_correction_mask_i;
                fpvec<uint32_t> overflow_and_cutlow_mask = cvtu32_mask16(combined_mask_i);

                // Load 16 consecutive elements from hashVec, starting from position hash_key
                fpvec<uint32_t> nextElements = load_epi32(oneMask, hashVec, aligned_start, HSIZE);
            
                // compare vector with broadcast value against vector with following elements for equality
                fpvec<uint32_t> compareRes = mask_cmpeq_epi32_mask(overflow_correction_mask, broadcastCurrentValue, nextElements);
    
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
                    uint32_t matchPos = ctz_onceBultin(compareRes); 
                    
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
                    fpvec<uint32_t> checkForFreeSpace = mask_cmpeq_epi32_mask(overflow_and_cutlow_mask, zeroM512iArray, nextElements);
                    uint32_t innerMask = mask2int(checkForFreeSpace);
                    if(innerMask != 0) {                // CASE B1    
                        // __mmask16 mask1 = _mm512_knot(innerMask);   
                        // uint32_t pos = (32-__builtin_clz(mask1))%16;

                        //this does not calculate the correct position. we should rather look at trailing zeros.
                        uint32_t pos = ctz_onceBultin(checkForFreeSpace);
                        
                        hashVec[aligned_start+pos] = (uint32_t)inputValue;
                        countVec[aligned_start+pos]++;
                        i++;
                        break;
                    }   
                    else    {                   // CASE B2                    
                        //aligned_start = (aligned_start+16) % HSIZE;
// since we now use the overflow mask we can do this to change our position
// we ALSO need to set the remainder to 0.
                        remainder = 0;
                        aligned_start += 16;
                        if(aligned_start >= HSIZE){
                            aligned_start = 0;
                        }
                    }
                }
            }
        }
        p+=16;
    }
}