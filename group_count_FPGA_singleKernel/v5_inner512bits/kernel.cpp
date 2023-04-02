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
class kernelV5;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 5 of a hasbased group_count implementation for FPGA.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param q device queue
 * @param arr_d the input data array
 * @param hashVec_d store value of k at position hashx(k)
 * @param countVec_d store the count of occurence of k at position hashx(k)
 * @param match_32bit_d																		// defined in concerned function, not part of paramater list anymore 
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]		// global defined, not part of paramater list anymore 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])					// global defined, not part of paramater list anymore 
 * @param size = number_CL*16 with number_CL = number_CL_buckets * (4096/16);
 */
void LinearProbingFPGA_variant5(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, size_t size) {
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

	#define EMPTY_SPOT 0
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// starting point of the logic of the algorithm

	q.submit([&](handler& h) {
		h.single_task<kernelV5>([=]() [[intel::kernel_args_restrict]] {

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

			// create buffer array with embedded memory on FPGA as register
			// create a buffer on FPGA to realize buffer[] which is needed to store conflicts and hash_values within the following algorithm
			[[intel::fpga_register]] std::array<Type, elements_per_inner_register> buffer;	
			#pragma unroll 16
			for(int i=0; i<elements_per_inner_register; i++) {	
				buffer[i] = 0;
			}

			// create counting parameter p on local memory of FPGA
			// [[intel::fpga_register]] size_t p[1];
			// p[0]=0;
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////

			////////////////////////////////////////////////////////////////////////////////
			//// declare some basic masks and arrays
			Type one = 1;
			Type zero = 0;
			// Because we use the oneMask and zeroMask only inside the inner-part of the algorithm, we create them with <Type, inner_regSize>
			// -> with 16 elements of type uint32_t (32bit) => 512bit per register
			fpvec<Type, inner_regSize> oneMask = set1<Type, inner_regSize>(one);
			fpvec<Type, inner_regSize> zeroMask = set1<Type, inner_regSize>(zero);
			
			// convert variable HSIZE to 32-bit
			// we don't need HSIZE as 64-bit integer, but this datatype cost much ressources on FPGA
			Type tmp_HSIZE = (Type)HSIZE;
			fpvec<Type, inner_regSize> tmp_HSIZE_mask = set1<Type, inner_regSize>(tmp_HSIZE);
			////////////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////////////
			
			// AVX512-implementation of this LinearProbing-algorithm_v5 (SoA_conflict_v1) was actually using this while-loop.
			// We replaced this solution through our for-loop for the loading cycles similar to the previous versions.
			// while(p + elements_per_register < dataSize){

			// #########################################
			// #### START OF FPGA parallelized part ####
			// #########################################
			// define dataVec register
			fpvec<Type, regSize> input_value;

			// iterate over input data with a SIMD register size of regSize bytes (elements_per_register elements)
			#pragma nounroll		// compiler should realize that this loop cannot be unrolled
			for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {

				// calculate chunk_idx and chunk_offset for current iteration step
				const int i_cnt_const = i_cnt;
				const int chunk_idx = i_cnt_const / kIterationsPerChunk;
				const int chunk_offset = i_cnt_const % kIterationsPerChunk;

				// Load complete CL (register) in one clock cycle (same for PCIe and DDR4)
				input_value = maxLoad_per_clock_cycle<Type, regSize>(input, kNumLSUs, kValuesPerLSU, chunk_idx, kValuesPerInterleavedChunk, chunk_offset);

				// split loaded data into 4 "working register" á 512 bit to work over data segments of 512bits
				std::array<fpvec<Type, inner_regSize>, (inner_regSize/sizeof(Type))> workingData {};
				#pragma unroll
				for (int i=0; i<(regSize/inner_regSize); i++) {				// regSize/inner_regSize should be 4
					#pragma unroll
					for (int j=0; j<elements_per_inner_register; j++) {
						workingData[i].elements[j] = input_value.elements[((i*elements_per_inner_register)+j)];
					}	
				}

				#pragma nounroll
				for (int i=0; i<(regSize/inner_regSize); i++) {				// regSize/inner_regSize should be 4
					// read 512-bit segments of loaded data and work through the algorithm with segments of only 512-bits
					fpvec<Type, inner_regSize> tmp_workingData = workingData[i];

					// how much the given count should be increased for the given input.
					fpvec<Type, inner_regSize> input_add = set1<Type, inner_regSize>(one);

					// search for conflicts
					fpvec<Type, inner_regSize> conflicts = conflict_epi32(tmp_workingData);
					// masked to indicate were there is a conflict in the input_values and were not.
					fpvec<Type, inner_regSize> no_conflicts_mask = cmpeq_epi32_mask(zeroMask, conflicts);
					fpvec<Type, inner_regSize> negativ_no_conflicts_mask = knot(no_conflicts_mask);

					// we need to store the conflicts so we can interprete them as masks. and access them.
					// we are only interested in the enties that are not zero. That means the conflict cases.					
					mask_compressstoreu_epi32(buffer, negativ_no_conflicts_mask, conflicts);

					size_t conflict_count = popcount_builtin(negativ_no_conflicts_mask);
					// add at all the places where the conflict masks indicates that there is an overlap
					for(size_t i = 0; i < conflict_count; i++){
						fpvec<Type, inner_regSize> tmp_buffer_mask = setX_singleValue<Type, inner_regSize>(buffer[i]);
						input_add = mask_add_epi32<Type, inner_regSize>(input_add, tmp_buffer_mask, input_add, oneMask);
					}
						// alternative calculation:
						//fpvec<Type, inner_regSize> tmp_buffer_mask = setX_multipleValues<Type, inner_regSize>(buffer, conflict_count);
						//input_add = mask_add_epi32<Type, inner_regSize>(input_add, oneMask, input_add, tmp_buffer_mask);

					// we override the value and what to add with zero in the positions where we have a conflict.
					// NOTE: This steps might not be necessary.
					tmp_workingData = mask_set1(tmp_workingData, negativ_no_conflicts_mask, zero);
					input_add = mask_set1(input_add, negativ_no_conflicts_mask, zero);

					// now we can calculate the hashes.
					// for this we can store the tmp_workingData hash it and load it
					// OR we use the input and hash it save it in to buffer and than make a maskz load for the hashed data
					// OR we have a simdifyed Hash Algorithm! For the most cases we would need an avx... mod. 
					// _mm512_store_epi32(buffer, tmp_workingData);
					#pragma unroll
					for(size_t i = 0; i < elements_per_inner_register; i++){
						// old : buffer[i] = hashx(input[p + i], HSIZE);
						// we don't need this offset-calculation (p+i), because we iterate through our data-register (tmp_workingData), which
						// will be loaded with new data in every data-loading-iteration. So we just have to iterate through the elements within this register. 
						buffer[i] = hashx(tmp_workingData.elements[i], HSIZE);
					}
					
					fpvec<Type, inner_regSize> hash_map_position = mask_loadu_from_buffer<Type, inner_regSize>(no_conflicts_mask, buffer, (Type)0); 	// these are the hash values

					while(mask2int(no_conflicts_mask) !=0) {		
						// now we can gather the data from the different positions where we have no conflicts.
						fpvec<Type, inner_regSize> hash_map_value = mask_i32gather_epi32(zeroMask, no_conflicts_mask, hash_map_position, hashVec);
						// with these we can calculate the different possible hits. Real hits and empty positions.
						fpvec<Type, inner_regSize> foundPos = mask_cmpeq_epi32_mask(no_conflicts_mask, tmp_workingData, hash_map_value);
						fpvec<Type, inner_regSize> foundEmpty = mask_cmpeq_epi32_mask(no_conflicts_mask, zeroMask, hash_map_value);

						if(mask2int(foundPos) != 0){		//A
							// Now we have to gather the count. IMPORTANT! the count is a 32bit integer. 
							// FOR NOW THIS IS CORRECT BUT MIGHT CHANGE LATER!
							// For 64bit integers we would need to find a different solution!
							fpvec<Type, inner_regSize> hash_map_value = mask_i32gather_epi32(zeroMask, foundPos, hash_map_position, countVec);
							// on this count we can know add the pre calculated values. and scatter it back to their positions
							hash_map_value = maskz_add_epi32(foundPos, hash_map_value, input_add);
							mask_i32scatter_epi32<Type, inner_regSize>(countVec, foundPos, hash_map_position, hash_map_value);
								
							// finaly we remove the entries we just saved from the no_conflicts_mask such that the work to be done shrinkes.
							no_conflicts_mask = kAndn(foundPos, no_conflicts_mask);
						}

						if(mask2int(foundEmpty) != 0){		//B1
							// now we have to check for conflicts to prevent two different entries to write to the same position.
							fpvec<Type, inner_regSize> saveConflicts = maskz_conflict_epi32<Type, inner_regSize>(foundEmpty, hash_map_position);
							fpvec<Type, inner_regSize> empty = set1<Type, inner_regSize>(mask2int_uint32_t(foundEmpty));
							saveConflicts = register_and<Type, inner_regSize>(saveConflicts, empty);
							
							fpvec<Type, inner_regSize> to_save_data = cmpeq_epi32_mask<Type, inner_regSize>(zeroMask, saveConflicts);

							to_save_data = kAnd(to_save_data, foundEmpty);

							// with the cleaned mask we can now save the data.
							mask_i32scatter_epi32<Type, inner_regSize>(hashVec, to_save_data, hash_map_position, tmp_workingData);
							mask_i32scatter_epi32<Type, inner_regSize>(countVec, to_save_data, hash_map_position, input_add);

							//and again we need to remove the data from the todo list
							no_conflicts_mask = kAndn(to_save_data, no_conflicts_mask);
						}

						// afterwards we add one on the current positions of the still to be handled values.
						hash_map_position = maskz_add_epi32(no_conflicts_mask, hash_map_position, oneMask);

						// Since there isn't a modulo operation we have to check if the values are bigger or equal the HSIZE AND IF we have to set them to zero
						// fpvec<Type, inner_regSize> tmp_HSIZE_mask = set1<Type, inner_regSize>(tmp_HSIZE);			// << we declare this mask before the loop
						fpvec<Type, inner_regSize> tobig = mask_cmp_epi32_mask_NLT(no_conflicts_mask, hash_map_position, tmp_HSIZE_mask);
						hash_map_position = mask_set1(hash_map_position, tobig, (Type)0);

						// we repeat this for one vector as long as their is still a value to be saved.
					}
					//p[0] += elements_per_inner_register;
				}
			}	
			// #######################################
			// #### END OF FPGA parallelized part ####
			// #######################################

			/*		
			//scalar remainder	
			// As long as we use a dataSize mod 4096 = 0, the algorithm doesn't leave any rest; scalar remainder will not be entered --> we deactivate them for performance reasons
 			while(p[0] < dataSize){
				// get the possible possition of the element.
				Type currentValue = input[p[0]];
				Type hash_key = hashx(currentValue, HSIZE);

				while(1){
					// get the value of this position
					Type value = hashVec[hash_key];
					
					// Check if it is the correct spot
					if(value == currentValue){
						countVec[hash_key]++;
						break;
						
					// Check if the spot is empty
					}else if(value == EMPTY_SPOT){
						hashVec[hash_key] = currentValue;
						countVec[hash_key] = 1;
						break;
						
					}
					else{
						//go to the next spot
						hash_key = (hash_key + 1);
						if (hash_key >= HSIZE) {
							hash_key = hash_key-HSIZE;
						}	
						//we assume that the hash_table is big enough
					}
				}
				p[0]++;
			} */
					
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

			// multiple improvements are possible:
			// 1.   we could increase the performance of the worst case first write.
			// 2.   we could absorbe the scalar remainder with overflow masks
			// these would probably have a negative impact on  the overall performance.
		});
	}).wait();
}   
//// end of LinearProbingFPGA_variant5()
////////////////////////////////////////////////////////////////////////////////