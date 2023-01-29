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
#include "kernel.h"
#include "helper_kernel.cpp"

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

class LinearProbingFPGA_variant1;
class LinearProbingFPGA_variant2;
class LinearProbingFPGA_variant3;

////////////////////////////////////////////////////////////////////////////////
//// declare some (global) basic masks and arrays
uint32_t one = 1;
uint32_t zero = 0;
fpvec<uint32_t> oneMask = set1(one);
fpvec<uint32_t> zeroMask = set1(zero);
fpvec<uint32_t> zeroM512iArray = set1(zero);
fpvec<uint32_t> oneM512iArray = set1(one);
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
void LinearProbingFPGA_variant1(queue& q, uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t dataSize, uint64_t HSIZE, size_t size) {
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

	q.submit([&](handler& h) {
		h.single_task<kernels>([=]() [[intel::kernel_args_restrict]] {

//		host_ptr<uint32_t> in(input);
//		host_ptr<uint32_t> out(hashVec);
//		host_ptr<uint32_t> out(countVec);


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
void LinearProbingFPGA_variant2(queue& q, uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t dataSize, uint64_t HSIZE, size_t size) {
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

/**
 * ...
 * ...
 * ...
 * ...
 * ...
 * ...
*/


}  
//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////

/**
 * Variant 3 of a AVX512-based group_count implementation.
 * The algorithm uses the LinearProbing approach to perform the group-count aggregation.
 * @param arr the input data array
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param hashVec store value of k at position hashx(k)
 * @param countVec store the count of occurence of k at position hashx(k)
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
void LinearProbingFPGA_variant3(queue& q, uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t dataSize, uint64_t HSIZE, size_t size) {
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

/**
 * ...
 * ...
 * ...
 * ...
 * ...
 * ...
*/
  
}
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////