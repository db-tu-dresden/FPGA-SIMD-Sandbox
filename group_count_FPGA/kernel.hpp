#ifndef kernel_h__
#define kernel_h__

#include <CL/sycl.hpp>
#include "lib/lib.hpp"

class LinearProbingFPGA_variant1;
class LinearProbingFPGA_variant2;
class LinearProbingFPGA_variant3;

// @todo add queue and other necessary parameters for FPGA
void LinearProbingFPGA_variant1(queue& q, uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t dataSize, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant2(queue& q, uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t dataSize, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant3(queue& q, uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t dataSize, uint64_t HSIZE, size_t size);

#endif  // kernel_h__