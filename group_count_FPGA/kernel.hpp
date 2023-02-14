#ifndef kernel_h__
#define kernel_h__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "global_settings.h"

#include "lib/lib.hpp"

using namespace sycl;

class kernelV1;
class kernelV2;
class kernelV3;

// @todo add queue and other necessary parameters for FPGA
void LinearProbingFPGA_variant1(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, long *out_d, uint64_t dataSize, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant2(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, long *out_d, uint64_t dataSize, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant3(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, long *out_d, uint64_t dataSize, uint64_t HSIZE, size_t size);

#endif  // kernel_h__