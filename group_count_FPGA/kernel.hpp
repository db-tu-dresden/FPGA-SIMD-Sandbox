#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "lib/lib.hpp"
#include "global_settings.hpp"
#include "primitives.hpp"

using namespace sycl;

class kernelV1;
class kernelV2;
class kernelV3;
class kernelV4;

void LinearProbingFPGA_variant1(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE);
void LinearProbingFPGA_variant2(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE);
void LinearProbingFPGA_variant3(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE);
void LinearProbingFPGA_variant4(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint64_t dataSize, uint64_t HSIZE, 
                                fpvec<Type, regSize> *hash_map_d, fpvec<Type, regSize> *count_map_d, size_t m_elements_per_vector, size_t m_HSIZE_v, size_t m_HSIZE);

#endif  // KERNEL_HPP