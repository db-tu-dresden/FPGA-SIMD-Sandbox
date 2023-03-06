#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "lib/lib.hpp"
#include "global_settings.hpp"
#include "primitives.hpp"

using namespace sycl;


class kernelV5;

void LinearProbingFPGA_variant5(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, uint32_t *buffer_d, uint64_t dataSize, uint64_t HSIZE);

#endif  // KERNEL_HPP