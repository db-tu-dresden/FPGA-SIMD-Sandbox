#ifndef KERNEL_HPP__
#define KERNEL_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "lib/lib.hpp"
#include "../config/global_settings.hpp"
#include "../primitives/primitives.hpp"

using namespace sycl;

class kernelV4;

void LinearProbingFPGA_variant4(queue& q, uint32_t *arr_d, uint32_t *hashVec_d, uint32_t *countVec_d, size_t size);


#endif  // KERNEL_HPP__