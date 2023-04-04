#ifndef SCALAR_REMAINDER_HPP
#define SCALAR_REMAINDER_HPP

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "lib/lib.hpp"
#include "../config/global_settings.hpp"
#include "../primitives/primitives.hpp"

using namespace sycl;


class kernelV5_scalar_remainder;


void scalar_remainder_variant5(uint32_t *arr_h, uint32_t *hashVec_h, uint32_t *countVec_h, size_t *p_h);

#endif  // SCALAR_REMAINDER_HPP