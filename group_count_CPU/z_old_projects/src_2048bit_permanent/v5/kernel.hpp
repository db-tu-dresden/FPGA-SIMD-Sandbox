#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "../../config/global_settings.hpp"
#include "primitives_v5_64bit_edit.hpp"

class kernelV5;

void LinearProbingFPGA_variant5(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, uint64_t *match_64bit, size_t size);

#endif  // KERNEL_HPP