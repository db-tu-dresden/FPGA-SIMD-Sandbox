#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "../config/global_settings.hpp"
#include "../primitives/primitives.hpp"

class kernelV1;
class kernelV2;
class kernelV3;
class kernelV4;
// class kernelV5;

void LinearProbingFPGA_variant1(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size);
void LinearProbingFPGA_variant2(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size);
void LinearProbingFPGA_variant3(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size);
void LinearProbingFPGA_variant4(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size);
// void LinearProbingFPGA_variant5(uint32_t *input, uint32_t *hashVec, uint32_t *countVec, size_t size);

#endif  // KERNEL_HPP