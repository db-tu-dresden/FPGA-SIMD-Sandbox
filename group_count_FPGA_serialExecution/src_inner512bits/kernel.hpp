#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "../config/global_settings.hpp"
#include "../primitives/primitives.hpp"

class kernelV1;
class kernelV2;
class kernelV3;
class kernelV4;
class kernelV5;

void LinearProbingFPGA_variant1(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant2(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant3(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant4(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE, size_t size);
void LinearProbingFPGA_variant5(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, Type *match_32bit, uint64_t HSIZE, size_t size);

#endif  // KERNEL_HPP