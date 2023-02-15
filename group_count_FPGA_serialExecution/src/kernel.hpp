#ifndef KERNEL_HPP
#define KERNEL_HPP

class kernelV1;
class kernelV2;
class kernelV3;

void LinearProbingFPGA_variant1(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE);
void LinearProbingFPGA_variant2(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE);
void LinearProbingFPGA_variant3(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE);

#endif  // KERNEL_HPP