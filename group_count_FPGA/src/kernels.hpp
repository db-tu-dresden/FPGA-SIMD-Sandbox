#ifndef KERNELS_HPP
#define KERNELS_HPP




// void printBits(size_t const size, void const * const ptr);
void LinearProbingFPGA_variant1(uint32_t *input, uint64_t dataSize, uint32_t *hashVec, uint32_t *countVec, uint64_t HSIZE);

#endif