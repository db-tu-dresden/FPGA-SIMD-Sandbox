#ifndef LinearProbingScalar_HPP
#define LinearProbingScalar_HPP

class kernelScalar;


void LinearProbingScalar(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, int HSIZE);

#endif  // LinearProbingScalar_HPP