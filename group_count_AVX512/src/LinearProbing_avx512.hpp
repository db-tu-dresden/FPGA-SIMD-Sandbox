#ifndef LINEAR_PROBING_AVX512__HPP
#define LINEAR_PROBING_AVX512__HPP

using namespace std;

void LinearProbingAVX512Variant1(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);
void LinearProbingAVX512Variant2(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);
void LinearProbingAVX512Variant3(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);
void LinearProbingAVX512Variant4(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);
void LinearProbingAVX512Variant5(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);

#endif  // LINEAR_PROBING_AVX512__HPP