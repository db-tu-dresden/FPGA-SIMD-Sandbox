#ifndef HELPER_KERNEL_HPP
#define HELPER_KERNEL_HPP

#include <CL/sycl.hpp>

using namespace std;

extern SYCL_EXTERNAL unsigned int hashx(int key, int HSIZE);

uint32_t exponentiation_primitive_uint32_t(int x, int a);
uint64_t exponentiation_primitive_uint64_t(int x, int a);
void printBits(size_t const size, void const * const ptr);

#endif  // HELPER_KERNEL_HPP