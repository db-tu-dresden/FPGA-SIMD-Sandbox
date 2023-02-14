#ifndef HELPER_KERNEL_HPP
#define HELPER_KERNEL_HPP

#include <CL/sycl.hpp>

using namespace std;

extern SYCL_EXTERNAL unsigned int hashx(int key, int HSIZE);
void printBits(size_t const size, void const * const ptr);

#endif  // HELPER_KERNEL_HPP