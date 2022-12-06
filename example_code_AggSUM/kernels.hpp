#ifndef KERNELS_HPP
#define KERNELS_HPP


using namespace sycl;

void aggregation_kernel(queue& q, int *in_host, long *out_host, size_t size);

#endif