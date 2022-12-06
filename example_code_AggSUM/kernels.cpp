#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>


#include "primitives.hpp"
#include "kernels.hpp"


class kernels;

void aggregation_kernel(queue& q, int *in_host, long *out_host, size_t size) {

  size_t iterations =  size / 16 ;

  q.submit([&](handler& h) {
    h.single_task<kernels>([=]() [[intel::kernel_args_restrict]] {

      host_ptr<int> in(in_host);
	    host_ptr<long> out(out_host);

      // define two registers
      fpvec<int> dataVec;
      fpvec<int> resVec;

      // initialize resVec with zero
      resVec = set1(0);

			// iterate over input data with a SIMD registers size of 512-bit (16 elements)
      for (int i_cnt = 0; i_cnt < iterations; i_cnt++) {
				    // Load complete CL (register) in one clock cycle
            dataVec = load<int>(in, i_cnt);

            // elementwise addition
            resVec = add(resVec, dataVec);

			}
      // horizontal aggregation at the end
      out[0] = hadd<int>(resVec);

    });
  }).wait();
}