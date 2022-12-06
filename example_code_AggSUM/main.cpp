#include <algorithm>
#include <array>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <vector>
#include <time.h>
#include <tuple>
#include <utility>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>



// Time
#include <sys/time.h>
// Sleep
#include <unistd.h>

#include "kernels.hpp"


using namespace sycl;
using namespace std::chrono;




////////////////////////////////////////////////////////////////////////////////
//// Board globals. Can be changed from command line.
// default to values in pac_s10_usm BSP
#ifndef DDR_CHANNELS
#define DDR_CHANNELS 4
#endif

#ifndef DDR_WIDTH
#define DDR_WIDTH 64 // bytes (512 bits)
#endif

#ifndef PCIE_WIDTH
#define PCIE_WIDTH 64 // bytes (512 bits)
#endif

#ifndef DDR_INTERLEAVED_CHUNK_SIZE
#define DDR_INTERLEAVED_CHUNK_SIZE 4096 // bytes
#endif

//constexpr size_t kDDRChannels = DDR_CHANNELS;
//constexpr size_t kDDRWidth = DDR_WIDTH;
constexpr size_t kDDRInterleavedChunkSize = DDR_INTERLEAVED_CHUNK_SIZE;
//constexpr size_t kPCIeWidth = PCIE_WIDTH;
////////////////////////////////////////////////////////////////////////////////


template<typename T>
bool validate(T *in_host, T *out_host, size_t size);

void exception_handler(exception_list exceptions);

// Function prototypes

////////////////////////////////////////////////////////////////////////////////


// main
int main(int argc, char* argv[]) {


  // make default input size enough to hide overhead
#ifdef FPGA_EMULATOR
  size_t size = kDDRInterleavedChunkSize * 4;
#else
  size_t size = kDDRInterleavedChunkSize * 16384;
#endif


  // the device selector
#ifdef FPGA_EMULATOR
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  // create the device queue
  auto props = property_list{property::queue::enable_profiling()};
  queue q(selector, exception_handler, props);

  // make sure the device supports USM device allocations
  device d = q.get_device();
  if (!d.get_info<info::device::usm_device_allocations>()) {
    std::cerr << "ERROR: The selected device does not support USM device"
              << " allocations\n";
    std::terminate();
  }
  if (!d.get_info<info::device::usm_host_allocations>()) {
    std::cerr << "ERROR: The selected device does not support USM host"
              << " allocations\n";
    std::terminate();
  }


	if ( argc != 2 ) // argc should be 2 for correct execution
	{
		size = 1024;
	}
	  else
	{
		size = atoi(argv[1]);
	}

	printf("Vector length: %zd \n", size);


 using Type = int;  // type to use for the test

  // Define for Allocate input/output data in pinned host memory
  // Used in all three tests, for convenience
  Type *in;
  //int *out;
  long *out_aggr;




	printf("\n \n ### aggregation ### \n\n");

    size_t number_CL = 0;

	if(size % 16 == 0)
	{
		number_CL = size / 16;
	}
	else
	{
		number_CL = size / 16 + 1;
	}

	printf("Number CLs: %zd \n", number_CL);




  // Allocate input/output data in pinned host memory
  // Used in both tests, for convenience

  if ((in = malloc_host<Type>(number_CL*16, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'in'\n";
    std::terminate();
  }
  if ((out_aggr = malloc_host<long>(1, q)) == nullptr) {
    std::cerr << "ERROR: could not allocate space for 'out'\n";
    std::terminate();
  }

	// Init input buffer
	for(int i=0; i< (number_CL*16); ++i)
    {
		if(i < size)
		{
			in[i] = 1;
		}
		else
		{
			in[i] = 0;
		}
    }

	// Init output buffer
	out_aggr[0] = 312;

	printf("Init buffers done with linear series \n");

	printf("in[0],in[1],in[2] ... : %i,%i,%i ... \n",in[0],in[1],in[2]);
	printf(" ... in[size-1] : %i \n",in[size-1]);

	printf("out[0]: %ld \n\n",out_aggr[0]);

  // track timing information, in ms
  double pcie_time=0.0;

  try {

    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Running HOST-Aggregation test " << "with an " << "size of " << size << " numbers \n";


	  aggregation_kernel(q, in, out_aggr, 16); // dummy run to program FPGA, dont care first run for measurement
    auto start = high_resolution_clock::now();
	  aggregation_kernel(q, in, out_aggr, number_CL*16);
    auto end = high_resolution_clock::now();
    duration<double, std::milli> diff = end - start;
    pcie_time=diff.count();
  

    ////////////////////////////////////////////////////////////////////////////
  } catch (exception const& e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  printf("out[0]: %ld \t (check with gaussian sum formula) \n\n",out_aggr[0]);
  printf("out: %ld \n",out_aggr[0]);


  // free USM memory
  sycl::free(in, q);
  sycl::free(out_aggr, q);

  // print result


    double input_size_mb = size * sizeof(Type) * 1e-6;

	printf("input_size_mb %lf \n", input_size_mb);


    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb / (pcie_time * 1e-3)) << " MB/s\n";

}



void exception_handler (exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
        std::rethrow_exception(e);
    } catch(exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
            << e.what() << std::endl;
    }
  }
}
