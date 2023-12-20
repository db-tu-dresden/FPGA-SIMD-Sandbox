
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cmath>

#include <tslintrin.hpp>

int main(int argc, char** argv){

    using ps = tsl::simd<uint8_t, tsl::sse>;

    std::cout << ps::vector_element_count() << std::endl;
    std::cout << ps::vector_alignment() << std::endl;
}