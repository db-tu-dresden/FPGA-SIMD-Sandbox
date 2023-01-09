#include <iostream>
#include <stdint.h>
#include <vector>

#include <cstdlib>

#include "../operator/physical/group_count/scalar_group_count.hpp"
#include "../datagen/datagen.hpp"


// simple multiplicative hashing function
uint32_t hashx(uint32_t key, size_t HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

template<typename T>
T id_mod(T key, size_t HSIZE) {
    return key % HSIZE;
}

template <typename T>
void hashall(T from, T to, T step, size_t HSIZE, T(*hash_function)(T, size_t)){
    for(T i = from; i < to; i += step){
        std::cout << "\t" << hash_function(i, HSIZE);
    }
    std::cout << std::endl;
}

using ps_type = uint64_t;

int main(int argc, char** argv){
    size_t distinctValuesCount = 6;
    float scale = 1.8f;
    size_t HSIZE = (size_t)(scale * distinctValuesCount + 0.5f);
    size_t dataSize = 2000;
    ps_type* data = new ps_type[dataSize];

    generate_data<ps_type>(data, dataSize, distinctValuesCount, Density::SPARSE);

    // std::cout << std::endl << std::endl;
    // for(size_t i = 0; i < dataSize; i++){
    //     std::cout << "\t" << data[i];
    // }std::cout << std::endl << std::endl;

    Scalar_group_count<ps_type> x(HSIZE, &id_mod);
    x.create_hash_table(data, dataSize);
    x.print2();
    std::cout << "Expected count:\t" << dataSize << std::endl; 
    // hashall<uint32_t>(0, 10, 1, 4, &id_mod);
    // hashall<uint32_t>(0, 10, 1, 4, &hashx);

    std::cout << std::endl << std::endl;

    // for(int32_t i = 0; i < 50; i ++){
    //     std::cout << i << "\t" << noise(i,1) << "\t" << noise(i,1)%5 << std::endl;
    // }
}