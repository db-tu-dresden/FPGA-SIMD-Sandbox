#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>

#include "main/utility.hpp"
#include "main/hash_function.hpp"
#include "main/datagenerator/datagenerator.hpp"

int main(int argc, char** argv){
    fill_tab_table();
    

    std::cout << "DATATESTING:\n";
    size_t seed = 15839355070940700635;
    size_t data_size = 33554432;
    size_t distinct = 16384;
    size_t scale = 2;
    size_t hsize = distinct * scale;
    size_t collision_count = 2;

    std::cout << "settings test" << std::endl;
    uint32_t* data = (uint32_t*) aligned_alloc(64, (data_size+24) * sizeof(uint32_t));
    std::cout << "settings test done!" << std::endl;
    generate_strided_data<uint32_t>(data, data_size, distinct, hsize, get_hash_function<uint32_t>(HashFunction::MODULO), collision_count, seed);

    for(size_t collision_count = 0; collision_count < 3; collision_count++){
        std::cout <<"Collision Count:" << collision_count << std::endl;
        generate_strided_data<uint32_t>(data, data_size, distinct, hsize, get_hash_function<uint32_t>(HashFunction::MODULO), collision_count, seed);
        // std::cout << std::endl;
        // for(size_t i = 0; i < data_size; i++){
        //     if(i % 1 == 0){
        //         std::cout << std::endl;
        //     }
        //     // std::cout <<"\t";
        //     size_t placement = data[i] %hsize;
        //     // if(placement < 10){
        //     //     std::cout << " ";
        //     // }
        //     std::cout << data[i];//%(1000);
        // }std::cout << "\n\n\n\n";
    }


}