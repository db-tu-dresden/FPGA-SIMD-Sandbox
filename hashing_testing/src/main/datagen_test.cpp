#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>

#include "main/utility.hpp"
#include "main/hash_function.hpp"
#include "main/datagenerator/datagenerator.hpp"

template<typename T>
void create_Datagenerator(
    Datagenerator<T> *& datagen,
    size_t hsize, 
    size_t (*hash_function)(T, size_t),
    size_t max_collision_size, 
    size_t number_seed)
{
    if(datagen != nullptr){
        delete datagen;
    }
    datagen = new Datagenerator<T>(hsize, hash_function, max_collision_size, number_seed);
}


int main(int argc, char** argv){
    fill_tab_table();

    using ps_type = uint16_t;

    size_t seed = 11;
    size_t distinct = 1 * 128;
    size_t data_size = distinct * 10;
    size_t max_scale = 8;
    size_t hsize = distinct * 2;
    size_t max_test = 5;

    size_t distinct_max = distinct;
    size_t collision_max = 10;

    auto hfunction = get_hash_function<ps_type>(HashFunction::MODULO);
    ps_type* data = (ps_type*) aligned_alloc(64, (data_size+24) * sizeof(ps_type));
    Datagenerator<ps_type> *datagen = nullptr;
    
    std::chrono::high_resolution_clock::time_point time;

    std::cout << "get-data-testing" << std::endl;
    time = time_now();
    
    std::cout << "\tDatageneration\t" << std::flush;
    create_Datagenerator<ps_type>(datagen, distinct_max, hfunction, collision_max, seed);
    std::cout << "\n\t    it took: "; print_time(time); time = time_now(); 
    
    datagen->transform_hsize(hsize+1);
    std::cout << "\n\t    it took: "; print_time(time); time = time_now(); 
    

    size_t collisions = 0;
    for(size_t collisions = 0; collisions <= distinct; collisions += distinct/4){
        std::cout << "\tCollisions = " << collisions << std::flush;
        datagen->get_data_strided(data, data_size, distinct, collisions, seed);
        std::cout << "\n\t    it took: "; print_time(time); time = time_now();     
    }

    std::cout << "gen-time-testing" << std::endl;
    for(size_t i = 0; i < 8; i++){
        std::cout << "\td: "<< distinct_max << "\tc: "<< collision_max << "\t"<< std::flush;
        create_Datagenerator<ps_type>(datagen, distinct_max, hfunction, collision_max, seed);
        std::cout << "\n\t    it took: "; print_time(time); time = time_now();    
        distinct_max /= 2;
        collision_max *= 2;
    }


    std::cout << "DATATESTING: " << distinct << std::endl;
    std::cout << "settings test" << std::endl;
    
}