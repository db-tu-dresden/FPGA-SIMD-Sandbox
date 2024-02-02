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

template<typename T>
void print_data(T* data, size_t data_size, size_t mod = 0){
    size_t breaks  = 10;
    bool small = mod != 0;
    
    for(size_t i = 0, r = 0; i < data_size; i++, r++){        
        if(r >= breaks){
            r = 0;
            std::cout << "\n";
        }
        if(small){
            std::cout << data[i] % mod << "\t";
        }else{
            std::cout << data[i] << "\t";
        }
    }
    std::cout << "\n";
}

int main(int argc, char** argv){
    fill_tab_table();

    using ps_type = uint64_t;

    size_t seed = 11;
    size_t distinct =  100;
    size_t data_size = distinct * 5;
    size_t max_scale = 8;
    size_t test_scale = 2;
    size_t hsize = distinct * test_scale;
    size_t max_test = 5;

    size_t distinct_max = hsize * max_scale;
    size_t collision_max_gen = distinct / 2;
    size_t collision_max_dat = 11;// distinct / 10;

    auto hfunction = get_hash_function<ps_type>(HashFunction::MODULO);
    ps_type* data = (ps_type*) aligned_alloc(64, data_size * sizeof(ps_type));
    Datagenerator<ps_type> *datagen = nullptr;
    

    time_stamp a = time_now();
    create_Datagenerator<ps_type>(datagen, distinct_max, hfunction, collision_max_gen, seed);
    std::cout << "\nraw data gen:\t";print_time(a, true);
    // datagen->print_data();
    // exit(1);
    time_stamp b = time_now();
    datagen->transform_hsize(hsize);
    std::cout << "\nhsize change:\t";print_time(b, true);

    time_stamp c = time_now();
    datagen->get_data_blocked(data, data_size, distinct, collision_max_dat, seed + 1);
    std::cout << "\nbuilt data gen:\t";print_time(c, true);
    print_data(data, data_size);
    print_data(data, data_size, hsize);

    time_stamp d = time_now();
    datagen->get_probe_blocked(data, data_size, 1, seed + 2);
    std::cout << "\nprobe data gen:\t";print_time(d, true);
    print_data(data, data_size);
}