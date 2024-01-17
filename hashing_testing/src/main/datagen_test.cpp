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
    size_t distinct = 1 * 50;
    size_t data_size = distinct * 3;
    size_t max_scale = 8;
    size_t hsize = distinct * 2;
    size_t max_test = 5;

    size_t distinct_max = hsize * 1;
    size_t collision_max = 8;

    auto hfunction = get_hash_function<ps_type>(HashFunction::MODULO);
    ps_type* data = (ps_type*) aligned_alloc(64, (data_size+2) * sizeof(ps_type));
    Datagenerator<ps_type> *datagen = nullptr;
    


    create_Datagenerator<ps_type>(datagen, distinct_max, hfunction, collision_max, seed);

    datagen->transform_hsize(hsize);

    


    for(size_t collisions = 0; collisions <= distinct; collisions += distinct/5){
        datagen->get_data_strided(data, data_size, distinct, collisions, seed);
        std::cout << "Col: " << collisions << std::endl << "\t";
        for(size_t i = 0; i < distinct; i++){
            if(i % (distinct / 5) == 0){
                std::cout << std::endl << "\t";
            }
            std::cout << data[i] % (hsize*1000) << "\t" << std::flush;
        }
        std::cout << std::endl << std::endl;

        for(float sel = 1; sel >= 0; sel -= 1./3){
            sel *= 100.;
            sel = (size_t) sel;
            sel /= 100.;
            std::cout << "\n\n\tsel: " << sel;
            size_t da = datagen->get_probe_strided(data, data_size,sel, seed, true);
            
            for(size_t i = 0; i < da; i++){
                if(i % (da / 5) == 0){
                    std::cout << std::endl << "\t\t";
                }
                std::cout << data[i] % (hsize*1000) << "\t" << std::flush;
            }
        }
        

        std::cout << std::endl << std::endl;
    }

}