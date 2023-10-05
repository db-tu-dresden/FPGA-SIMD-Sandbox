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
    

    size_t seed = 01;
    size_t distinct = 16 * 1024;
    size_t data_size = distinct * 1024;
    size_t max_scale = 8;
    size_t hsize = distinct * 2;
    size_t max_test = 5;

    size_t distinct_max = distinct * 1024 * 4;
    size_t collision_max = 16;

    uint32_t* data = (uint32_t*) aligned_alloc(64, (data_size+24) * sizeof(uint32_t));
    Datagenerator<uint32_t> *datagen = nullptr;
    
    auto hfunction = get_hash_function<uint32_t>(HashFunction::MODULO);
    std::chrono::high_resolution_clock::time_point time;

    std::cout << "get-data-testing" << std::endl;
    time = time_now();
    
    std::cout << "\tDatageneration\t" << std::flush;
    create_Datagenerator(datagen, distinct_max, hfunction, collision_max, seed);
    std::cout << "\n\t    it took: "; print_time(time); time = time_now(); 
    
    datagen->transform_hsize(hsize);
    std::cout <<"\ttransform:\t"<< datagen->transform_finalise();
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
        create_Datagenerator(datagen, distinct_max, hfunction, collision_max, seed);
        std::cout << "\n\t    it took: "; print_time(time); time = time_now();    
        distinct_max /= 2;
        collision_max *= 2;
    }


    std::cout << "DATATESTING: " << distinct << std::endl;
    std::cout << "settings test" << std::endl;
    
 
    // size_t h =  1024 * 16; // * 1024
    // size_t c = 32;
    // std::cout << "generate:\t" << h * c << std::endl;
    // std::cout << "\nvariation testing:"<< std::endl;
    // for(size_t i = 0; h > 0 && i < 2; h/=8, c*=4, i++){
    //     std::cout << "\thsize:" << h << "\t collisions:" << c << "   \tit took: " << std::flush;
    //     time = time_now();
    //     create_Datagenerator(datagen, h, hfunction, c, seed);
    //     print_time(time); 
    // }
    // std::cout << "\thsize: 32 * 1024\t collisions: 512  \tit took: ";
    // time = time_now();
    // create_Datagenerator(datagen, 4 * 16 * 1024 * 2, hfunction, 1024 /2, seed);
    // print_time(time); 

    // std::cout << "\thsize: 1024 * 1024\t collisions: 16  \tit took: ";
    // time = time_now();
    // create_Datagenerator(datagen, 4 * 16 * 1024 * 128, hfunction, 1024 / 128, seed);
    // print_time(time); 






//     std::cout << "\nraw data generation time" << std::endl;
//     size_t collision_count = 0;
//     for(size_t collision_count_id = 0; collision_count_id < 2; collision_count_id++){
//         if(datagen != nullptr){
//             delete datagen;
//         }
//         std::cout <<"Collision Count: " << collision_count << std::flush;
//         std::chrono::high_resolution_clock::time_point begin = time_now();
//         datagen = new Datagenerator<uint32_t>(hsize, get_hash_function<uint32_t>(HashFunction::MODULO), collision_count , seed);
//         std::cout << "\tit took: ";
//         print_time(begin); 
        
//         collision_count += distinct/(3-1);
//     }

//     // datagen->transform_collision(distinct /3);
//     std::cout << "\nresult generation time" << std::endl;
//     collision_count = 0;
//     for(size_t collision_count_id = 0; collision_count_id < max_test; collision_count_id++){
//         std::cout <<"Collision Count: " << collision_count << std::flush;
//         std::chrono::high_resolution_clock::time_point begin = time_now();
//         datagen->get_data_strided(data, data_size, distinct, collision_count, seed);
//         std::cout << "\tit took: ";
//         print_time(begin); 
//         collision_count += distinct/(max_test-1);
//     }

// return 0;
//     create_Datagenerator(datagen, distinct, hfunction, 32, seed);

//     std::cout << "\ntransformation time" << std::endl;
//     size_t scale = 1;
//     distinct = 16 * 1024;
//     for(scale = 1; scale <= max_scale; scale *= 2){
//         std::cout <<"Scale " << scale << std::flush;
//         std::chrono::high_resolution_clock::time_point begin = time_now();
//         datagen->transform_hsize(distinct * scale);
//         // datagen->transform_collision(distinct * scale);
//         std::cout <<"  "<< datagen->transform_finalise();
//         std::cout << "\tit took: ";
//         print_time(begin); 
//         // datagen->get_data_strided(data, data_size, distinct, distinct, seed);

//     }
//     //testing

//     std::chrono::high_resolution_clock::time_point begin = time_now();
//     datagen->revert();
//     print_time(begin); 
    


//     std::cout << "\nscale time raw" << std::endl;
//     scale = 1;
//     for(scale = 1; scale <= max_scale; scale *= 2){
//         if(datagen != nullptr){
//             delete datagen;
//         }
//         std::cout <<"Scale " << scale << std::flush;
//         std::chrono::high_resolution_clock::time_point begin = time_now();
//         std::cout << "\tit took: ";
//         datagen = new Datagenerator<uint32_t>(distinct * scale, get_hash_function<uint32_t>(HashFunction::MODULO), distinct, seed);
//         print_time(begin); 
//     }


    // for(size_t i = 0; i < data_size; i++){
    //     if(i % 20 == 0){
    //         std::cout << std::endl;
    //     }
    //     std::cout << data[i] % hsize << ":" << data[i] % 1000<<"\t";
    // }
    // std::cout << "\n\n";
}