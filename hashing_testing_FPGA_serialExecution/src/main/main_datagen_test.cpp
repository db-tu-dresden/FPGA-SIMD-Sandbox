#include <iostream>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>

#include "datagen.hpp"




template <typename T>
void analyse1(T* data, size_t data_size){
    uint64_t bits = sizeof(T) * 8;
    size_t total_bits = 0;
    size_t *bit_at_position = new size_t[bits];
    
    for(size_t i = 0; i < bits; i++){
        bit_at_position[i] = 0;
    }
    
    for(size_t i = 0; i < data_size; i++){
        size_t b_count = 0;
        T val = data[i];
        for(size_t b = 0; b < bits; b++){
            bit_at_position[b] += (val >> b) & 0x1;
            b_count += (val >> b) & 0x1;
        }
        total_bits += b_count;
    }
    
    std::cout << "Average Nr of Bits set:\t" << (total_bits * 1.) / data_size << "\t of " << bits << " Bits\n";

    std::cout << "Nr of Bits per position\n";
    for(size_t i = 0; i < bits; i++){
        std::cout << "\t" << (int)((bit_at_position[i]* 1.) / data_size * 1000)/1000.;
        if((i+1)%16 == 0){
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

template <typename T>
void analyse2(T* data, size_t data_size){
    uint64_t bits = sizeof(T) * 8;

    size_t bit_at_position[bits][bits][4];
    for(size_t a = 0; a < bits; a++){
        for(size_t b = 0; b < bits; b++){
            for(size_t i = 0; i < 4; i++){
                bit_at_position[a][b][i] = 0;
            }
        }
    }

    for(size_t i = 0; i < data_size; i++){
        T val = data[i];
        for(size_t a = 0; a < bits; a++){
            for(size_t b = 0; b < bits; b++){
                size_t pos = 0;
                pos += ((val >> a) & 0x1) << 1;
                pos += (val >> b) & 0x1;

                bit_at_position[a][b][pos]++;
            }
        }
    }

    for(size_t a = 0; a < bits; a++){
        for(size_t b = 0; b < bits; b++){
            std::cout << a << "\t" << b << ":\t";
            double x = 0;
            for(size_t i = 0; i < 4; i++){
                x += (i + 1) * (bit_at_position[a][b][i] * 1.) / data_size;
                std::cout << "\t" << (bit_at_position[a][b][i] * 1.) / data_size;
            }
            std::cout << "\t" << (int)(x * 1000)/1000.0;
            if(b%2 == 1){
                std::cout << std::endl;
            }else{
                std::cout << "\t\t\t";
            }
        }
    }
    



}



using ps_type = uint64_t;

int main(int argc, char** argv){

    // size_t nr_of_values = 100000;
    // size_t data_size = nr_of_values *= 20;
    // ps_type* data = new ps_type[nr_of_values];
    // std::cout << "Data generation\n";
    // generate_data2<ps_type>(data, data_size, nr_of_values, Density::SPARSE);  
    // std::cout << "Analyse 1\n";
    // analyse1<ps_type>(data, nr_of_values);
    // std::cout << "Analyse 2\n";
    // analyse2<ps_type>(data, nr_of_values);

    // bool blocked [] = {0, 0, 1, 0, 0, 1, 0};
    // size_t res = find_place(blocked, 7, 1, 3);
    // std::cout << "RESULT:\t" << res << std::endl;

    uint32_t *data = new uint32_t[20];
    generate_collision_data<uint32_t>(data, 20, 13, 15, 3, 1);

}