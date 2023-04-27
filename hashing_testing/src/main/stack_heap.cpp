#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include "utility.hpp"
#include "datagenerator/datagen.hpp"

void swap(uint64_t &a, uint64_t &b){
    a ^= b;
    b ^= a;
    a ^= b;
}

void sort(uint64_t *data, size_t data_size){
    for(size_t i = 0; i < data_size; i++){
        for(size_t e = i + 1; e < data_size; e++){
            if(data[i] > data[e]){
                swap(data[i], data[e]);
            }
        }
    }
}

template<typename T>
T aggregate_seq(T* data0, T* data1, size_t data_size){
    T res = 0;
    for(size_t i = 0; i < data_size; i++){
        res += (data0[i] ^ data1[i]);
    }
    return res;
}


template<typename T>
T aggregate_ran(T* data0, T* data1, size_t data_size, size_t seed){
    T res = 0;
    for(size_t i = 0; i < data_size; i++){
        size_t r_id0 = noise(i, seed) % data_size;
        size_t r_id1 = noise(i, seed + r_id0) % data_size;

        res += (data0[r_id0] ^ data1[r_id1]);
    }
    return res;
}


//prints in ms
template<typename T>
void benchmark(T* data0, T* data1, size_t data_size, size_t repeats, size_t seed){
    
    double percent = 0.8;

    uint64_t * time_s = (uint64_t*) malloc(repeats * sizeof(uint64_t));
    uint64_t * time_r = (uint64_t*) malloc(repeats * sizeof(uint64_t));
    
    double t_seq = 0.0;
    double t_ran = 0.0;
    std::chrono::high_resolution_clock::time_point a, b;
    
    T ret = 0;

    for(size_t i = 0; i < repeats; i++){
        
        a = time_now();
        ret += aggregate_seq(data0, data1, data_size);
        b = time_now();
        time_s[i] = duration_time(a, b);

        a = time_now();
        ret += aggregate_ran(data0, data1, data_size, seed);
        b = time_now();
        time_r[i] = duration_time(a, b);
    }

    sort(time_s, repeats);
    sort(time_r, repeats);

    size_t useful_k = (repeats * percent + 0.5);
    useful_k &= 0xFFFFFFFFFFFFFFFE;

    size_t ignore = (repeats - useful_k) / 2;
    

    for(size_t i = ignore; i < repeats - ignore; i++){
        t_seq += time_s[i];
        t_ran += time_r[i];
    }
    free(time_s);
    free(time_r);


    // t_seq /= 1000000;
    t_seq /= (repeats - ignore * 2);
    t_seq = (size_t)(t_seq);
    t_ran /= 1000;
    t_ran /= (repeats - ignore * 2);
    t_ran = (size_t)(t_ran);

    std::cout << t_seq << "\t" << t_ran << "\t" << ret << std::endl;
}

template<typename T>
void fill(T* data, size_t data_size, size_t seed){
    for(size_t i = 0; i < data_size; i++){
        data[i] = (T)noise(i, seed);
    }
}



int main(int argc, char** argv){
    using type = uint32_t;
    size_t stack_size = 8192; // report of: ulimit -a 
    stack_size -= 16;
    size_t element_count = stack_size * 1024 / sizeof(type) / 2; // this should fit into the stack

    size_t repeats = 500;

    
    srand(std::time(nullptr));
    size_t seed = std::rand();

    std::cout << "stack_heap_test using " << element_count << " values and seed: " << seed <<  std::endl;

    type stack0[element_count];
    type stack1[element_count];

    type * heap0 = (type*) aligned_alloc(512, element_count * sizeof(type));
    type * heap1 = (type*) aligned_alloc(512, element_count * sizeof(type));;

    fill<type>(stack0, element_count, seed + 1);
    fill<type>(stack1, element_count, seed + 2);
    fill<type>(heap0, element_count, seed + 1);
    fill<type>(heap1, element_count, seed + 2);

    std::cout << "RESULT\tSEQ(ns)\tRAN(µs)\tret_val\n";
    std::cout << "STACK \t";
    benchmark<type>(stack0, stack1, element_count, repeats, seed);
    std::cout << "HEAP  \t";
    benchmark<type>(heap0, heap1, element_count, repeats, seed);
    std::cout << "MIXED \t";
    benchmark<type>(stack0, heap1, element_count, repeats, seed);
    
    free(heap0);
    free(heap1);
 
    
    return 0;
}