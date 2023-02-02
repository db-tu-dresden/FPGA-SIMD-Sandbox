#ifndef TUD_HASHING_TESTING_DATAGEN_HELP
#define TUD_HASHING_TESTING_DATAGEN_HELP

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>

//TODO CHECK IF it is okay to use.
uint64_t noise(size_t position, size_t seed){
    size_t BIT_NOISE1 = 0x68E31DA4;
    size_t BIT_NOISE2 = 0xB5297A4D;
    size_t BIT_NOISE3 = 0x1B56C4E9;

    uint64_t mangled = position;
    mangled *= BIT_NOISE1;
    mangled += seed;
    mangled ^= (mangled << 13);
    mangled += BIT_NOISE2;
    mangled ^= (mangled >> 7);
    mangled *= BIT_NOISE3;
    mangled ^= (mangled << 17);

    return mangled;
}















void next_position(size_t &pos, size_t &budget, size_t HSIZE, size_t seed){
    if(budget == 0){
        budget = 1;
    }
    if(budget == 1){
        pos = (pos + 1) % HSIZE;
    }
    size_t offset = noise(budget + pos, seed) % budget;
    budget -= offset;
    pos = (pos + 1 + offset) % HSIZE;
}


void generate_collision(
    std::vector<size_t> *result,
    std::multimap<size_t, size_t> *numbers,
    size_t HSIZE,
    size_t start_pos,
    size_t collision_lenght
){
    if(collision_lenght > HSIZE || HSIZE == 0){
        return;
    }

    size_t pos = start_pos % HSIZE;
    size_t data_to_generate = collision_lenght;
    
    bool GENERATE = data_to_generate > 0;
    
    while(GENERATE){
        std::multimap<size_t, size_t>::iterator itLow, itUp;
        itLow = numbers->lower_bound(pos);
        itUp = numbers->upper_bound(pos);

        for(std::multimap<size_t, size_t>::iterator it = itLow; it != itUp && GENERATE; it++){
            result->push_back(it->second);
            GENERATE = --data_to_generate > 0;
        }
        pos = (pos + 1) % HSIZE;
    }
}


void generate_cluster(
    std::vector<size_t> *result,
    std::multimap<size_t, size_t> *numbers,
    size_t HSIZE,
    size_t start_pos,
    size_t cluster_lenght
){
    if(cluster_lenght > HSIZE || HSIZE == 0){
        return;
    }

    size_t pos = start_pos % HSIZE;
    for(size_t i = 0; i < cluster_lenght; i++){
        std::multimap<size_t, size_t>::iterator it = numbers->find(pos);
        result->push_back(it->second);
        pos = (pos + 1) % HSIZE;
    }
}


template<typename T>
void generate_benchmark_data(
    T*& result, 
    size_t data_size,
    std::vector<size_t> *numbers,
    size_t seed
){
    for(size_t i = 0; i < data_size; i++){
        size_t ran_id = noise(i, seed) % numbers->size();
        result[i] = (T)(numbers->at(ran_id));
    }
}

#endif //TUD_HASHING_TESTING_DATAGEN_HELP