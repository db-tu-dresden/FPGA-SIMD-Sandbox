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



template<typename T>
bool enough_values_per_bucket(
    std::multimap<size_t, T> &numbers,
    size_t HSIZE,
    size_t min_numbers
){
    for(size_t i = 0; i < HSIZE; i++){
        if(numbers.count(i) < min_numbers){
            return false;
        }
    }
    return true;
}

template<typename T>
bool vector_contains(
    std::vector<T> *vec,
    T value
){
    for(T val: *vec){
        if(value == val){
            return true;
        }
    }
    return false;
}

void next_position(size_t &pos, size_t &budget, size_t HSIZE, size_t seed){
    if(budget == 0){
        budget = 1;
    }
    if(budget == 1){
        pos = (pos + 1) % HSIZE;
        return;
    }
    size_t offset = noise(budget + pos, seed) % budget;
    budget -= offset;
    pos = (pos + 1 + offset) % HSIZE;
}

template<typename T>
void generate_random_values(
    std::multimap<size_t, T> &numbers,
    size_t (*hash_function)(T, size_t),
    size_t number_of_values,
    size_t HSIZE,
    size_t seed,
    size_t min_numbers = 2
){
    size_t number_retry = 0;
    size_t retry = 0;
    size_t wanted_values = number_of_values;
    do{
        while(numbers.size() < wanted_values){
            T num;
            do{
                num = noise(numbers.size() + number_retry, seed);
                number_retry += num == 0;
            }while(num == 0);
            
            numbers.insert(std::pair<size_t, T>(hash_function(num, HSIZE), (T)(num)));
        }
        wanted_values += number_of_values;
        retry++;
        if(retry > 10 && !enough_values_per_bucket(numbers, HSIZE, 1)){    // second part tells us if every value got generated atleast once.
            throw std::runtime_error("bad hash function!");
        }
    }while(!enough_values_per_bucket(numbers, HSIZE, min_numbers));
}

// tries to generate a collision big enough form one bucket. 
// might need to get data from neighbouring buckets.
template<typename T>
void generate_collision(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
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
    
    typename std::multimap<size_t, T>::iterator itLow, itUp, it;
    while(GENERATE){
        itLow = numbers->lower_bound(pos);
        itUp = numbers->upper_bound(pos);

        for(it = itLow; it != itUp && GENERATE; it++){
            T n_val = it->second;
            bool contains = vector_contains<T>(result, n_val);
            if(!contains){
                result->push_back(it->second);
                GENERATE = --data_to_generate > 0;
            }
        }
        pos = (pos + 1) % HSIZE;
    }
}

// tries to generate a collision big enough form one bucket. 
// might need to get data from neighbouring buckets.
template<typename T>
void generate_collision_soaov(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t &h_pos,
    size_t &e_pos,
    size_t elements,
    size_t collision_lenght
){
    if(collision_lenght > (HSIZE * elements) || (HSIZE * elements) == 0){
        return;
    }

    size_t pos = h_pos % HSIZE;
    size_t data_to_generate = collision_lenght;
    
    bool GENERATE = data_to_generate > 0;
    
    typename std::multimap<size_t, T>::iterator itLow, itUp, it;
    while(GENERATE){
        itLow = numbers->lower_bound(pos);
        itUp = numbers->upper_bound(pos);

        for(it = itLow; it != itUp && GENERATE; it++){
            T n_val = it->second;
            bool CONTAINS = vector_contains<T>(result, n_val);
            
            if(!CONTAINS){
                result->push_back(n_val);
                GENERATE = --data_to_generate > 0;
                e_pos = (e_pos + 1) % elements;
                h_pos = (h_pos + (e_pos == 0)) % HSIZE;
            }
        }
        pos = (pos + 1) % HSIZE;
    }
}

template<typename T>
void generate_cluster_soaov(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t &h_pos,
    size_t &e_pos,
    size_t elements,
    size_t cluster_lenght
){
    if(cluster_lenght > (HSIZE * elements) || (HSIZE * elements) == 0){
        return;
    }

    typename std::multimap<size_t, T>::iterator itLow, itUp, it;
    itLow = numbers->lower_bound(h_pos);
    itUp = numbers->upper_bound(h_pos);
    it = itLow;
    size_t i = 0;
    while(i < cluster_lenght){
        T n_val = it->second;
        bool CONTAINS = vector_contains<T>(result, n_val);
        if(!CONTAINS){
            result->push_back(n_val);
            e_pos = (e_pos + 1) % elements;
            h_pos = (h_pos + (e_pos == 0)) % HSIZE;
            i++;
        }

        if(e_pos == 0 && !CONTAINS){
            itLow = numbers->lower_bound(h_pos);
            itUp = numbers->lower_bound(h_pos);
            it = itLow;
        }else{
            it++;
        }
    }

}



template<typename T>
void generate_cluster(
    std::vector<T> *result,
    std::multimap<size_t, T> *numbers,
    size_t HSIZE,
    size_t start_pos,
    size_t cluster_lenght
){
    if(cluster_lenght > HSIZE || HSIZE == 0){
        return;
    }

    size_t pos = start_pos % HSIZE;
    for(size_t i = 0; i < cluster_lenght; i++){
        typename std::multimap<size_t, T>::iterator it;
        it = numbers->find(pos);
        result->push_back(it->second);
        pos = (pos + 1) % HSIZE;
    }
}


template<typename T>
void generate_benchmark_data(
    T*& result, 
    size_t data_size,
    std::vector<T> *numbers,
    size_t seed
){
    for(size_t i = 0; i < data_size; i++){
        size_t ran_id = noise(i, seed) % numbers->size();
        result[i] = (T)(numbers->at(ran_id));
    }
}

#endif //TUD_HASHING_TESTING_DATAGEN_HELP