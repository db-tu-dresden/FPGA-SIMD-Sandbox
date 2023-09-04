#include "main/datagenerator/datagenerator.hpp"
#include "main/datagenerator/datagen_help.hpp"
#include <iostream>

template<typename T>
bool vector_contains(std::vector<T> vec, T value){
    for(T val: vec){
        if(value == val){
            return true;
        }
    }
    return false;
}

template<typename T>
void print_vector(std::vector<T> numbers){
    size_t count = 0;
    for(T i: numbers){
        if(count == 10){
            count = 0;
            std::cout << "\n";
        }
        std::cout << (uint64_t)(i) << "\t";
        count++;
    }
    std::cout << std::endl;
}


template<typename T> 
void distribute(
    T*& result, 
    std::vector<T> raw_collision,
    std::vector<T> raw_non_collision,
    size_t data_size,
    size_t distinct_values,
    size_t seed,
    bool non_collisions_first,
    bool evenly_distributed
){
    // std::cout << "collision\n";
    // print_vector(raw_collision);
    // std::cout << "non collision\n";
    // print_vector(raw_non_collision);
    std::vector<size_t> max_occurences_left;
    std::vector<T> value;

    size_t dist = data_size;
    if(evenly_distributed){
        dist = (dist+distinct_values-1) / distinct_values;
    }
    // std::cout << "DIST!\t" << dist << std::endl;
    size_t id = 0;

    for(; id < raw_non_collision.size(); id++){
        value.push_back(raw_non_collision[id]);
        max_occurences_left.push_back(dist - 1);
        result[id] = raw_non_collision[id];
    }
    for(T val: raw_collision){
        value.push_back(val);
        max_occurences_left.push_back(dist);
    }

    size_t help_id = 0;
    size_t stepper = 2;
    while(value.size() % stepper == 0){
        stepper++;
    }

    while(id < data_size){
        size_t ran = noise(help_id++, seed) % value.size();
        size_t occ = max_occurences_left[ran];
        while(occ == 0){
            ran = (ran + stepper) % value.size();
            occ = max_occurences_left[ran];
        }
        max_occurences_left[ran]--;
        T val = value[ran];
        result[id] = val;
        id++;
    }
}



template<typename T> 
size_t generate_strided_data(
    T*& result, 
    size_t data_size,
    size_t distinct_values,
    size_t hsize,
    hash_fptr<T> hash_function,
    size_t collision_size,
    size_t seed,
    bool non_collisions_first,
    bool evenly_distributed
){
    std::vector<T> raw_collision;
    std::vector<T> raw_non_collision;
    
    generate_strided_data_raw<T>(raw_collision, raw_non_collision, distinct_values, hsize, hash_function, collision_size, seed);

    distribute<T>(result, raw_collision, raw_non_collision, data_size, distinct_values, seed, non_collisions_first, evenly_distributed);

    return raw_collision.size() + raw_non_collision.size();
}

// generates a list of indices to for which to not generate collisions
// than selects random numbers according to that list.
// values that are not on the list generate a collision instead.

template<typename T>
void generate_strided_data_raw(
    std::vector<T> &collision_data,
    std::vector<T> &non_collision_data,
    size_t distinct_values,
    size_t hsize,
    hash_fptr<T> hash_function,
    size_t collision_size,
    size_t &seed
){
    if(collision_size > distinct_values){
        collision_size = distinct_values;
    }
    if(collision_size > 0){
        collision_size--;
    }
    //collisions size of 1 and 0 is the same. there can't be a one value collision. 

    collision_data.clear();
    non_collision_data.clear();
    bool inverse = collision_size >= distinct_values * 0.5;
    // if(collision_size >= distinct_values){
    //     collision_size = distinct_values - 1;
    // }
    size_t to_generate_ids = (collision_size * !inverse) + ((distinct_values - 1 - collision_size) * inverse);
    
    // std::cout << "collision_size " << collision_size << std::endl;
    // std::cout << "inverse " << inverse << std::endl;
    // std::cout << "to_gen " << to_generate_ids << std::endl;

    std::vector<size_t> ids;
    size_t help = 0;
    for(size_t c = 0; c < to_generate_ids; c++){
        size_t nid = (noise(c + help++, seed) % (distinct_values - 1)) + 1;
        bool okay = !vector_contains<size_t>(ids, nid);
        if(okay){
            ids.push_back(nid);
        }else{
            c--;
        }
    }
    seed++;
// print_vector(ids);
    std::vector<bool> collide;
    for(size_t i = 1; i < distinct_values; i++){
        bool is_in = vector_contains<size_t>(ids, i);
        bool insert = (is_in != inverse);
        collide.push_back(insert);
    }
// print_vector(collide);
    std::multimap<size_t, T> all_numbers;
    all_number_gen<T>(all_numbers, hash_function, hsize, collision_size + 2, seed++);

    size_t start_pos = noise(0, seed++) % hsize;
    double step_size = (hsize * 1.0) / distinct_values;
    double current_pos = start_pos;

    typename std::multimap<size_t, T>::iterator itLow, it;    
    itLow = all_numbers.lower_bound(start_pos);

    collision_data.push_back(itLow->second);    
    itLow++;
    it = itLow;
    current_pos += step_size;

    for(size_t i = 0; i < distinct_values-1; i++){
        size_t pos = (size_t)(current_pos) % hsize;
        if(collide[i]){
            collision_data.push_back(it->second);
            it++;
        }else{
            non_collision_data.push_back(all_numbers.lower_bound(pos)->second);
        }
        current_pos += step_size;
    }
}



template<typename T>
void template_help_function(){
    size_t count = 10;
    T * wowie = new T[count];
    generate_strided_data<T>(wowie, count, count/3, count/2, get_hash_function<T>(HashFunction::MODULO),0, 0);
};

// templatisierungs stuff. maybe easier a nother way.
template void template_help_function<uint64_t>();
template void template_help_function<uint32_t>();
template void template_help_function<uint16_t>();
template void template_help_function<uint8_t>();
template void template_help_function<int64_t>();
template void template_help_function<int32_t>();
template void template_help_function<int16_t>();
template void template_help_function<int8_t>();

template size_t generate_strided_data<>(
    uint32_t *&, size_t, size_t, size_t, 
    hash_fptr<uint32_t>, size_t, size_t, bool, bool);
// template size_t generate_strided_data<>(
//     uint16_t *&, size_t, size_t, size_t, 
//     hash_fptr<uint16_t>, size_t, size_t, bool, bool);
// template size_t generate_strided_data<>(
//     uint8_t *&, size_t, size_t, size_t, 
//     hash_fptr<uint8_t>, size_t, size_t, bool, bool);

// template size_t generate_strided_data<>(
//     int64_t *&, size_t, size_t, size_t, 
//     hash_fptr<int64_t>, size_t, size_t, bool, bool);
// template size_t generate_strided_data<>(
//     int32_t *&, size_t, size_t, size_t, 
//     hash_fptr<int32_t>, size_t, size_t, bool, bool);
// template size_t generate_strided_data<>(
//     int16_t *&, size_t, size_t, size_t, 
//     hash_fptr<int16_t>, size_t, size_t, bool, bool);
// template size_t generate_strided_data<>(
//     int8_t *&, size_t, size_t, size_t, 
//     hash_fptr<int8_t>, size_t, size_t, bool, bool);