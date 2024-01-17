#include "main/datagenerator/datagenerator.hpp"
#include <iostream>
#include <limits>
#include <omp.h>

template<typename T>
bool vector_contains(std::vector<T> vec, T value);
template<typename T>
bool vector_delete(std::vector<T>& x, size_t id);
template<typename T>
void vector_print(std::vector<T> x);
template<typename T>
void array_print(T* x, size_t len);

#define THREAD_COUNT 16


#include <chrono>
void p_time(size_t time_sec, bool new_line = true){
    size_t time_min = time_sec/60;
    time_sec -= time_min * 60;
    size_t time_hour = time_min/60;
    time_min -= time_hour * 60;
    if(time_hour < 10){
        std::cout << 0;
    }
    std::cout << time_hour << ":";
    if(time_min < 10){
        std::cout << 0;
    }
    std::cout << time_min << ":";
    if(time_sec < 10){
        std::cout << 0;
    }
    std::cout << time_sec;
    if(new_line){
        std::cout << std::endl;
    }
}
uint64_t dt_seconds (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
}
void p_time(std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end, bool new_line = true){
    uint64_t duration = dt_seconds(begin, end);
    p_time(duration, new_line);
}


///***************************************/
//* implementations of public functions *//
/***************************************///
template<typename T>
Datagenerator<T>::Datagenerator(
    size_t different_values, 
    size_t (*hash_function)(T, size_t),
    size_t max_collision_size, 
    size_t number_seed
):m_bucket_count{different_values},m_hash_function{hash_function},m_seed{number_seed}{

    m_original_bucket_count = different_values;
    if(m_original_bucket_count < 1){
        m_original_bucket_count = 1;
    }
    m_original_bucket_size = max_collision_size;
    if(m_original_bucket_size <= 1){
        m_original_bucket_size = 2;
    }
    
    m_original_data_matrix = new Data_Matrix<T>(different_values, max_collision_size, hash_function, number_seed);

    m_working_set_data_matrix = m_original_data_matrix;

    m_bucket_size = m_working_set_data_matrix->get_max_bucket_size();
    m_bucket_count = m_working_set_data_matrix->get_bucket_count();
}

template<typename T>
size_t Datagenerator<T>::get_data_bad(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t collision_count,
    size_t layout_seed,
    bool non_collision_first,
    bool evenly_distributed
){
    std::vector<T> normal_values;
    std::vector<T> collision_values;

    Datagenerator<T>::get_values_bad(
        normal_values, 
        collision_values, 
        distinct_values,
        collision_count,
        layout_seed
    );
    
    Datagenerator<T>::distribute(
        result,
        normal_values,
        collision_values,
        data_size,
        distinct_values,
        layout_seed,
        non_collision_first,
        evenly_distributed
    );
    
    Datagenerator<T>::safe_values(
        normal_values,
        collision_values,
        distinct_values
    );

    return data_size;
}

template<typename T>
size_t Datagenerator<T>::get_data_strided(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t collision_count,
    size_t layout_seed,
    bool non_collision_first,
    bool evenly_distributed
){
    
    m_working_set_data_matrix->clear_used();
    std::vector<T> normal_values;
    std::vector<T> collision_values;

    std::cout << " s1 " << std::flush;
    
    Datagenerator<T>::get_values_strided(
        normal_values, 
        collision_values, 
        distinct_values,
        collision_count,
        layout_seed
    );

    std::cout << " s2 " << std::flush;
    
    Datagenerator<T>::distribute(
        result,
        normal_values,
        collision_values,
        data_size,
        distinct_values,
        layout_seed,
        non_collision_first,
        evenly_distributed
    );
    
    std::cout << " s3 " << std::flush;

    Datagenerator<T>::safe_values(
        normal_values,
        collision_values,
        distinct_values
    );

    std::cout << " s4 " << std::flush;

    return data_size;
}

template<typename T>
void Datagenerator<T>::get_probe_values_random(std::vector<T> &values, size_t number, size_t seed){
    size_t id = noise(0, seed) % m_bucket_count;
    size_t space = m_bucket_count - number;
    if(space < 1){
        space = 1;
    }

    for(size_t i = 0; i < number; i++){
        size_t skip = noise(i, seed) % space;
        
        size_t soft_skip = 0;
        size_t help = 0;
        T n_val = m_working_set_data_matrix->get_unused_value(id, soft_skip, true, false);
        while(vector_contains(values, n_val)){
            help ++;
            n_val = m_working_set_data_matrix->get_unused_value(id, soft_skip, true, false, help);
        }
        values.push_back(n_val);
        
        if(soft_skip > skip && soft_skip < space){
           skip = soft_skip;
        }
        space -= skip;
        id += skip + 1;
        while(id > m_bucket_count){
            id -= m_bucket_count;
        }
    }
}

template<typename T>
void Datagenerator<T>::get_probe_values_strided(std::vector<T> &values, size_t number, size_t seed){
    if(number == 0){
        return;
    }
    float step_size = m_bucket_count / number;
    size_t base_id = noise(0, seed) % m_bucket_count;
    for(size_t i = 0; i < number; i++){
        size_t id = (size_t)(step_size * i + base_id) % m_bucket_count;   

        size_t soft_skip = 0;
        size_t help = 0;
        T n_val = m_working_set_data_matrix->get_unused_value(id, soft_skip, true, false);
        while(vector_contains(values, n_val)){
            help ++;
            n_val = m_working_set_data_matrix->get_unused_value(id, soft_skip, true, false, help);
        }
        values.push_back(n_val);
    }
}

template<typename T>
size_t Datagenerator<T>::get_probe_strided(
    T*& result,
    size_t data_size,
    float selectivity,
    size_t layout_seed,
    bool evenly_distributed 
){
    std::vector<T> hit_vals;
    std::vector<T> mis_vals;
 
    size_t hit = this->m_values_count * selectivity;
    size_t mis = this->m_values_count * (1-selectivity);
    std::vector<bool> select;
    
    std::cout << " s1 " << std::flush;

    get_collision_bit_map_random(select, this->m_values_count + 1, hit + (hit > 0), layout_seed++);
    
    std::cout << " s2 " << std::flush;
        
    for(size_t i = 1; i < select.size(); i++){
        if(select[i]){
            hit_vals.push_back(this->m_values[i-1]);
        }
    }

    std::cout << " s3 " << std::flush;
    get_probe_values_strided(mis_vals, mis, layout_seed++);

    std::cout << " s4 " << std::flush;
    Datagenerator<T>::distribute(
        result,
        hit_vals,
        mis_vals,
        data_size,
        hit + mis,
        layout_seed,
        false, 
        evenly_distributed
    );
    
    std::cout << " s5 " << std::flush;
    return hit_vals.size() + mis_vals.size();
}

///****************************************/
//* implementations of private functions *//
/****************************************///


/*
    the first entry is always a collision. the rest is random.
    this makes it perfect for strided. if no collision is wanted this still would be okay
*/
template<typename T>
void Datagenerator<T>::get_collision_bit_map_random(
    std::vector<bool> &collide,
    size_t distinct_values,
    size_t collision_count,
    size_t seed
){
    collide.clear();
    collide.resize(distinct_values, false);

    if(collision_count < 1){
        collision_count = 1;
    }
    collision_count --; //collision count tracks how many values collide

    size_t to_generate = collision_count;
    size_t space = (distinct_values - 1) - to_generate;

    // if inverse than is the inclusion in the vector ids no collision otherwise it is a collision id
    bool inverse = to_generate > space;
    if(inverse){
        to_generate = space;
    }

    std::vector<size_t> ids;    // save ids that are collisions (!inversed) or are non collisions (inversed)
    size_t help = 0;

    //we generate random values for ids 
    for(size_t c = 0; c < to_generate; c++){
        size_t nid;
        do{
            nid = (noise(c + help++, seed) % (distinct_values - 1)) + 1;
            bool contains = !vector_contains<size_t>(ids, nid);
        }while(vector_contains<size_t>(ids, nid));
        ids.push_back(nid);
    }

    collide[0] = true; // the first element is always a collision

    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(size_t i = 1; i < distinct_values; i++){
        bool is_in = vector_contains<size_t>(ids, i);
        bool insert = is_in != inverse;
        collide[i] = insert;
    }
}

template<typename T>
void Datagenerator<T>::get_collision_bit_map_bad(
    std::vector<bool> &collide,
    size_t distinct_values,
    size_t collision_count,
    size_t seed,
    size_t set_collisions // means that the first x elements should generate collisions, such that the collisions are of known lenght if necessary
){
    collide.clear();

    if(collision_count > distinct_values){
        collision_count = distinct_values;
    }
    if(collision_count < 1){
        collision_count = 1;
    }
    if(set_collisions > collision_count){
        set_collisions = collision_count;
    }
    if(set_collisions < 1){
        set_collisions = 1;
    }

    size_t space = distinct_values - set_collisions;
    size_t to_generate_col = collision_count - set_collisions;
    size_t to_generate_space = space - collision_count;

    for(size_t i = 0; i < set_collisions; i++){
        collide.push_back(true);
    }

    for(size_t i = 0; i < to_generate_space; i++){
        collide.push_back(false);
    }

    for(size_t i = 0; i < to_generate_col; i++){
        collide.push_back(true);
    }
}


template<typename T>
void Datagenerator<T>::get_values_strided(
    std::vector<T> &non_collision_data,
    std::vector<T> &collision_data,
    size_t distinct_values,
    size_t collision_count,
    size_t seed
){
    size_t * ids = new size_t[distinct_values];
    m_working_set_data_matrix->clear_used();
    collision_data.clear();
    non_collision_data.clear();

    std::cout << " a " << std::flush;
    std::vector<bool> collision_bit_map;
    get_collision_bit_map_random(collision_bit_map, distinct_values, collision_count, seed);
    
    std::cout << " b " << std::flush;

    //generate all ids
    size_t min_collision_pos;
    size_t max_collision_pos;
    // std::chrono::high_resolution_clock::time_point tb = std::chrono::high_resolution_clock::now();

    std::cout << " c " << std::flush;
    get_ids_strided(collision_bit_map, ids, distinct_values, min_collision_pos, seed);
    
    // std::chrono::high_resolution_clock::time_point tm = std::chrono::high_resolution_clock::now();

    std::cout << " d " << std::flush;
    get_values(collision_data, non_collision_data, collision_bit_map, ids, distinct_values, min_collision_pos);
    std::cout << " e " << std::flush;

    // std::chrono::high_resolution_clock::time_point te = std::chrono::high_resolution_clock::now();
    // std::cout << "\n\ta: \t"; p_time(tb, tm, true);
    // std::cout << "\tb:\t"; p_time(tm, te, true);

    delete[]ids;
}

// todo
template<typename T>
void Datagenerator<T>::get_values_bad(
    std::vector<T> &non_collision_data,
    std::vector<T> &collision_data,
    size_t distinct_values,
    size_t collision_count,
    size_t seed
){
    collision_data.clear();
    non_collision_data.clear();


    size_t neighboring_collisions = (collision_count + m_bucket_size - 1) / m_bucket_size;
    if(neighboring_collisions == 0){
        neighboring_collisions = 1;
    }

    std::vector<bool> collision_bit_map;
    get_collision_bit_map_bad(collision_bit_map, distinct_values, collision_count, seed, neighboring_collisions);

    //generate all ids
    size_t * ids = new size_t[distinct_values];
    size_t min_collision_pos;
    size_t max_collision_pos;
    get_ids_packed(collision_bit_map, ids, distinct_values, min_collision_pos, max_collision_pos, seed);
    
    get_values(collision_data, non_collision_data, collision_bit_map, ids,  distinct_values, min_collision_pos);

    delete[]ids;
}

template<typename T>
void Datagenerator<T>::get_values(
    std::vector<T> &collision_data, std::vector<T> &non_collision_data, std::vector<bool> collision_bit_map, 
    size_t * ids, size_t distinct_values, size_t min_collision_pos
){
    bool okay = true;
    bool first = true;
restart:
    collision_data.clear();
    non_collision_data.clear();
    m_working_set_data_matrix->clear_used();

    for(size_t i = 0; i < distinct_values; i++){
        size_t pos = ids[i];
        size_t next_bucket = 0;
        if(collision_bit_map[i]){
            collision_data.push_back(m_working_set_data_matrix->get_unused_value(min_collision_pos, next_bucket));
            min_collision_pos += next_bucket;
            if(min_collision_pos >= m_working_set_data_matrix->get_bucket_count()){
                min_collision_pos = 0;
            }
        }else{
            non_collision_data.push_back(m_working_set_data_matrix->get_unused_value(pos, next_bucket));
        }
    }

    for(size_t i = 0; i < collision_data.size(); i++){
        okay = okay && collision_data[i] != 0;
    }

    for(size_t i = 0; i < non_collision_data.size(); i++){
        okay = okay && non_collision_data[i] != 0;
    }

    if(!first && !okay){
        std::cout << "bad data generation\n";
        exit(-1);
    }
    if(!okay){
        std::cout << "first bad generation attempt" << std::endl;
        first = false;
        goto restart;
    }
    
}

template<typename T>
void Datagenerator<T>::distribute(
    T*& result,
    std::vector<T> raw_non_collision,
    std::vector<T> raw_collision,
    size_t data_size,
    size_t distinct_values,
    size_t seed,
    bool non_collisions_first,
    bool evenly_distributed
){
    size_t dist = data_size + distinct_values - 1;
    if(evenly_distributed){
        dist = (dist / distinct_values) + 1;
    }
    dist -= non_collisions_first;
    size_t help = 1;
    while(help < distinct_values){
        help *= 2;
    }
    help --;
    size_t pos = 0;

    std::vector<T> val_values;
    std::vector<int64_t> val_counts;
    val_values.resize(distinct_values, 0);
    val_counts.resize(distinct_values, dist);
    //add non collision values
    std::cout << "a" << std::flush;
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(size_t i = 0; i < raw_non_collision.size(); i++){
        if(non_collisions_first){
            result[i + pos] = raw_non_collision[i];
        }
        val_values[i + pos] = raw_non_collision[i];
    }
    pos += raw_non_collision.size();
    std::cout << "b" << std::flush;
    
    //add collision values
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(size_t i = 0; i < raw_collision.size(); i++){
        if(non_collisions_first){
            result[i + pos] = raw_collision[i];
        }
        val_values[i + pos] = raw_collision[i];
    }
    pos += raw_collision.size();
    std::cout << "c" << std::flush;

    for(size_t i = 0; i < val_counts.size(); i++){
        if(val_values[i] == 0){
            std::cout << i << ": is zero\n";
            exit(1);
        }
        if(val_counts[i] == 0){
            vector_delete(val_counts, i);
            vector_delete(val_values, i);
        }
    }

    const size_t vc_size = val_counts.size();
    
    for(size_t c_write_pos = pos; c_write_pos < data_size; c_write_pos++){        
        size_t id = ((c_write_pos ^ seed) ^ (val_counts.size() >> 1 )) % val_counts.size(); 

        T val = val_values[id];
        val_counts[id]--;
        result[c_write_pos] = val;
        if(val_counts[id] <= 0){
            vector_delete(val_counts, id);
            vector_delete(val_values, id);
            if(val_counts.size() == 0){
                std::cout << "WE GOT A PROBLEM!" << std::endl;
                return;
            }
        }
    }
    std::cout << "d" << std::flush;

}

// the postition ids are strided. with equal space between them.
template<typename T>
void Datagenerator<T>::get_ids_strided(
    std::vector<bool> collision_bit_map, 
    size_t *& ids, 
    size_t distinct_values, 
    size_t & min_collision_pos,
    size_t seed
){
    size_t start_pos = noise(0, seed) % m_bucket_count;
    double current_pos = start_pos;
    double step_size = (m_bucket_count * 1.0) / distinct_values;

    bool first_collision = false;

    for(size_t i = 0; i < collision_bit_map.size(); i++){
        if(collision_bit_map[i]){
            min_collision_pos = ids[i];
            break;
        }
    }

    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(size_t i = 0; i < collision_bit_map.size(); i++){
        ids[i] = (size_t)(i * step_size) % m_bucket_count;
    }
}

// the postition ids are directly next to each other. TODO
template<typename T>
void Datagenerator<T>::get_ids_packed(
    std::vector<bool> collision_bit_map, 
    size_t *& ids, 
    size_t distinct_values, 
    size_t & min_collision_pos,
    size_t & max_collision_pos,
    size_t seed
){
    size_t start_pos = noise(0, seed) % m_bucket_count;
    size_t current_pos = start_pos;
    size_t step_size = 1;

    bool started = false;
    bool ended = false;
    
    for(size_t i = 0; i < distinct_values; i++){
        ids[i] = (size_t)(current_pos) % m_bucket_count;
        current_pos += step_size;

        if(collision_bit_map[i] && !started){
            started = true;
            min_collision_pos = ids[i];
        }else if(!collision_bit_map[i] && !ended){
            ended = true;
            max_collision_pos = ids[i];
        }
    }
    if(!ended){
        max_collision_pos = ids[distinct_values - 1];
    }
}

template<typename T>
void Datagenerator<T>::safe_values(
    std::vector<T> normal_values,
    std::vector<T> collision_values,
    size_t distinct_values
){
    if(this->m_values != nullptr){
        delete[] this->m_values;
        this->m_values = nullptr;
    }
    this->m_values = new T[distinct_values];
    this->m_values_count = distinct_values;
    size_t id = 0;
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(size_t i = 0; i < normal_values.size(); i++){
        m_values[id + i] = normal_values[i];
    }
    id = normal_values.size();

    #pragma omp parallel for num_threads(THREAD_COUNT)
    for(size_t i = 0; i < collision_values.size(); i++){
        m_values[id + i] = collision_values[i];
    }
}

///*********************************************/
//* template definitions of the Datagenerator *//
/*********************************************///

template class Datagenerator<uint64_t>;
template class Datagenerator<int64_t>;
template class Datagenerator<uint32_t>;
template class Datagenerator<int32_t>;
template class Datagenerator<uint16_t>;
template class Datagenerator<int16_t>;
template class Datagenerator<uint8_t>;
template class Datagenerator<int8_t>;

///***************************************/
//* implementations of helper functions *//
/***************************************///

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
bool vector_delete(std::vector<T>& x, size_t id){
    if(id < x.size()){
        x.erase(x.begin() + id);
        return true;
    }
    return false;
}

template<typename T>
void vector_print(std::vector<T> x){
    for(T val : x){
        std::cout << "\t" << val;
    }
    std::cout << std::endl;
}

template<>
void vector_print(std::vector<bool> x){
    for(bool val : x){
        std::cout << "\t" << val;
    }
    std::cout << std::endl;
}

template<typename T>
void array_print(T* x, size_t len){
    for(size_t i = 0; i < len; i++){
        if((i)%20 == 0){
            std::cout << std::endl;
        }
        std::cout << x[i] << "\t";
    }
    std::cout << std::endl;
}