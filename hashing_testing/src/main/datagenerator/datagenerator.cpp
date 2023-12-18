#include "main/datagenerator/datagenerator.hpp"
#include <iostream>
#include <limits>

template<typename T>
bool vector_contains(std::vector<T> vec, T value);

template<typename T>
bool vector_delete(std::vector<T>& x, size_t id);

template<typename T>
void vector_print(std::vector<T> x);

template<typename T>
void array_print(T* x, size_t len);


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

    m_bucket_size = m_working_set_data_matrix->get_bucket_size();
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
    std::vector<T> normal_values;
    std::vector<T> collision_values;
    Datagenerator<T>::get_values_strided(
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
    return data_size;
}

///****************************************/
//* implementations of private functions *//
/****************************************///
template<typename T>
void Datagenerator<T>::get_collision_bit_map(
    std::vector<bool> &collide,
    size_t distinct_values,
    size_t collision_count,
    size_t &seed,
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

    size_t to_generate = collision_count - set_collisions;
    size_t space = distinct_values - set_collisions;

    bool inverse = to_generate > space - to_generate;
    if(inverse){
        to_generate = space - to_generate;
    }

    std::vector<size_t> ids;
    size_t help = 0;
    for(size_t c = 0; c < to_generate; c++){
        size_t nid = (noise(c + help++, seed) % (space)) + set_collisions;
        bool contains = !vector_contains<size_t>(ids, nid);
        if(contains){
            ids.push_back(nid);
        }else{
            c--;
        }
    }

    for(size_t i = 0; i < set_collisions; i++){
        collide.push_back(true);
    }

    for(size_t i = set_collisions; i < distinct_values; i++){
        bool is_in = vector_contains<size_t>(ids, i);
        bool insert = is_in != inverse;
        collide.push_back(insert);
    }
}

template<typename T>
void Datagenerator<T>::get_collision_bit_map_bad(
    std::vector<bool> &collide,
    size_t distinct_values,
    size_t collision_count,
    size_t &seed,
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
    collision_data.clear();
    non_collision_data.clear();

    size_t neighboring_collisions = (collision_count + m_bucket_size - 1) / m_bucket_size;
    if(neighboring_collisions == 0){
        neighboring_collisions = 1;
    }

    std::vector<bool> collision_bit_map;
    get_collision_bit_map(collision_bit_map, distinct_values, collision_count, seed, neighboring_collisions);

    //generate all ids
    size_t * ids = (size_t*)malloc(distinct_values * sizeof(size_t));
    size_t min_collision_pos;
    size_t max_collision_pos;

    get_ids_strided(collision_bit_map, ids, distinct_values, min_collision_pos, max_collision_pos, seed);
    

    get_values(collision_data, non_collision_data, collision_bit_map, ids, distinct_values, min_collision_pos);

    free(ids);
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
    size_t * ids = (size_t*)malloc(distinct_values * sizeof(size_t));
    size_t min_collision_pos;
    size_t max_collision_pos;
    get_ids_packed(collision_bit_map, ids, distinct_values, min_collision_pos, max_collision_pos, seed);
    
    get_values(collision_data, non_collision_data, collision_bit_map, ids,  distinct_values, min_collision_pos);

    free(ids);
}

template<typename T>
void Datagenerator<T>::get_values(
    std::vector<T> &collision_data, std::vector<T> &non_collision_data, std::vector<bool> collision_bit_map, 
    size_t * ids, size_t distinct_values, size_t min_collision_pos
){
    for(size_t i = 0; i < distinct_values; i++){
        size_t pos = ids[i];
        if(collision_bit_map[i]){
            collision_data.push_back(m_working_set_data_matrix->get_next_value(min_collision_pos));
        }else{
            non_collision_data.push_back(m_working_set_data_matrix->get_next_value(pos));
        }
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
    std::vector<size_t> val_counts;
    std::vector<T> val_values;
    
    size_t dist = data_size + distinct_values - 1;
    if(evenly_distributed){
        dist = dist / distinct_values;
    }
    size_t pos = 0;
    size_t write_pos = 0;

    //add non collision values
    for(size_t i = 0; i < raw_non_collision.size(); i++){
        if(non_collisions_first){
            result[write_pos++] = raw_non_collision[i];
        }
        val_values.push_back(raw_non_collision[i]);
        val_counts.push_back(dist - non_collisions_first);
        pos++;
    }
    
    //add collision values
    for(size_t i = 0; i < raw_collision.size(); i++){
        val_values.push_back(raw_collision[i]);
        val_counts.push_back(dist);
        pos++; 
    }

    while(write_pos < data_size){
        size_t id = noise(write_pos, seed) % val_counts.size();
        size_t oid;
        size_t count = val_counts[id];
        T val = val_values[id];
        result[write_pos++] = val;
        val_counts[id]--;
        if(count <= 1){
            vector_delete(val_values, id);
            vector_delete(val_counts, id);
        }
    }

}

// the postition ids are strided. with equal space between them.
template<typename T>
void Datagenerator<T>::get_ids_strided(
    std::vector<bool> collision_bit_map, 
    size_t *& ids, 
    size_t distinct_values, 
    size_t & min_collision_pos,
    size_t & max_collision_pos,
    size_t seed
){
    size_t start_pos = noise(0, seed) % m_bucket_count;
    double current_pos = start_pos;
    double step_size = (m_bucket_count * 1.0) / distinct_values;

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

// the postition ids are directly next to each other.
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