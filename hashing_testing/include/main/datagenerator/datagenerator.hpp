#ifndef TUD_HASHING_TESTING_DATAGENERATOR
#define TUD_HASHING_TESTING_DATAGENERATOR


#include <stdlib.h>
#include <vector>
#include <cstdint>

#include <iostream>

#include "main/hash_function.hpp"
#include "data_matrix.hpp"

template<typename T>
class Datagenerator{
    private:
        size_t m_seed;
        size_t (*m_hash_function)(T, size_t);

        size_t m_bucket_count, m_original_bucket_count;
        size_t m_bucket_size, m_original_bucket_size;

        Data_Matrix<T> *m_original_data_matrix = nullptr;
        Data_Matrix<T> *m_data_matrix = nullptr;
        Data_Matrix<T> *m_working_set_data_matrix = nullptr;
        
        //saved for probeing data gen
        T * m_values = nullptr;
        size_t m_values_count = 0;

        std::vector<std::vector<T>> m_blocks;

        void get_collision_bit_map_random(std::vector<bool> &collide, size_t space, size_t wanted, size_t seed);

        void get_collision_bit_map_bad(std::vector<bool> &collide, size_t space, size_t wanted, size_t seed, size_t set_collisions = 1);

        void get_values_blocked(
            std::vector<std::vector<T>> & blocks,
            size_t number_of_blocks,
            size_t distinct_values,
            size_t hsize,
            size_t collision_size,
            size_t layout_seed,
            bool probeing = false
        );

        void get_values_strided(
            std::vector<T> &collision_data, std::vector<T> &non_collision_data, 
            size_t nr_values, size_t collision_count, size_t seed
        );

        void get_values_bad(
            std::vector<T> &collision_data, std::vector<T> &non_collision_data, 
            size_t nr_values, size_t collision_count, size_t seed)
        ;

        void get_values(
            std::vector<T> &collision_data, std::vector<T> &non_collision_data, std::vector<bool> collision_bit_map, 
            size_t * ids, size_t distinct_values, size_t min_collision_pos
        ); 

        void distribute(
            T*& result, std::vector<T> raw_collision, std::vector<T> raw_non_collision, size_t data_size,
            size_t distinct_values, size_t seed, bool non_collisions_first, bool evenly_distributed
        );

        void distribute(
            T*& result, std::vector<std::vector<T>> blocks, size_t data_size, 
            size_t number_of_blocks, size_t seed, bool non_collisions_first, bool evenly_distributed
        );

        void get_ids_packed(
            std::vector<bool> collision_bit_map, size_t *& ids, size_t distinct_values, 
            size_t & min_collision_pos, size_t & max_collision_pos, size_t seed
        );
        
        void get_ids_strided(
            std::vector<bool> collision_bit_map, size_t *& ids, size_t distinct_values, 
            size_t & min_collision_pos, size_t seed
        );

        void safe_values(
            std::vector<T> normal_values,
            std::vector<T> collision_values,
            size_t distinct_values
        );

        void get_probe_values_random(std::vector<T> &values, size_t number, size_t seed);
        void get_probe_values_strided(std::vector<T> &values, size_t number, size_t seed);

    public:
        Datagenerator(
            size_t different_values, size_t (*hash_function)(T, size_t),
            size_t max_collision_size, size_t number_seed
        );
        
        ~Datagenerator(){
            if(m_data_matrix != nullptr){
                delete m_data_matrix;
            }
            if(m_values != nullptr){
                delete[] m_values;
            }
            delete m_original_data_matrix;
        }

        size_t get_data_bad(
            T*& result, 
            size_t data_size,
            size_t distinct_values,
            size_t collision_count,
            size_t layout_seed,
            bool non_collisions_first = true,
            bool evenly_distributed = true
        );

        size_t get_data_strided(
            T*& result, 
            size_t data_size,
            size_t distinct_values,
            size_t collision_count,
            size_t layout_seed,
            bool non_collisions_first = true,
            bool evenly_distributed = true
        );

        size_t get_probe_strided(
            T*& result,
            size_t data_size,
            float selectivity,
            size_t layout_seed,
            bool evenly_distributed = true
        );

        size_t get_data_blocked(    
            T*& result,
            size_t data_size,
            size_t distinct_values,
            size_t collision_count,
            size_t layout_seed,
            bool evenly_distributed = true
        );

        size_t get_probe_blocked(
            T*& result,
            size_t data_size,
            float selectivity,
            size_t layout_seed,
            bool evenly_distributed = true
        );

        bool transform_hsize(size_t n_hsize, bool set_collisions = true){   
            if(m_data_matrix != nullptr){
                delete m_data_matrix;
                m_data_matrix = nullptr;
            }
            m_data_matrix = m_original_data_matrix->transform(n_hsize);
            
            m_working_set_data_matrix = m_data_matrix;
            
            m_bucket_count = m_working_set_data_matrix->get_bucket_count();
            return true;           
        }

        void revert_hsize(){
            if(m_data_matrix != nullptr){
                delete m_data_matrix;
                m_data_matrix = nullptr;
            }
            m_working_set_data_matrix = m_original_data_matrix;
            
            m_bucket_size = m_working_set_data_matrix->get_max_bucket_size();
            m_bucket_count = m_working_set_data_matrix->get_bucket_count();
        }

        size_t get_bucket_count(){return m_bucket_count;}
        size_t get_original_bucket_count(){return m_original_bucket_count;}
        size_t get_bucket_size(){return m_bucket_size;}
        size_t get_original_bucket_size(){return m_original_bucket_size;}

        void print_data(){
            m_original_data_matrix->print();
        }
};



#endif