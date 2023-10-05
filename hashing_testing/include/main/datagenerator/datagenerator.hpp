#ifndef TUD_HASHING_TESTING_DATAGENERATOR
#define TUD_HASHING_TESTING_DATAGENERATOR

#include "main/datagenerator/datagen_help.hpp"
#include "main/hash_function.hpp"

template<typename T>
class Datagenerator{
    private:
        size_t m_seed;
        size_t (*m_hash_function)(T, size_t);

        size_t m_bucket_count, m_original_bucket_count;
        size_t m_bucket_size, m_original_bucket_size;
        T** m_all_numbers = nullptr;
        T** m_original_numbers = nullptr;

        void get_collision_bit_map(std::vector<bool> &collide, size_t space, size_t wanted, size_t &seed, size_t set_collisions = 1);

        void get_collision_bit_map_bad(std::vector<bool> &collide, size_t space, size_t wanted, size_t &seed, size_t set_collisions = 1);

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

        void get_ids_packed(
            std::vector<bool> collision_bit_map, size_t *& ids, size_t distinct_values, 
            size_t & min_collision_pos, size_t & max_collision_pos, size_t seed
        );
        
        void get_ids_strided(
            std::vector<bool> collision_bit_map, size_t *& ids, size_t distinct_values, 
            size_t & min_collision_pos, size_t & max_collision_pos, size_t seed
        );

    public:
        Datagenerator(
            size_t different_values, size_t (*hash_function)(T, size_t),
            size_t max_collision_size, size_t number_seed
        );
        
        ~Datagenerator(){
            free_all_numbers(m_all_numbers, m_bucket_count);
            free_all_numbers(m_original_numbers, m_original_bucket_count);
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

        bool transform_finalise();

        bool transform_hsize(size_t n_hsize, bool set_collisions = true){
            free_all_numbers(m_all_numbers, m_bucket_count);
            if((m_original_bucket_count * m_original_bucket_size / m_bucket_size) > n_hsize){
                m_bucket_count = n_hsize;

                if(set_collisions){
                    size_t n_collision = (m_original_bucket_count * m_original_bucket_size / m_bucket_count) - 1;
                    return transform_collision(n_collision);
                }
                return true;
            }
            return false;
        }

        bool transform_collision(size_t n_collision){
            free_all_numbers(m_all_numbers, m_bucket_count);
            if((m_original_bucket_count * m_original_bucket_size / m_bucket_count) > n_collision){
                m_bucket_size = n_collision;
                return true;
            }
            return false;
        }
        
        void revert(){
            free_all_numbers(m_all_numbers, m_bucket_count);
            malloc_all_numbers(m_all_numbers, m_original_bucket_count, m_original_bucket_size);
            for(size_t i = 0; i < m_original_bucket_count; i++){
                for(size_t e = 0; e < m_original_bucket_size + 1; e++){
                    m_all_numbers[i][e] = m_original_numbers[i][e];
                }
            }

            m_bucket_count = m_original_bucket_count;
            m_bucket_size = m_original_bucket_size;
        }

        size_t get_bucket_count(){return m_bucket_count;}
        size_t get_original_bucket_count(){return m_original_bucket_count;}
        size_t get_bucket_size(){return m_bucket_size;}
        size_t get_original_bucket_size(){return m_original_bucket_size;}

};


// template<typename T>
// size_t generate_strided_data(
//     T*& result, 
//     size_t data_size,
//     size_t distinct_values,
//     size_t hsize,
//     hash_fptr<T> hash_function,
//     size_t collision_size,
//     size_t seed,
//     bool non_collisions_first = true,
//     bool evenly_distributed = true
// );

// template<typename T>
// void generate_strided_data_raw(
//     std::vector<T> &collision_data,
//     std::vector<T> &non_collision_data,
//     size_t distinct_values,
//     size_t hsize,
//     hash_fptr<T> hash_function,
//     size_t collision_size,
//     size_t &seed
// );


#endif