#ifndef TUD_HASHING_TESTING_DATA_MATRIX
#define TUD_HASHING_TESTING_DATA_MATRIX

#include <stdlib.h>
#include "main/hash_function.hpp"

//todo integration
template<typename T>
class Data_Matrix{
    private:
        size_t m_seed;
        hash_fptr<T> m_function;
        size_t m_bucket_count;
        size_t m_bucket_size;
        size_t m_reserved_bucket_size = 1; // only for probing data.
        
        T* m_all_numbers;
        size_t * m_values_per_bucket; //count of values in a bucket
        size_t * m_used_cursor; //skip list for used values

        void generate_numbers();
        bool insert_number(T number, bool force = false);
        
    public:
        Data_Matrix(size_t bucket_count, size_t bucket_size, hash_fptr<T> function, size_t seed);
        Data_Matrix(T* all_numbers, size_t* sizes, size_t old_bucket_count, size_t old_bucket_size, size_t bucket_count, size_t bucket_size, hash_fptr<T> function, size_t seed);

        ~Data_Matrix(){
            free(m_all_numbers);
            delete[] m_values_per_bucket;
            delete[] m_used_cursor;
        }

        T get_value(size_t bucket, size_t value);
        T get_unused_value(size_t bucket, size_t &next_bucket, bool probing = false, bool mark_used = true, size_t skip_x = 0); // next free of this bucket.
        
        T* get_start(size_t bucket);
        T* get_end(size_t bucket);

        Data_Matrix *transform(size_t bucket_count);

        void clear_used();

        size_t get_bucket_count(){
            return m_bucket_count;
        }
        
        size_t get_max_bucket_size(){
            return m_bucket_size - m_reserved_bucket_size;
        }

        size_t get_bucket_size(size_t bucket, bool probing = false){
            if(bucket < m_bucket_count){
                return m_values_per_bucket[bucket] - (!probing) * m_reserved_bucket_size;
            }
            return 0;
        }

        void print();
};

#endif //TUD_HASHING_TESTING_DATA_MATRIX
