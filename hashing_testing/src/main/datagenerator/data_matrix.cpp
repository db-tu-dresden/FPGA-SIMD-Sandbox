#include "main/datagenerator/data_matrix.hpp"
#include <iostream>

template<typename T>
Data_Matrix<T>::Data_Matrix(size_t bucket_count, size_t bucket_size, hash_fptr<T> function, size_t seed):m_bucket_count{bucket_count}, m_function{function}, m_seed{seed}{
    m_bucket_size = bucket_size + m_reserved_bucket_size;

    m_all_numbers = (T*) malloc(m_bucket_count * m_bucket_size * sizeof(T));
    m_values_per_bucket = new size_t[m_bucket_count];
    m_used_cursor = new size_t[m_bucket_count];

    generate_numbers();
}

template<typename T>
Data_Matrix<T>::Data_Matrix(T* all_numbers, size_t *sizes, size_t old_bucket_count, size_t old_bucket_size, size_t bucket_count, size_t bucket_size, hash_fptr<T> function, size_t seed)
    :m_bucket_count{bucket_count}, m_function{function}, m_seed{seed}{

    m_bucket_size = bucket_size + m_reserved_bucket_size;
    
    m_all_numbers = (T*) malloc(m_bucket_count * m_bucket_size * sizeof(T));
    m_values_per_bucket = new size_t[m_bucket_count];
    
    m_used_cursor = new size_t[m_bucket_count];
    
    for(size_t b = 0; b < old_bucket_count; b++){
        for(size_t i = 0; i < sizes[b]; i++){
            insert_number(all_numbers[b * old_bucket_size + i], true);
        }
    }
    
}

template<typename T>
void Data_Matrix<T>::generate_numbers(){ 
    for(size_t i = 0; i < m_bucket_count; i++){
        m_values_per_bucket[i] = 0;
    }

    T h = ~0;
    size_t max_different_values = h;
    size_t wanted_different_values = m_bucket_count * m_bucket_size;
    if(wanted_different_values > max_different_values){
        std::cout << "To many different values need to be generated.\t" << wanted_different_values << " wanted\t" << max_different_values << " possible" << std::endl;
        wanted_different_values = max_different_values;
        m_bucket_size = wanted_different_values / m_bucket_count;
        wanted_different_values = m_bucket_count * m_bucket_size;
    }

    size_t non_values = max_different_values - wanted_different_values;
    non_values *= 0.7;
    
    size_t chunk_count = 12;
    size_t chunk_size = (wanted_different_values / (chunk_count-2)) + 1;
    if(chunk_size < 2 * m_bucket_count){
        chunk_size = 2 * m_bucket_count;
        chunk_count = wanted_different_values/ chunk_size;
    }

    T nr = noise(chunk_size ^ max_different_values, m_seed);

    size_t generated_numbers = 0;
    size_t unsucessful_numbers = 0;
    std::cout << "fast creation: " << std::flush;
    for(size_t i = 0; i < chunk_count; i++){
        
        for(size_t e = 0; e < chunk_size; e++){
            nr += 1;
            nr += nr == 0;
            bool successful_insert = insert_number(nr, true);
            generated_numbers += successful_insert;
            unsucessful_numbers += !successful_insert;
        }
        size_t mod = (non_values - unsucessful_numbers);
        if(mod != 0 && mod <= non_values){
            size_t skip = noise(i, m_seed) % mod ;
            non_values -= skip;
            nr += skip;
        }
        std::cout << "." << std::flush;
    }
    // std::cout << "generated_numbers " << generated_numbers << "\tunsucessful: " << unsucessful_numbers << "\twanted: " << wanted_different_values << std::endl;
    if(generated_numbers < wanted_different_values){
        std::cout << std::endl << "slow creation" << std::endl;
        unsucessful_numbers = 0;
        for(size_t i = 0; i < max_different_values && (generated_numbers + unsucessful_numbers) < max_different_values && generated_numbers < wanted_different_values; i++){
            nr += 1;
            nr += nr == 0;
            bool successful_insert = insert_number(nr);
            generated_numbers += successful_insert;
            unsucessful_numbers += !successful_insert;
        }
    }
    std::cout << "\tgeneration done\n";
}

template<typename T>
bool Data_Matrix<T>::insert_number(T number, bool force){
    size_t bucket = m_function(number, this->m_bucket_count);
    size_t base = bucket * m_bucket_size;

    if(m_values_per_bucket[bucket] < m_bucket_size){
        if(force){
            m_all_numbers[base + m_values_per_bucket[bucket]] = number;
            m_values_per_bucket[bucket]++;
            return true;
        }
        bool insert = true;
        for(size_t i = 0; i < m_values_per_bucket[bucket] && insert; i++){
            insert = number != m_all_numbers[base + i];
        }
        m_all_numbers[base + m_values_per_bucket[bucket]] = number;
        m_values_per_bucket[bucket] += insert;
        return insert;
    }
    return false;
}

template<typename T>
void Data_Matrix<T>::clear_used(){
    for(size_t i = 0; i < m_bucket_count; i++){
        m_used_cursor[i] = 0;
    }
}

template<typename T>
T Data_Matrix<T>::get_value(size_t bucket, size_t k){
    size_t b_size = m_values_per_bucket[bucket];
    while(b_size > k){
        k -= b_size;
        bucket++;
        if(bucket >= m_bucket_count){
            bucket = 0;
        }
        b_size = m_values_per_bucket[bucket];
    }
    return m_all_numbers[bucket * m_bucket_size + k];
}


template<typename T>
T Data_Matrix<T>::get_unused_value(size_t bucket, size_t &next_bucket, bool probing, bool mark_used, size_t skip_x){
    size_t check_engine = 0;
    skip_x = (!mark_used) * skip_x;

    for(uint64_t b = bucket, t = 0; t < m_bucket_count * m_bucket_size; t++){
        if(b >= m_bucket_count){
            b = 0;
            check_engine = 1;
        }else if(check_engine == 1 && b == bucket){
            break;
        }
        size_t offset_i = m_used_cursor[b];
        size_t bucket_size = get_bucket_size(b, probing);
        if(offset_i + skip_x < bucket_size){
            m_used_cursor[b] += mark_used;
            return m_all_numbers[b* m_bucket_size + offset_i];
        }else{
            if(offset_i < bucket_size){
                skip_x -= (bucket_size - offset_i);
            }
            t--;
            b++;
            next_bucket++;
        }
    }
    // std::cout << "DATA GEN WARNING: 0 returned as value\n";
    return 0;
}

template<typename T>
Data_Matrix<T>* Data_Matrix<T>::transform(size_t bucket_count){
    size_t max_fill = 0;
    for(size_t i = 0; i < m_bucket_count; i++){
        if(max_fill < m_values_per_bucket[i]){
            max_fill = m_values_per_bucket[i];
        }
    }
    size_t n_bucket_size = m_bucket_count * m_bucket_size / bucket_count;
    if(n_bucket_size < max_fill){
        n_bucket_size *= 1.5;
        if(n_bucket_size > max_fill){
            n_bucket_size = max_fill;
        }
    }
    n_bucket_size -= m_reserved_bucket_size;
    Data_Matrix<T>* res = new Data_Matrix(m_all_numbers, m_values_per_bucket, m_bucket_count, m_bucket_size, bucket_count, n_bucket_size, m_function, m_seed);
    return res;
}

template<typename T>
void Data_Matrix<T>::print(){
    for(size_t b = 0; b < m_bucket_count; b++){
        std::cout << " " << m_values_per_bucket[b];
        // std::cout << b << ":" << m_values_per_bucket[b];
        // for(size_t i = 0; i < m_values_per_bucket[b]; i++){
        //     std::cout << "\t" << m_all_numbers[b * m_bucket_size + i]; 
        // }
        // std::cout << std::endl;
    }
    std::cout << std::endl;
}

template class Data_Matrix<uint64_t>;
template class Data_Matrix<uint32_t>;
template class Data_Matrix<uint16_t>;
template class Data_Matrix<uint8_t>;
template class Data_Matrix<int64_t>;
template class Data_Matrix<int32_t>;
template class Data_Matrix<int16_t>;
template class Data_Matrix<int8_t>;
