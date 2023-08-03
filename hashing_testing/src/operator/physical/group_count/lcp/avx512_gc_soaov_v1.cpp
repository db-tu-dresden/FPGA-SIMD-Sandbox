



#include "operator/physical/group_count/lcp/avx512_gc_soaov_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_SoAoV_v1<T>::AVX512_gc_SoAoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  Scalar_gc_SoA<T>(HSIZE, hash_function, true)
{
    this->m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE_v = (HSIZE + this->m_elements_per_vector - 1) / this->m_elements_per_vector;
    
    this->m_hash_map_v = (__m512i *) aligned_alloc(512, this->m_HSIZE_v * sizeof(__m512i));
    this->m_count_map_v = (__m512i *) aligned_alloc(512, this->m_HSIZE_v * sizeof(__m512i));

    this->m_masks = (__mmask16 *) aligned_alloc(512, (this->m_elements_per_vector + 1) * sizeof(__mmask16));

    this->m_masks[0] = _cvtu32_mask16(0);
    for(size_t i = 1; i <= this->m_elements_per_vector; i++){
        this->m_masks[i] = _cvtu32_mask16(1 << (i-1));
    }

    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        m_hash_map_v[i] = _mm512_set1_epi32(EMPTY_SPOT);
        m_count_map_v[i] = _mm512_set1_epi32(EMPTY_SPOT);
    }
}

template <typename T>
AVX512_gc_SoAoV_v1<T>::~AVX512_gc_SoAoV_v1(){
    free(this->m_hash_map_v);
    free(this->m_count_map_v);
    free(this->m_masks);
}

template <>
void AVX512_gc_SoAoV_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    
    __m512i one_512i = _mm512_set1_epi32 (1);    
    
    for(size_t p = 0; p < data_size; p++){

        uint32_t value = input[p];
        uint32_t hash_key = this->m_hash_function(value, this->m_HSIZE_v);

        __m512i value_vector = _mm512_set1_epi32(value);

        while(1) {
            __mmask16 check_match = _mm512_cmpeq_epi32_mask(value_vector, m_hash_map_v[hash_key]);
            __mmask16 check_free = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), m_hash_map_v[hash_key]);

            if (check_match > 0) {
                m_count_map_v[hash_key] = _mm512_mask_add_epi32(m_count_map_v[hash_key], check_match, m_count_map_v[hash_key], one_512i);
                break;
            } else if(check_free > 0) {    
                uint32_t pos = __builtin_ctz(check_free) + 1; 
                m_hash_map_v[hash_key] = _mm512_mask_set1_epi32(m_hash_map_v[hash_key], m_masks[pos], value);
                m_count_map_v[hash_key] = _mm512_mask_set1_epi32(m_count_map_v[hash_key], m_masks[pos], 1);
                break;
            }
            
            hash_key = (hash_key + 1) % this->m_HSIZE_v;
        }
    }

    m_transfer = false;
}

template <typename T>
void AVX512_gc_SoAoV_v1<T>::print(bool horizontal){
    size_t count = 0;
    size_t HSIZE = this->m_HSIZE;

    size_t trash = get(EMPTY_SPOT);

    if(horizontal){
        
        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << i;
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
                std::cout << "\t" << this->m_hash_vec[i];
        }
        std::cout << std::endl;

        for(size_t i = 0; i < HSIZE; i++){
            std::cout << "\t" << this->m_count_vec[i];
            count += this->m_count_vec[i];
        }
        std::cout << std::endl << "Total Count:\t" << count << std::endl;
    }
    else{
        for(size_t i = 0; i < HSIZE; i++){
            std::cout << i << "\t" << this->m_hash_vec[i] << "\t" << this->m_count_vec[i] << std::endl;
            count += this->m_count_vec[i];
        }
        std::cout << "Total Count:\t" << count << std::endl;
    }
}

template <typename T>
T AVX512_gc_SoAoV_v1<T>::get(T input){
    size_t rounds = 0;
    size_t HSIZE = this->m_HSIZE;
    
    size_t hash_key = this->m_hash_function(input, HSIZE);

    while(rounds <= 1){
        T value = this->m_hash_vec[hash_key];
        
        if(value == input){
            return this->m_count_vec[hash_key];
        }else if(value == EMPTY_SPOT){
            return 0;
        }else{
            hash_key = (hash_key + 1) % HSIZE;
            rounds += (hash_key == 0);
        }
    }
    return 0;
}

template <>
uint32_t AVX512_gc_SoAoV_v1<uint32_t>::get(uint32_t input){
    size_t HSIZE = this->m_HSIZE_v * this->m_elements_per_vector;

    size_t hash_key = this->m_hash_function(input, this->m_HSIZE_v) * this->m_elements_per_vector;
    const size_t start_hash_key = hash_key;

    if(!m_transfer){
        for(size_t i = 0; i < this->m_HSIZE_v; i++){
            size_t h = i * this->m_elements_per_vector;
            _mm512_store_epi32(&this->m_hash_vec[h], m_hash_map_v[i]);
            _mm512_store_epi32(&this->m_count_vec[h], m_count_map_v[i]);
        }
        m_transfer = true;
    }

    do{
        uint32_t value = this->m_hash_vec[hash_key];
        
        if(value == input){
            return this->m_count_vec[hash_key];
        }else if(value == EMPTY_SPOT){
            return 0;
        }

        hash_key = (hash_key + 1) % HSIZE;
    }while(hash_key != start_hash_key);
    
    return 0;
}

template class AVX512_gc_SoAoV_v1<uint32_t>;