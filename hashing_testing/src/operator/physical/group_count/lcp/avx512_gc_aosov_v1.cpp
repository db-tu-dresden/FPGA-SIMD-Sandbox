



#include "operator/physical/group_count/lcp/avx512_gc_aosov_v1.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_AoSoV_v1<T>::AVX512_gc_AoSoV_v1(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  Scalar_gc_AoS<T>(HSIZE, hash_function, true)
{
    this->m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE_v = (HSIZE + this->m_elements_per_vector - 1) / this->m_elements_per_vector;
    
    //we allocate twice the space needed because of the AoS approach
    this->m_hash_table_v = (__m512i *) aligned_alloc(512, this->m_HSIZE_v * 2 * sizeof(__m512i));
    
    this->m_masks = (__mmask16 *) aligned_alloc(512, (this->m_elements_per_vector + 1) * sizeof(__mmask16));

    this->m_masks[0] = _cvtu32_mask16(0);
    for(size_t i = 1; i <= this->m_elements_per_vector; i++){
        this->m_masks[i] = _cvtu32_mask16(1 << (i-1));
    }

    for(size_t i = 0; i < this->m_HSIZE_v * 2; i++){
        m_hash_table_v[i] = _mm512_set1_epi32(EMPTY_SPOT);
    }
}

template <typename T>
AVX512_gc_AoSoV_v1<T>::~AVX512_gc_AoSoV_v1(){
    free(this->m_hash_table_v);
    free(this->m_masks);
}

template <>
void AVX512_gc_AoSoV_v1<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    
    __m512i one_512i = _mm512_set1_epi32 (1);    
    size_t index;
    for(size_t p = 0; p < data_size; p++){

        uint32_t value = input[p];
        uint32_t hash_key = this->m_hash_function(value, this->m_HSIZE_v);

        __m512i value_vector = _mm512_set1_epi32(value);

        while(1) {
            index = hash_key << 1;
            
            __mmask16 check_match = _mm512_cmpeq_epi32_mask(value_vector, m_hash_table_v[index]);
            __mmask16 check_free = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), m_hash_table_v[index]);

            if (check_match > 0) {
                m_hash_table_v[index + 1] = _mm512_mask_add_epi32(m_hash_table_v[index + 1], check_match, m_hash_table_v[index + 1], one_512i);
                break;
            } else if(check_free > 0) {    
                uint32_t pos = __builtin_ctz(check_free) + 1; 
                m_hash_table_v[index] = _mm512_mask_set1_epi32(m_hash_table_v[index], m_masks[pos], value);
                m_hash_table_v[index + 1] = _mm512_mask_set1_epi32(m_hash_table_v[index + 1], m_masks[pos], 1);
                break;
            }
            
            hash_key = (hash_key + 1) % this->m_HSIZE_v;
        }
    }

    m_transfer = false;
}

template <typename T>
void AVX512_gc_AoSoV_v1<T>::print(bool horizontal){
    // size_t count = 0;
    // size_t HSIZE = this->m_HSIZE;

    // size_t trash = get(EMPTY_SPOT);

    // if(horizontal){
        
    //     for(size_t i = 0; i < HSIZE; i++){
    //             std::cout << "\t" << i;
    //     }
    //     std::cout << std::endl;

    //     for(size_t i = 0; i < HSIZE; i++){
    //             std::cout << "\t" << this->m_hash_vec[i];
    //     }
    //     std::cout << std::endl;

    //     for(size_t i = 0; i < HSIZE; i++){
    //         std::cout << "\t" << this->m_count_vec[i];
    //         count += this->m_count_vec[i];
    //     }
    //     std::cout << std::endl << "Total Count:\t" << count << std::endl;
    // }
    // else{
    //     for(size_t i = 0; i < HSIZE; i++){
    //         std::cout << i << "\t" << this->m_hash_vec[i] << "\t" << this->m_count_vec[i] << std::endl;
    //         count += this->m_count_vec[i];
    //     }
    //     std::cout << "Total Count:\t" << count << std::endl;
    // }
}

template <typename T>
T AVX512_gc_AoSoV_v1<T>::get(T input){
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
uint32_t AVX512_gc_AoSoV_v1<uint32_t>::get(uint32_t input){

    uint32_t *res = (uint32_t*) aligned_alloc(64, 1 * sizeof(__m512i));
    uint32_t res_val = 0;
    __m512i value_vector = _mm512_set1_epi32(input);
    
    size_t hash_key = this->m_hash_function(input, this->m_HSIZE_v);
    const size_t start_hash_key = hash_key;
    
    size_t index = 0;    
    __mmask16 check_match, check_free;
    do{
        index = hash_key << 1;
        check_match = _mm512_cmpeq_epi32_mask(value_vector, m_hash_table_v[index]);
        check_free = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), m_hash_table_v[index]);
        if(check_match > 0 || check_free > 0){
            break;
        }
        hash_key = (hash_key + 1) % this->m_HSIZE_v;
    }while(hash_key != start_hash_key);
    
    if(check_match){
        uint32_t *res = (uint32_t*) aligned_alloc(64, 1*sizeof(__m512i));
        _mm512_store_epi32(res, m_hash_table_v[index+1]);
        res_val = res[__builtin_ctz(check_match)];
    }
    free(res);
    return res_val;
}

template class AVX512_gc_AoSoV_v1<uint32_t>;