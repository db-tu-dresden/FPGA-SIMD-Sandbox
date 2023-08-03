



#include "operator/physical/group_count/lcp/avx512_gc_aosov_v3.hpp"
#include "operator/utility.hpp"

#define EMPTY_SPOT 0

template <typename T>
AVX512_gc_AoSoV_v3<T>::AVX512_gc_AoSoV_v3(size_t HSIZE, size_t (*hash_function)(T, size_t))
    :  Scalar_gc_AoS<T>(HSIZE, hash_function, true)
{
    this->m_elements_per_vector = (512 / 8) / sizeof(T);
    this->m_HSIZE_v = (HSIZE * 2 + this->m_elements_per_vector - 1) / this->m_elements_per_vector;
    
    this->m_hash_table_v = (__m512i *) aligned_alloc(512, this->m_HSIZE_v * sizeof(__m512i));
    
    this->m_masks = (__mmask16 *) aligned_alloc(512, (this->m_elements_per_vector + 1) * sizeof(__mmask16));

    this->m_masks[0] = _cvtu32_mask16(0);
    for(size_t i = 1; i <= this->m_elements_per_vector; i++){
        this->m_masks[i] = _cvtu32_mask16(1 << (i-1));
    }

    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        m_hash_table_v[i] = _mm512_setr_epi32(EMPTY_SPOT, 1, EMPTY_SPOT, 1, EMPTY_SPOT, 1, EMPTY_SPOT, 1, 
                                            EMPTY_SPOT, 1, EMPTY_SPOT, 1, EMPTY_SPOT, 1, EMPTY_SPOT, 1);
    }
}

template <typename T>
AVX512_gc_AoSoV_v3<T>::~AVX512_gc_AoSoV_v3(){
    free(this->m_hash_table_v);
    free(this->m_masks);
}

template <>
void AVX512_gc_AoSoV_v3<uint32_t>::create_hash_table(uint32_t* input, size_t data_size){
    
    __m512i one_512i = _mm512_set1_epi32 (1);    
    size_t index;
    
    for(size_t p = 0; p < data_size; p++){
        uint32_t value = input[p];
        uint32_t hash_key = this->m_hash_function(value, this->m_HSIZE_v);

        __m512i value_vector = _mm512_setr_epi32(value, EMPTY_SPOT, value, EMPTY_SPOT, value, EMPTY_SPOT, value, EMPTY_SPOT, 
                                                value, EMPTY_SPOT, value, EMPTY_SPOT, value, EMPTY_SPOT, value, EMPTY_SPOT);

        while(1) {
            __mmask16 check_match = _mm512_cmpeq_epi32_mask(value_vector, m_hash_table_v[hash_key]);
            __mmask16 check_free = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), m_hash_table_v[hash_key]);

            if (check_match > 0) {
                m_hash_table_v[hash_key] = _mm512_mask_add_epi32(m_hash_table_v[hash_key], (check_match << 1), m_hash_table_v[hash_key], one_512i);
                break;
            } else if(check_free > 0) {    
                uint32_t pos = __builtin_ctz(check_free) + 1; 
                m_hash_table_v[hash_key] = _mm512_mask_set1_epi32(m_hash_table_v[hash_key], m_masks[pos], value);
                break;
            }
            hash_key = (hash_key + 1) % this->m_HSIZE_v;
        }
    }
}

template <typename T>
void AVX512_gc_AoSoV_v3<T>::print(bool horizontal){
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


template <>
uint32_t AVX512_gc_AoSoV_v3<uint32_t>::get(uint32_t input){

    uint32_t *res = (uint32_t*) aligned_alloc(64, 1 * sizeof(__m512i));
    uint32_t res_val = 0;
    __m512i value_vector = _mm512_setr_epi32(input, EMPTY_SPOT, input, EMPTY_SPOT, input, EMPTY_SPOT, input, EMPTY_SPOT, 
                                            input, EMPTY_SPOT, input, EMPTY_SPOT, input, EMPTY_SPOT, input, EMPTY_SPOT);

    size_t hash_key = this->m_hash_function(input, this->m_HSIZE_v);
    const size_t start_hash_key = hash_key;
    
    __mmask16 check_match, check_free;
    do{
        check_match = _mm512_cmpeq_epi32_mask(value_vector, m_hash_table_v[hash_key]);
        check_free = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(), m_hash_table_v[hash_key]);
        if(check_match > 0 || check_free > 0){
            break;
        }
        hash_key = (hash_key + 1) % this->m_HSIZE_v;
    }while(hash_key != start_hash_key);
    
    if(check_match){
        uint32_t *res = (uint32_t*) aligned_alloc(64, 1*sizeof(__m512i));
        _mm512_store_epi32(res, m_hash_table_v[hash_key]);
        res_val = res[__builtin_ctz(check_match) + 1];
    }
    free(res);
    return res_val;
}

template class AVX512_gc_AoSoV_v3<uint32_t>;