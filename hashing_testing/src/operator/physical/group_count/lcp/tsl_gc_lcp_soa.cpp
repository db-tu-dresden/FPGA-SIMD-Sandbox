#include "operator/physical/group_count/lcp/tsl_gc_lcp_soa.hpp"

#include <numa.h>

#define EMPTY_SPOT 0


//todo
// elements per vector
template <class SimdT, typename T>
TSL_gc_LCP_SoA<SimdT, T>::TSL_gc_LCP_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t), size_t numa_node)
    :  Group_Count_TSL_SOA<T>(HSIZE, hash_function, numa_node)
{
    this->m_HSIZE_v = HSIZE;

    this->m_hash_map_v = (vec_t *) numa_alloc_onnode(this->m_HSIZE_v * sizeof(vec_t), numa_node);
    this->m_count_map_v = (vec_t *) numa_alloc_onnode(this->m_HSIZE_v * sizeof(vec_t), numa_node);

    this->m_masks = (imask_t *) aligned_alloc(ps::vector_alignment(), (this->m_elements_per_vector + 1) * sizeof(imask_t));

    this->m_masks[0] = tsl::to_integral<ps>(tsl::to_mask<ps>(0));
    for(imask_t i = 1; i <= this->m_elements_per_vector; i++){
        this->m_masks[i] = tsl::to_integral<ps>(tsl::to_mask<ps>(1 << (i-1)));
    }

    for(size_t i = 0; i < this->m_HSIZE_v; i++){
        m_hash_map_v[i] = tsl::set1<ps>(EMPTY_SPOT);
        m_count_map_v[i] = tsl::set1<ps>(0);
    }
}

template <class SimdT, typename T>
TSL_gc_LCP_SoA<SimdT, T>::~TSL_gc_LCP_SoA(){
    numa_free(this->m_hash_map_v, this->m_HSIZE_v * sizeof(vec_t));
    numa_free(this->m_count_map_v, this->m_HSIZE_v * sizeof(vec_t));
    free(this->m_masks);
}

template <class SimdT, typename T>
void TSL_gc_LCP_SoA<SimdT, T>::create_hash_table(T* input, size_t data_size){
    
    vec_t one_vec = tsl::set1<ps>(1);    
    vec_t empty_value_vec = tsl::set1<ps>(EMPTY_SPOT);

    for(size_t p = 0; p < data_size; p++){

        uint32_t value = input[p];
        size_t hash_key = this->m_hash_function(value, this->m_HSIZE_v);

        vec_t value_vector = tsl::set1<ps>(value);

        while(1) {
            mask_t check_for_match = tsl::equal<ps>(value_vector, m_hash_map_v[hash_key]);
            imask_t check_for_match_i = tsl::to_integral<ps>(check_for_match);
            
            mask_t check_for_empty_space = tsl::equal<ps>(empty_value_vec, m_hash_map_v[hash_key]);
            imask_t check_for_empty_space_i = tsl::to_integral<ps>(check_for_empty_space);

            if (check_for_match_i > 0) {
                m_count_map_v[hash_key] = tsl::add<ps>(check_for_match, m_count_map_v[hash_key], one_vec);
                break;
            } else if(check_for_empty_space_i > 0) {
                size_t pos = tsl::tzc<ps>(check_for_empty_space_i) + 1;
                m_hash_map_v[hash_key] = tsl::masked_set1<ps>(m_hash_map_v[hash_key], m_masks[pos], value);
                m_count_map_v[hash_key] = tsl::masked_set1<ps>(m_count_map_v[hash_key], m_masks[pos], 1);
                break;
            }

            //branch prediction should be faster than modulo.
            ++hash_key;
            if(hash_key >= this->m_HSIZE_v){
                hash_key = 0;
            }
            // hash_key = (hash_key + 1) % this->m_HSIZE_v;
        }
    }

    m_transfer = false;
}


//todo
template <class SimdT, typename T>
void TSL_gc_LCP_SoA<SimdT, T>::print(bool horizontal){
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

template<class SimdT, typename T>
T TSL_gc_LCP_SoA<SimdT, T>::get(T input){
    vec_t empty_value_vec = tsl::set1<ps>(EMPTY_SPOT);

    size_t hash_key = this->m_hash_function(input, this->m_HSIZE_v);
    vec_t value_vector = tsl::set1<ps>(input);

    const size_t start_hash_key = hash_key;

    do{
        mask_t check_for_match = tsl::equal<ps>(value_vector, m_hash_map_v[hash_key]);
        imask_t check_for_match_i = tsl::to_integral<ps>(check_for_match);
        mask_t check_for_empty_space = tsl::equal<ps>(empty_value_vec, m_hash_map_v[hash_key]);
        imask_t check_for_empty_space_i = tsl::to_integral<ps>(check_for_empty_space);

        if (check_for_match_i > 0) {
            size_t pos = tsl::tzc<ps>(check_for_empty_space_i);
            return 0;
            // return tsl::extract_value<ps>(pos, m_count_map_v[hash_key]);
        } else if(check_for_empty_space_i > 0) {
            break;
        }

        hash_key++;
        if(hash_key >= this->m_HSIZE_v){
            hash_key = 0;
        }
    } while(hash_key != start_hash_key);
    return 0;
}

template<class SimdT, typename T>
void TSL_gc_LCP_SoA<SimdT, T>::clear(){
    for(size_t i = 0; i < this->m_HSIZE_v; i ++){
        m_hash_map_v[i] = tsl::set1<ps>(EMPTY_SPOT);
        m_count_map_v[i] = tsl::set1<ps>(0);
    }
}


template class TSL_gc_LCP_SoA<tsl::avx512, uint64_t>;
template class TSL_gc_LCP_SoA<tsl::avx512, uint32_t>;
template class TSL_gc_LCP_SoA<tsl::avx512, uint16_t>;
template class TSL_gc_LCP_SoA<tsl::avx512, uint8_t>;

template class TSL_gc_LCP_SoA<tsl::avx2, uint64_t>;
template class TSL_gc_LCP_SoA<tsl::avx2, uint32_t>;
template class TSL_gc_LCP_SoA<tsl::avx2, uint16_t>;
template class TSL_gc_LCP_SoA<tsl::avx2, uint8_t>;

template class TSL_gc_LCP_SoA<tsl::sse, uint64_t>;
template class TSL_gc_LCP_SoA<tsl::sse, uint32_t>;
template class TSL_gc_LCP_SoA<tsl::sse, uint16_t>;
template class TSL_gc_LCP_SoA<tsl::sse, uint8_t>;

template class TSL_gc_LCP_SoA<tsl::scalar, uint64_t>;
template class TSL_gc_LCP_SoA<tsl::scalar, uint32_t>;
template class TSL_gc_LCP_SoA<tsl::scalar, uint16_t>;
template class TSL_gc_LCP_SoA<tsl::scalar, uint8_t>;