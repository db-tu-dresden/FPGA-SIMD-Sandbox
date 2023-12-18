#ifndef TUD_HASHING_TESTING_GROUP_COUNT_TSL
#define TUD_HASHING_TESTING_GROUP_COUNT_TSL

#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <numa.h>

// #include <tslintrin.hpp>


template<typename T>
class Group_Count_TSL_SOA{
    private:

    protected:
        size_t (*m_hash_function)(T, size_t);
        size_t m_HSIZE;
        size_t m_mem_numa_node;

        T* m_hash_vec;
        T* m_count_vec;

        Group_Count_TSL_SOA(size_t HSIZE, size_t (*hash_function)(T, size_t), size_t numa_node):m_HSIZE{HSIZE},m_hash_function{hash_function},m_mem_numa_node{numa_node}{}
    public:
        virtual ~Group_Count_TSL_SOA(){}

        virtual void create_hash_table(T* input, size_t data_size) = 0;
        virtual T get(T input) = 0;
        // virtual T* probe(T* input, size_t input_size) = 0; // TODO!
        virtual void clear() = 0;

        virtual std::string identify() = 0;
        size_t get_HSIZE(){
            return m_HSIZE;
        }
};

template class Group_Count_TSL_SOA<uint64_t>;
template class Group_Count_TSL_SOA<uint32_t>;
template class Group_Count_TSL_SOA<uint16_t>;
template class Group_Count_TSL_SOA<uint8_t>;

#endif //TUD_HASHING_TESTING_GROUP_COUNT_TSL