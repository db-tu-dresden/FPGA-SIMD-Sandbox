#ifndef TUD_HASHING_TESTING_GROUP_COUNT_TSL
#define TUD_HASHING_TESTING_GROUP_COUNT_TSL

#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <numa.h>
#include <omp.h>

template<typename T>
class Group_Count_TSL_SOA{
    private:

    protected:
        size_t (*m_hash_function)(T, size_t);
        size_t m_HSIZE;
        size_t m_mem_numa_node;

        size_t m_thread_count = 1;

        T* m_hash_vec;
        T* m_count_vec;

        Group_Count_TSL_SOA(size_t HSIZE, size_t (*hash_function)(T, size_t), size_t numa_node):m_HSIZE{HSIZE},m_hash_function{hash_function},m_mem_numa_node{numa_node}{}
    public:
        virtual ~Group_Count_TSL_SOA(){}

        virtual void create_hash_table(T* input, size_t data_size) = 0;
        virtual T get(T input) = 0;

        virtual void probe(T*& result, T* input, size_t size) = 0;
        virtual void clear() = 0;

        virtual void move_numa(size_t mem_numa_node) = 0; 

        virtual std::string identify() = 0;

        size_t get_HSIZE(){
            return m_HSIZE;
        }

        void set_thread_count(size_t thread_count){
            if(thread_count > 0){
                this->m_thread_count = thread_count;
            }
        }
};

template class Group_Count_TSL_SOA<uint64_t>;
template class Group_Count_TSL_SOA<uint32_t>;
template class Group_Count_TSL_SOA<uint16_t>;
template class Group_Count_TSL_SOA<uint8_t>;

#endif //TUD_HASHING_TESTING_GROUP_COUNT_TSL