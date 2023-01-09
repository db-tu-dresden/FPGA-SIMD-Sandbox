#ifndef TUD_HASHING_TESTING_GROUP_COUNT
#define TUD_HASHING_TESTING_GROUP_COUNT

#include <stdint.h>
#include <stdlib.h> 

template <typename T>
class Group_count{
    private:

    protected:
        T (* m_hash_function) (T, size_t);
        size_t m_HSIZE;
        Group_count(size_t HSIZE, T (*hash_function)(T, size_t)):m_HSIZE{HSIZE},m_hash_function{hash_function}{}

    public:
        virtual void create_hash_table(T* input, size_t dataSize) = 0;
        
        virtual void print() = 0;

};

template class Group_count<uint32_t>;
template class Group_count<uint64_t>;

#endif //TUD_HASHING_TESTING_GROUP_COUNT