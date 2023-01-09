#ifndef TUD_HASHING_TESTING_SCALAR_GROUP_COUNT
#define TUD_HASHING_TESTING_SCALAR_GROUP_COUNT

#include <stdint.h>
#include <stdlib.h> 
#include "../../logical/group_count.hpp"

template <typename T>
class Scalar_group_count : public Group_count<T>{
    private:
        T* m_hash_vec;
        T* m_count_vec;

    public:
        Scalar_group_count(size_t HSIZE, T (*hash_function)(T, size_t));
        ~Scalar_group_count();
        
        void create_hash_table(T* input, size_t dataSize);
        
        void print();
        void print2();

};


#endif //TUD_HASHING_TESTING_SCALAR_GROUP_COUNT