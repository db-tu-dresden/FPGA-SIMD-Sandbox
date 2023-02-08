#ifndef TUD_HASHING_TESTING_SCALAR_GROUP_COUNT
#define TUD_HASHING_TESTING_SCALAR_GROUP_COUNT

#include <stdint.h>
#include <stdlib.h> 
#include "../../logical/group_count.hpp"

template <typename T>
class Scalar_group_count : public Group_count<T>{
    protected:
        T* m_hash_vec;
        T* m_count_vec;

    public:
        Scalar_group_count(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend = false);
        ~Scalar_group_count();
        
        void create_hash_table(T* input, size_t dataSize);
        
        T get(T value);

        T getval(size_t id){
            return m_hash_vec[id];
        }

        void print(bool horizontal);

        std::string identify();
        
        size_t get_HSIZE();
        
        void clear();

};


#endif //TUD_HASHING_TESTING_SCALAR_GROUP_COUNT