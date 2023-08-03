#ifndef TUD_HASHING_TESTING_SCALAR_GROUP_COUNT_LP_AOS_V2
#define TUD_HASHING_TESTING_SCALAR_GROUP_COUNT_LP_AOS_V2

#include <stdint.h>
#include <stdlib.h> 
#include "operator/logical/group_count.hpp"


template<typename T>
struct Entry{
    T key;
    T value;
};

template <typename T>
class Scalar_gc_AoS_V2 : public Group_count<T>{
    protected:
        Entry<T>* m_hash_table;

    public:
        Scalar_gc_AoS_V2(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend = false, int32_t bonus_scale = 1);
        virtual ~Scalar_gc_AoS_V2();
        
        void create_hash_table(T* input, size_t dataSize);
        
        T get(T value);

        void print(bool horizontal);

        std::string identify(){
            return "LP AoS V2";
        }
        
        size_t get_HSIZE();
        
        void clear();

};


#endif //TUD_HASHING_TESTING_SCALAR_GROUP_COUNT_LP_AOS_V2