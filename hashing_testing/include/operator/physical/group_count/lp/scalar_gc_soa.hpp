#ifndef TUD_HASHING_TESTING_SCALAR_GROUP_COUNT_LP_SOA
#define TUD_HASHING_TESTING_SCALAR_GROUP_COUNT_LP_SOA

#include <stdint.h>
#include <stdlib.h> 
#include "operator/logical/group_count.hpp"

template <typename T>
class Scalar_gc_SoA : public Group_count<T>{
    protected:
        T* m_hash_vec;
        T* m_count_vec;

    public:
        Scalar_gc_SoA(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend = false, int32_t bonus_scale = 1);
        virtual ~Scalar_gc_SoA();
        
        void create_hash_table(T* input, size_t dataSize);
        
        T get(T value);

        T getval(size_t id){
            return m_hash_vec[id];
        }

        void print(bool horizontal);

        std::string identify(){
            return "LP SoA";
        }
        
        size_t get_HSIZE();
        
        void clear();

};


#endif //TUD_HASHING_TESTING_SCALAR_GROUP_COUNT_SOA