#ifndef TUD_HASHING_TESTING_CHAINED_CHAINED_COUNT
#define TUD_HASHING_TESTING_CHAINED_CHAINED_COUNT

#include <stdint.h>
#include <stdlib.h> 
#include "../../../logical/group_count.hpp"
#include <unordered_map>
// #include "../../../main/hash_function.hpp"
template <typename T>
class Chained : public Group_count<T>{

    protected:
        std::unordered_map<T, T> *map;


    public:

        Chained(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend = false, int32_t bonus_scale = 1);
        virtual ~Chained();
        
        void create_hash_table(T* input, size_t dataSize);
        
        T get(T value);

        void print(bool horizontal);

        std::string identify(){
            return "unordered_map";
        }
        
        size_t get_HSIZE();
        
        void clear();

};


#endif //TUD_HASHING_TESTING_CHAINED_COUNT