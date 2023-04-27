#ifndef TUD_HASHING_TESTING_CHAINED_CHAINED2_COUNT
#define TUD_HASHING_TESTING_CHAINED_CHAINED2_COUNT

#include <stdint.h>
#include <stdlib.h> 
#include "../../../logical/group_count.hpp"
#include <unordered_map>
// #include "../../../main/hash_function.hpp"

struct Key{
    uint32_t id;
    size_t HSIZE;

    bool operator==(const Key &other)const{
        return id == other.id;
    }
};


struct KeyHasher
{
    size_t range(uint32_t val, size_t N) const{
        uint64_t a = val;
        uint64_t b = N;

        size_t bit = 32 - __builtin_clz(N) + 4;
        uint64_t c = (a & (0xFFFFFFFF >> (32 - bit))) * b;
        return (c) >> (bit);
    }

    size_t operator()(const Key& k) const
    {
        size_t BIT_NOISE1 = 0x300a8352005996ae;
        size_t BIT_NOISE2 = 0x512eb6f10ed4909d;
        size_t BIT_NOISE3 = 0xae2008421fd52b1f;

        uint64_t mangled = k.id;

        mangled *= BIT_NOISE1;
        mangled += k.HSIZE;
        mangled ^= (mangled << 13);
        mangled += BIT_NOISE2;
        mangled ^= (mangled >> 7);
        mangled *= BIT_NOISE3;
        mangled ^= (mangled << 17);

        return range(mangled, k.HSIZE);
    }
};


template <typename T>
class Chained2 : public Group_count<T>{

    protected:
        std::unordered_map<Key, T, KeyHasher> *map;

    public:
        Chained2(size_t HSIZE, size_t (*hash_function)(T, size_t), bool extend = false, int32_t bonus_scale = 1);
        virtual ~Chained2();
        
        void create_hash_table(T* input, size_t dataSize);
        
        T get(T value);

        void print(bool horizontal);

        std::string identify(){
            return "unordered_map collision";
        }
        
        size_t get_HSIZE();
        
        void clear();

};


#endif //TUD_HASHING_TESTING_CHAINED2_COUNT