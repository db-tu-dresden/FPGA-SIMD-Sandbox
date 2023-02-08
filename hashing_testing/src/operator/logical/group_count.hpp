#ifndef TUD_HASHING_TESTING_GROUP_COUNT
#define TUD_HASHING_TESTING_GROUP_COUNT

#include <stdint.h>
#include <stdlib.h> 

template <typename T>
class Group_count{
    private:

    protected:
        size_t (* m_hash_function) (T, size_t);
        size_t m_HSIZE;
        Group_count(size_t HSIZE, size_t (*hash_function)(T, size_t)):m_HSIZE{HSIZE},m_hash_function{hash_function}{}


void printBits(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u ", byte);
        }
    }
    puts("");
}

    public:
        virtual void create_hash_table(T* input, size_t dataSize) = 0;
        
        virtual T get(T input) = 0;

        virtual void print(bool horizontal=true) = 0;

        virtual std::string identify() = 0;

        virtual void clear() = 0;

        virtual size_t get_HSIZE() = 0;

};

template class Group_count<uint32_t>;
template class Group_count<uint64_t>;

#endif //TUD_HASHING_TESTING_GROUP_COUNT