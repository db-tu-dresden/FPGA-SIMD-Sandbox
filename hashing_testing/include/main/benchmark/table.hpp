#ifndef TUD_HASHING_TESTING_BENCHMARK_LOAD
#define TUD_HASHING_TESTING_BENCHMARK_LOAD

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdint.h>

enum Comparison{
    equal = 1,
    lessthan = 2,
    lessequal = 3,
    greaterthan = 4,
    greaterequal = 5
};


template<typename T>
class Table{
    private:
        size_t m_col_number;
        size_t m_row_number;
        T ** m_values;
        char* buffer;

    protected:
    public:
        Table(std::string filename, char seperator = '|', bool header = false);
        ~Table();


        T* get_column(size_t id);

        size_t get_column_count(){
            return m_col_number;
        }

        size_t get_row_count(){
            return m_row_number;
        }

        size_t get_distinct_values(size_t column_id);

        // Table filter(size_t col, size_t value, Comparison comp)

};



#endif //TUD_HASHING_TESTING_BENCHMARK_LOAD