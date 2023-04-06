
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <string>
#include <cstring>

#include <regex>

#include "table.hpp"

size_t convert_number(std::string str){
    size_t pos{};
    try
    {
        size_t value = std::stoull(str, &pos);
        return value;
    }
    catch(std::invalid_argument const& ex)
    {
        std::cout << "INVALID ARGUMENT DURING CONVERSION OF: " << str << std::endl;
        std::cout << "std::invalid_argument::what(): " << ex.what() << std::endl;
    }
    catch(std::out_of_range const& ex)
    {
        std::cout << "ARGUMENT OUT OF RANGE: " << str << std::endl;
        std::cout << "std::out_of_range::what(): " << ex.what() << std::endl;
    }
    return 0;
}

size_t convert_date(std::string str){
    size_t pos{};
    size_t nr = 0;

    std::istringstream f(str);
    std::string s; 

    for(size_t i = 0; i < 3; i++){
        std::getline(f, s, '-');
        size_t help = 0;
        try
        {
            help += std::stoull(s, &pos);
        }
        catch(std::invalid_argument const& ex)
        {
            std::cout << "INVALID ARGUMENT DURING CONVERSION OF: " << s << std::endl;
            std::cout << "std::invalid_argument::what(): " << ex.what() << std::endl;
        }
        catch(std::out_of_range const& ex)
        {
            std::cout << "ARGUMENT OUT OF RANGE: " << s << std::endl;
            std::cout << "std::out_of_range::what(): " << ex.what() << std::endl;
        }
        for(size_t i = 0; i < pos; i++){
            nr *= 10;
        }
        nr += help;
    }
    return nr;
}

template<size_t bits>
size_t convert_string_rotate_left(const size_t v){
    const size_t left = v << bits;
    const size_t right = v >> (64-bits);
    return left | right;
}

void convert_string_compress(size_t& v0, size_t& v1, size_t& v2, size_t& v3, const size_t rounds){
    for(size_t i = 0; i < rounds; i++){
        v0 += v1;
        v2 += v3;
        v1 = convert_string_rotate_left<13>(v1);
        v3 = convert_string_rotate_left<16>(v3);
        v1 ^= v0;
        v3 ^= v2;

        v0 = convert_string_rotate_left<32>(v0);

        v2 += v1;
        v0 += v3;
        v1 = convert_string_rotate_left<17>(v1);
        v3 = convert_string_rotate_left<21>(v3);
        v1 ^= v2;
        v3 ^= v0;

        v2 = convert_string_rotate_left<32>(v2);
    }
}

size_t convert_string(char* buffer, std::string str){
    size_t len = str.length();

    strcpy(buffer, str.c_str());

    size_t _k_update_rounds = 1;
    size_t _k_finalize_rounds = 1;

    size_t val = 0;
    
    size_t v0 = 0x736f6d6570736575ull;
    size_t v1 = 0x646f72616e646f6dull;
    size_t v2 = 0x6c7967656e657261ull;
    size_t v3 = 0x7465646279746573ull;

    for(size_t i = 0; i < len; i++){
        v3 ^= buffer[i];
        convert_string_compress(v0, v1, v2, v3, _k_update_rounds);
        v0 ^= buffer[i];
    }

    //FINALIZE
    v2 ^= 0xFF;
    convert_string_compress(v0, v1, v2, v3, _k_finalize_rounds);
    val = (v0 ^ v1) ^ (v2 ^ v3);

    return val;
}




template <typename T>
Table<T>::Table(std::string filename, char seperator, bool header) {
    std::cout << "START LOADING: " << filename << std::endl;
    std::vector<std::string> lines;
    std::string col_names;

    std::string line;
    std::ifstream file(filename);
    
    this->buffer = (char*)malloc(200 * sizeof(char));

    if(file.is_open()){
        while(std::getline(file, line)){
            lines.push_back(line); 
        }    

        file.close();
    } else {
        throw std::runtime_error("Unable to open File!");
    }


    std::cout << "START PROCESSING: " << filename << std::endl;
    m_row_number = lines.size();
    m_row_number -= header;
    // m_row_number /= 64;
    // m_row_number *= 64;

    m_col_number = 0;


    std::istringstream f(lines[0]);
    std::vector<std::string> strings;
    std::string s;

    std::regex number("^((\\d+\\.?\\d*)|(\\d*\\.\\d+))$"); //number
    std::regex date("^(\\d{4}-\\d{2}-\\d{2})$"); //matches a date with the format of 2023-03-29
    std::regex all{"^(.+)$"};    // matches everything except empty
    
    while(std::getline(f, s, seperator)){
        strings.push_back(s);
        if(regex_match(s, all)){
            m_col_number++;
        }
    }

    m_values = (T**) aligned_alloc(64, m_col_number * sizeof(uint32_t*));
    for(size_t i = 0; i < m_col_number; i++){
        m_values[i] = (T*) aligned_alloc(64, m_row_number * sizeof(T));
    }

    for(size_t i = 0; i < m_row_number; i++){
        std::istringstream line_split(lines[i]);
        std::string str;
        size_t c = 0;
        while(getline(line_split, str, seperator) && c < m_col_number){
            
            T parsed = 0;
            if(regex_match(str, number)){
                parsed = convert_number(str);
            }else if(regex_match(str, date)){
                parsed = convert_date(str);
            }else if(regex_match(str, all)){
                parsed = convert_string(buffer, str); // using sip from google
            }

            m_values[c][i] = parsed;
            c++;
        }
    }

    free(this->buffer);
    
    std::cout << "LOADING DONE" << std::endl;
}


template <typename T>
Table<T>::~Table(){
    for(size_t i = 0; i < m_col_number; i++){
        free(m_values[i]);
    }
    free(m_values);
}





template <typename T>
T* Table<T>::get_column(size_t id) {
    if(id < m_col_number);
        return m_values[id];
    return nullptr;
}



template <typename T>
size_t Table<T>::get_distinct_values(size_t column_id){
    if(column_id >= m_col_number){
        return 0;
    }
    T* val2 = (T*) aligned_alloc(64, m_row_number * sizeof(T));
    size_t count = 0;
    for(size_t i = 0 ; i < m_row_number; i++){
        bool found = false;
        T v = m_values[column_id][i];
        for(size_t e = 0; e < count; e++){
            if(val2[e] == v){
                found = true;
                break;
            }
        }
        if(!found){
            val2[count] = v;
            count++;
        }
    }
    free(val2);
    return count;
}

template class Table<uint32_t>;