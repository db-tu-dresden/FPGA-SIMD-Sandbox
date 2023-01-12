#include <iostream>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>


#include "../operator/physical/group_count/scalar_group_count.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v1.hpp"

#include "datagen.hpp"

// simple multiplicative hashing function
uint32_t hashx(uint32_t key, size_t HSIZE);

template <typename T>
T id_mod(T key, size_t HSIZE);

template <typename T>
void hashall(T from, T to, T step, size_t HSIZE, T(*hash_function)(T, size_t));


template <typename T> 
size_t createCountValidationTable(T** res_table, T** res_count, T* data, size_t dataSize, size_t HSIZE);

template <typename T> 
bool validate(Group_count<T>* grouping, T* table_value, T* table_count, size_t dataSize);

template <typename T>
void run_test(Group_count<T>* group_count, T* data, size_t dataSize, T* validation_value, T* validation_count, size_t validation_size);
    


std::chrono::high_resolution_clock::time_point time_now();
uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end);



// using ps_type = uint64_t;
using ps_type = uint32_t;

int main(int argc, char** argv){
    size_t distinctValuesCount = 12;
    float scale = 1.8f;
    size_t HSIZE = (size_t)(scale * distinctValuesCount + 0.5f);
    size_t dataSize = 16 * 4;//000000;
    ps_type* data = new ps_type[dataSize];


    std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    std::cout << "Generate Data\n";    
    // generate_data<ps_type>(data, dataSize, distinctValuesCount, Density::SPARSE);
    // for(size_t i = 0; i < dataSize; i++){
    //     data[i] = data[i] % (HSIZE * 30);
    // }
    dataSize = 5;
    data = new ps_type[dataSize];
    data[0] = 247;
    data[1] = 518;
    data[2] = 313;
    data[3] = 518;
    data[4] = 247;
    
    //generating data for validation so that we only need to calculate it once per data
    ps_type *table_value;
    ps_type *table_count;
    size_t slots = createCountValidationTable(&table_value, &table_count, data, dataSize, HSIZE);
    std::cout << "Prepare Validation data\n";
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";

    run_test<ps_type>(new Scalar_group_count<ps_type>(HSIZE, &hashx), data, dataSize, table_value, table_count, slots);
    run_test<ps_type>(new AVX512_group_count_SoA_v1<ps_type>(HSIZE, &hashx), data, dataSize, table_value, table_count, slots);

}









/// @brief 
/// @tparam T 
/// @param group_count 
/// @param data 
/// @param dataSize 
/// @param validation_value 
/// @param validation_count 
/// @param validation_size 
template <typename T>
void run_test(Group_count<T>* group_count, T* data, size_t dataSize, T* validation_value, T* validation_count, size_t validation_size){
    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    uint64_t duration;
    double duration_s;
    double data_amount = (dataSize * sizeof(ps_type) * 8)/1000000000.0; // Gbit
    double data_count = dataSize / 1000000000.0; // Million Values

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    std::cout << group_count->identify() << std::endl;

    time_begin = time_now();
    group_count->create_hash_table(data, dataSize);;
    time_end = time_now();
    
    std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    duration = duration_time(time_begin, time_end);
    duration_s = duration / 1000000000.0;

    std::cout << "\tTime:\t" << duration << " ns\n";
    std::cout << "\tTime:\t" << duration_s << " s\n";
    std::cout << "\tData:\t" << data_amount << " Gbit\n"; 
    std::cout << "\tData:\t" << data_count << " Million Values(Gval)\n"; 
    std::cout << "\ttput:\t" << (data_amount)/(duration_s) << " Gbit/s\n";
    std::cout << "\tperf:\t" << (data_count)/(duration_s) << " Gval/s\n";
    std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    bool errors = validate<ps_type>(group_count, validation_value, validation_count, validation_size);
    // if(errors)
    // {
    //     group_count->print(false);
    // }
    free(group_count);
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";
}



// simple multiplicative hashing function
uint32_t hashx(uint32_t key, size_t HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

template<typename T>
T id_mod(T key, size_t HSIZE) {
    return key % HSIZE;
}

template <typename T>
void hashall(T from, T to, T step, size_t HSIZE, T(*hash_function)(T, size_t)){
    for(T i = from; i < to; i += step){
        std::cout << "\t" << hash_function(i, HSIZE);
    }
    std::cout << std::endl;
}

/*
    creates a time point
*/
std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

/*
    gives the time between begin and end in nanoseconds
*/
uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

}

/// @brief Creates a arrays with the expected result values, that can be used to evaluate
/// @tparam T 
/// @param res_table pointer to the result array containing all values
/// @param res_count pointer to the result array containing the counts
/// @param data array contaning all the values
/// @param dataSize number of entries in the data array
/// @param HSIZE the size of the Hash table 
/// @return the number of slots used in the result arrays
template <typename T> 
size_t createCountValidationTable(T** res_table, T** res_count, T* data, size_t dataSize, size_t HSIZE){
    *res_table = new T[HSIZE];
    *res_count = new T[HSIZE];
    size_t m_id = 0;
    
    for(size_t p = 0; p < dataSize; p++){
        T value = data[p];
        bool found = false;
        for(size_t i = 0; i < m_id; i++){
            if((*res_table)[i] == value){
                (*res_count)[i]++;
                found = true;
                break;
            }
        }
        if(!found){
            (*res_table)[m_id] = value;
            (*res_count)[m_id] = 1;
            m_id++;
        }
    }
    return m_id;
}


template <typename T> 
bool validate(Group_count<T>* grouping, T* table_value, T* table_count, size_t dataSize){
    std::cout << "Start Validation\n";
    size_t nr_of_errors = 0;
    for(size_t i = 0; i < dataSize; i++){
        T value = table_value[i];
        size_t count = grouping->get(value);

        if(count != table_count[i]){
            std::cout << "\tERROR Count\t" << value << " has a count of\t" 
                << count << " but expected to have\t" << table_count[i] << std::endl;
            nr_of_errors++;
        }
    }
    std::cout << "End of Validation";
    if(nr_of_errors == 1){
        std::cout << "\tFound one Error\n";
    }else if(nr_of_errors > 1){
        std::cout << "\tFound " << nr_of_errors << " Errors\n";
    }else{
        std::cout << std::endl;
    }
    return nr_of_errors != 0;
}

