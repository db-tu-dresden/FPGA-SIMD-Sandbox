#include <iostream>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>


#include "../operator/physical/group_count/scalar_group_count.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v1.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v2.hpp"
#include "../operator/physical/group_count/avx512_group_count_soa_v3.hpp"
// #include "../operator/physical/group_count/avx512_group_count_soaov_v1.hpp"

#include "datagen.hpp"

// simple multiplicative hashing function
uint32_t hashx(uint32_t key, size_t HSIZE);

template <typename T>
T id_mod(T key, size_t HSIZE);

template <typename T>
void hashall(T from, T to, T step, size_t HSIZE, T(*hash_function)(T, size_t));


template <typename T> 
size_t createCountValidationTable(T** res_table, T** res_count, T* data, size_t data_size, size_t HSIZE);

template <typename T> 
bool validate(Group_count<T>* grouping, T* table_value, T* table_count, size_t data_size);

template <typename T>
bool run_test(Group_count<T>* group_count, T* data, size_t data_size, T* validation_value, T* validation_count, size_t validation_size, bool cleanup = true);
    


std::chrono::high_resolution_clock::time_point time_now();
uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end);



// using ps_type = uint64_t;
using ps_type = uint32_t;

int main(int argc, char** argv){

    size_t distinct_value_count = 40; // setting the number of Distinct Values
    float scale = 1.2f; // setting the hash map scaling factor
    size_t data_size = 16 * 10;//0000;; // setting the number of entries
    
    ps_type* data = new ps_type[data_size];  

    ps_type (*function) (ps_type, size_t) = &hashx; // setting the function
    size_t HSIZE = (size_t)(scale * distinct_value_count + 0.5f);


//Generate and prepare Validation data.
    std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    std::cout << "Generate Data\n\t" << data_size << " entries with " << distinct_value_count << " distinct values with " << sizeof(ps_type) * 8 << "bit\n";    
    generate_data<ps_type>(data, data_size, distinct_value_count, Density::SPARSE);
    
    for(size_t i = 0; i < data_size; i++){
        data[i] = data[i] % (HSIZE * 30);
    }

    //generating data for validation so that we only need to calculate it once per data
    ps_type *table_value;
    ps_type *table_count;
    std::cout << "Prepare Validation data\n";
    size_t validation_size = createCountValidationTable(&table_value, &table_count, data, data_size, HSIZE);
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";

    // Run test with the given configuaration
    run_test<ps_type>(new Scalar_group_count<ps_type>(HSIZE, function), data, data_size, table_value, table_count, validation_size);
    run_test<ps_type>(new AVX512_group_count_SoA_v1<ps_type>(HSIZE, function), data, data_size, table_value, table_count, validation_size);
    run_test<ps_type>(new AVX512_group_count_SoA_v2<ps_type>(HSIZE, function), data, data_size, table_value, table_count, validation_size);
    run_test<ps_type>(new AVX512_group_count_SoA_v3<ps_type>(HSIZE, function), data, data_size, table_value, table_count, validation_size);
    // run_test<ps_type>(new AVX512_group_count_SoAoV_v1<ps_type>(HSIZE, function), data, data_size, table_value, table_count, validation_size);

}




/// @brief Executes the hash function and collecting performance Data. 
/// @tparam T 
/// @param group_count The group_count operation that shall be executed.
/// @param data The data on which the operation shall be evaluated
/// @param data_size 
/// @param validation_value //Information for validation Key column
/// @param validation_count //Information for validation count column
/// @param validation_size 
/// @param cleanup true if group_count should be delete when the benchmark is finished. 
template <typename T>
bool run_test(Group_count<T>* group_count, T* data, size_t data_size, T* validation_value, T* validation_count, size_t validation_size, bool cleanup){
// prepare for the testing of the function    
    std::chrono::high_resolution_clock::time_point time_begin, time_end;
    uint64_t duration;
    double duration_s;
    double data_amount = (data_size * sizeof(ps_type) * 8)/1000000000.0; // Gbit
    double data_count = data_size / 1000000000.0; // Million Values

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    std::cout << group_count->identify() << std::endl;
// run the test and time it
    time_begin = time_now();
    group_count->create_hash_table(data, data_size);;
    time_end = time_now();
    
    std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
// out put result
    duration = duration_time(time_begin, time_end);
    duration_s = duration / 1000000000.0;

    std::cout << "\tTime:\t" << duration << " ns\n";
    std::cout << "\tTime:\t" << duration_s << " s\n";
    std::cout << "\tData:\t" << data_amount << " Gbit\n"; 
    std::cout << "\tData:\t" << data_size << " Values\n"; 
    std::cout << "\ttput:\t" << (data_amount)/(duration_s) << " Gbit/s\n";
    std::cout << "\tperf:\t" << (data_count)/(duration_s) << " Gval/s\n";
    std::cout << "\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

// validate run
    bool errors = validate<ps_type>(group_count, validation_value, validation_count, validation_size);
    // if(errors)
    // {
    //     group_count->print(false);
    // }
    if(cleanup){
        free(group_count);
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";
    return errors;
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
/// @param data_size number of entries in the data array
/// @param HSIZE the size of the Hash table 
/// @return the number of slots used in the result arrays
template <typename T> 
size_t createCountValidationTable(T** res_table, T** res_count, T* data, size_t data_size, size_t HSIZE){
    *res_table = new T[HSIZE];
    *res_count = new T[HSIZE];
    size_t m_id = 0;
    
    for(size_t p = 0; p < data_size; p++){
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
bool validate(Group_count<T>* grouping, T* table_value, T* table_count, size_t data_size){
    std::cout << "Start Validation\n";
    size_t nr_of_errors = 0;
    for(size_t i = 0; i < data_size; i++){
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

