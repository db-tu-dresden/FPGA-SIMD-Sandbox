#include <iostream>
#include <stdint.h>
#include <vector>
#include <cstdlib>
#include <chrono>


#include <unordered_map>
#include "datagenerator/datagen.hpp"
#include "hash_function.hpp"
#include "benchmark/table.hpp"


int main(int argc, char** argv){
    
    Table<uint32_t> orders("orders_short.tbl");   
    // uint32_t* column = orders.get_column(1);
    std::cout << orders.get_distinct_values(2) << std::endl;
    // for(size_t i = 0; i < orders.get_row_count(); i++){
    //     std::cout << column[i] << ", ";
    // }
    // std::cout << std::endl;


}
