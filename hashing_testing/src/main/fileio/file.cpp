#include <stdexcept>

#include "file.hpp"

void create_result_file(std::string filename, size_t test_number){
    std::ofstream myfile;
    myfile.open (filename);
    if(myfile.is_open()){

        switch(test_number){
            case 3:
                myfile << "Algorithm,time,data_size,bytes,distinct_value_count,scale,hash_table_size,hash_function_ID,seed,run_ID,collision_count,collision_length,layout,layout,config_ID\n";
                break;
            case 4:
                myfile << "Algorithm,time,data_size,bytes,distinct_value_count,scale,hash_table_size,hash_function_ID,seed,run_ID,collision_count,collision_length,layout,space,config_ID\n";
                break;
            default:
                myfile << "Algorithm,time,data_size,bytes,distinct_value_count,scale,hash_table_size,hash_function_ID,seed,run_ID,collision_count,collision_length,cluster_count,cluster_length,config_ID\n";
                break;
        }
        myfile.close();
    } else {
        throw std::runtime_error("Could not open file to write results!");
    }
}

void write_to_file( 
    std::string filename,
    std::string alg_identification,
    uint64_t time, 
    size_t data_size,
    size_t bytes,
    size_t distinct_value_count,
    float scale,
    size_t HSIZE,
    HashFunction hash_function_enum, 
    size_t seed,  
    size_t rsd,
    size_t config_collision_count,
    size_t config_collition_size,
    size_t config_cluster_count,
    size_t config_cluster_size, 
    size_t config_id
){
    std::ofstream myfile;
    myfile.open (filename, std::ios::app);
    if(myfile.is_open()){
        // "Algorithm,time,data size,bytes,distinct value count,scale,hash table size,hash function ID,seed,run ID";
        myfile << alg_identification << "," << time << "," 
            << data_size << "," << bytes 
            << "," << distinct_value_count  << "," << scale << "," 
            << HSIZE << "," << get_hash_function_name(hash_function_enum) << "," 
            << seed << "," << rsd << "," 
            << config_collision_count << "," << config_collition_size << "," 
            << config_cluster_count << "," << config_cluster_size << ","
            << config_id << "\n"; 
        myfile.close();
    } else {
        throw std::runtime_error("Could not open file to write results!");
    }
}
