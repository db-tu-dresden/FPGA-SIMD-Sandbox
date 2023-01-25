#ifndef TUD_HASHING_TESTING_DATAGEN
#define TUD_HASHING_TESTING_DATAGEN

#include <stdint.h>
#include <stdlib.h>
#include <vector>

/*
    Density sets the information about the data layout between the values.
    In the DENSE Case this should mean that the numbers are in an interval from [x:y]
        with a step size of 1.
    In the SPARSE case the numbers are taken at random from the full range of values.    
*/
enum Density{DENSE, SPARSE};

std::string density_to_string(Density x){
    switch(x){
        case Density::DENSE:
            return "dense";
        case Density::SPARSE:
            return "sparse";
    }
    return "unknown";
}

/*
    The Distribution gives us another tuning factor.
    For this we disregard the order of the keys.
    With NORMAL we try to generate the data such that the different keys follow a
        normal distributuion.
    With UNIFORM we try to achieve a uniform distribution. This includes some variations
        in the relative frequencies.
*/
enum Distribution{NORMAL, UNIFORM};

std::string distribution_to_string(Distribution x){
    switch(x){
        case Distribution::NORMAL:
            return "normal";
        case Distribution::UNIFORM:
            return "uniform";
    }
    return "unknown";
}

/*
    With Generation we try to achieve the same affect the paper: "A Seven-Dimensional 
        Analysis of Hashing Methods and its Implications on Query Processing"
    With the FLAT generation we have no prerequirements of how the data should look like.
    GRID on the other hand has the prerequrement that every 
*/
enum Generation{FLAT, GRID};

std::string generation_to_string(Generation z){
    switch(z){
        case Generation::FLAT:
            return "flat";
        case Generation::GRID:
            return "grid";
    }
    return "unknown";
}

template<typename T>
T make_grid(size_t x){
    size_t halfword[] ={0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE};
    size_t nr_half = 14;
    T result = 0;

    for(size_t i = 0; i < sizeof(T); i++){
        size_t k = x % nr_half;
        x /= nr_half;
     
        result ^= halfword[k] << (8 * i);
    }
    
    return result;
}

//TODO CHECK IF it is okay to use.
uint64_t noise(size_t position, size_t seed){
    size_t BIT_NOISE1 = 0x68E31DA4;
    size_t BIT_NOISE2 = 0xB5297A4D;
    size_t BIT_NOISE3 = 0x1B56C4E9;

    // size_t BIT_NOISE1 = 0x7FFFFFFFFFFFFF5B;
    // size_t BIT_NOISE2 = 0x68E31DA4B5297A4D;
    // size_t BIT_NOISE3 = 0xA8F8628ADC17D6CB;

    uint64_t mangled = position;
    mangled *= BIT_NOISE1;
    mangled += seed;
    mangled ^= (mangled << 13);
    // mangled ^= (mangled >> 8);
    mangled += BIT_NOISE2;
    mangled ^= (mangled >> 7);
    // mangled ^= (mangled << 8);
    mangled *= BIT_NOISE3;
    mangled ^= (mangled << 17);
    // mangled ^= (mangled >> 8);

    // mangled *= BIT_NOISE1;
    // mangled += seed;
    // mangled ^= (mangled << 13);
    // // mangled ^= (mangled >> 8);
    // mangled += BIT_NOISE2;
    // mangled ^= (mangled >> 7);
    // // mangled ^= (mangled << 8);
    // mangled *= BIT_NOISE3;
    // mangled ^= (mangled << 17);

    return mangled;
}


/*
    helping function that turns the index into random numbers
*/
template<typename T>
void flat_number_gen(
    std::vector<T>& numbers,
    std::vector<size_t> index,
    size_t distinct_values
){
    for(size_t i = 0; numbers.size() < distinct_values && i < index.size(); i++){
        T num = index[i];
        bool in = (num == 0);

        for(T x: numbers){
            if(x == num){
                in = true;
                break;
            }
        }

        if(!in){
            numbers.push_back((T)num);
            // std::cout << "\t" << num; 
        }
    }
    // std::cout << std::endl;
}


/*
    helping function that turns the index into random numbers
*/
template<typename T>
void grid_number_gen(
    std::vector<T>& numbers,
    std::vector<size_t> index,
    size_t distinct_values
){
    for(size_t i = 0; numbers.size() < distinct_values && i < index.size(); i++){
        size_t id = index[i];
        T num = make_grid<T>(i);
        bool in = (num == 0);

        for(T x: numbers){
            if(x == num){
                in = true;
                break;
            }
        }

        if(!in){
            numbers.push_back((T)num);
        }
    }
}

/*
    random number generator for different layouts dense and sparse
*/
void index_dense(std::vector<size_t>& index, size_t distinct_values, size_t start){
    for(size_t i = start; i < distinct_values+start; i++){
        index.push_back(i);
    }
}

void index_sparse(std::vector<size_t>&index, size_t distinct_values, size_t seed){   
    for(size_t i = 1; i <= distinct_values; i++){
        index.push_back(noise(i, seed));
    }
}


/*
    Data generator with different options for data layout. 
    DOES NOT ALLOCATE THE MEMORY JUST FILLS IT!
*/
template<typename T>
size_t generate_data(
    T*& result, 
    size_t data_size,   // number of values to be generated
    size_t distinct_values, // number of distinct values
    Density den = Density::DENSE,
    Generation gen = Generation::FLAT,
    Distribution dis = Distribution::UNIFORM,
    size_t start = 0,   // starting offset for consecutive numbers (dense)
    size_t seed = 0     // for sparse number generation 0 true random, 1.. reproducible
){
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    }
    // std::cout << "\tThe seed is:\t" << seed << std::endl;
// std::cout << "Generate " << data_size << " Entries with a seed of " << seed << " and following characteristics: " 
//     << density_to_string(den) << " " << generation_to_string(gen) << " " << distribution_to_string(dis) << std::endl;
    
    double mul = 1.5;
    size_t retries = 0;
retry:
    mul++;
    retries++;
    if(retries < 10)
    {
        std::vector<size_t> index;
        std::vector<T> numbers;
        switch(den){
        case Density::DENSE:
            index_dense(index, distinct_values*mul, start);
            break;
        case Density::SPARSE:
            index_sparse(index, distinct_values*mul, seed);
            break;    
        default:
            throw std::runtime_error("Unknown Density input");
        }
        
        switch (gen){
        case Generation::FLAT:
            flat_number_gen<T>(numbers, index, distinct_values);
            break;
        case Generation::GRID:
            grid_number_gen<T>(numbers, index, distinct_values);
            break;
        default:
            throw std::runtime_error("Unknown Generation methoed input");    
        }
        
        if(numbers.size() < distinct_values){
            goto retry;
        }
        switch(dis){
        case Distribution::NORMAL:
            throw std::runtime_error("Normal Distribution not yet implemented");    
            break;
        case Distribution::UNIFORM:
            for(size_t i = 0; i < data_size; i++){
                size_t ran = noise(i, seed + start + 1) % distinct_values;
                result[i] = numbers[ran];
            }
            break;
        default:
            throw std::runtime_error("Unknown Distribution input");    
        }
    }else{
        throw std::runtime_error("To many retries during data generation.");
    }
    return seed;
}


/*
    Data generator with different options for data layout. 
    DOES NOT ALLOCATE THE MEMORY JUST FILLS IT!
*/
template<typename T>
void generate_data2(
    T*& result, 
    size_t data_size,   // number of values to be generated
    size_t distinct_values, // number of distinct values
    Density den = Density::DENSE,
    Generation gen = Generation::FLAT,
    Distribution dis = Distribution::UNIFORM,
    size_t start = 0,   // starting offset for consecutive numbers (dense)
    size_t seed = 0     // for sparse number generation 0 true random, 1.. reproducible
){
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    }
    std::cout << "\tThe seed is:\t" << seed << std::endl;
    double mul = 1.5;
    size_t retries = 0;
retry:
    mul++;
    retries++;
    if(retries < 10)
    {
        std::vector<size_t> index;
        std::vector<T> numbers;

        switch(den){
        case Density::DENSE:
            index_dense(index, distinct_values*mul, start);
            break;
        case Density::SPARSE:
            index_sparse(index, distinct_values*mul, seed);
            break;    
        default:
            throw std::runtime_error("Unknown Density input");
        }

        switch (gen){
        case Generation::FLAT:
            flat_number_gen<T>(numbers, index, distinct_values);
            break;
        case Generation::GRID:
            grid_number_gen<T>(numbers, index, distinct_values);
            break;
        default:
            throw std::runtime_error("Unknown Generation methoed input");    
        }
        
        if(numbers.size() < distinct_values){
            goto retry;
        }

        switch(dis){
        case Distribution::NORMAL:
            throw std::runtime_error("Normal Distribution not yet implemented");    
            break;
        case Distribution::UNIFORM:
            for(size_t i = 0; i < data_size; i++){
                size_t ran = noise(i, seed + start + 1) % distinct_values;
                result[i] = numbers[ran];
            }
            break;
        default:
            throw std::runtime_error("Unknown Distribution input");    
        }
    }else{
        throw std::runtime_error("To many retries in data gen.");
    }
}


#endif //TUD_HASHING_TESTING_DATAGEN