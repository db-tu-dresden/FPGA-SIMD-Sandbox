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


enum Alignment{UNALIGNED, BAD, GOOD};

std::string alignment_to_string(Alignment x){
    switch(x){
        case Alignment::UNALIGNED:
            return "unaligned";
        case Alignment::BAD:
            return "bad";
        case Alignment::GOOD:
            return "good";
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
    mangled += BIT_NOISE2;
    mangled ^= (mangled >> 7);
    mangled *= BIT_NOISE3;
    mangled ^= (mangled << 17);

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
        }
    }
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


bool in_interval(size_t a, size_t s, size_t e){
    if(s < e){
        return a >= s && a <= e; 
    }else if(s > e){
        return a >= s || a <= e;
    }else{
        return true; // we have to assume that everything is meant
    }
}

size_t find_place(bool * blocked, size_t HSIZE, size_t place, size_t elements_to_place){


    size_t fist_blocked = HSIZE;
    bool loop = true;
    size_t etp = elements_to_place;
    std::vector<size_t> empty_space;
    std::vector<size_t> start_space;

    size_t empty = 0;
    size_t start = 0, end;

    bool enough_space = false;
    bool last_blocked = false;


    // collect information about the layout in form of intervals for the given array
    // this also includes if there even is a spot big enough for the placement
    for(size_t i = 0; i < HSIZE; i++){
        if(blocked[i]){
            if(!last_blocked){
                start_space.push_back(start);
                empty_space.push_back(empty);
                enough_space |= (empty >= elements_to_place);
                last_blocked = true;
            }
        }else{
            if(last_blocked){
                start = i;
                empty = 0;
                last_blocked = false;
            }
            empty++;
        }
    }
    if(empty_space.size() > 0 && !last_blocked){
        empty_space[0] += empty;
        start_space[0] = start;
        enough_space |= (empty_space[0] >= elements_to_place);
    }else if(!last_blocked){
        empty_space.push_back(empty);
        start_space.push_back(start);
        enough_space |= (empty >= elements_to_place);
    }
    




    for(size_t i = 0; i < start_space.size(); i++){
        std::cout << i << ":\t" << start_space[i] << ",\t" << empty_space[i] << std::endl;
    }


    if(!enough_space){   // we have not enough space for this collision!
        return HSIZE;
    }
    
//Find the nearest spot on or after to start searching for placement    
    size_t total_entries = start_space.size();
    size_t entry = total_entries;
    size_t entry_alt = total_entries;
    for(size_t i = 0; i < start_space.size(); i++){
        if(in_interval(place, start_space[i], (start_space[i] + empty_space[i] + HSIZE - 1 ) % HSIZE)){
            entry = i;
            break;
        }
        if(in_interval(place, start_space[i], HSIZE-1)){
            entry_alt = i;
        }
    }

    if(entry == total_entries){
        entry = entry_alt;
    }

    entry %= empty_space.size();
    size_t i = entry;
    std::cout << "\nwish starting place: " << place << std::endl;
    do{
        std::cout << "\twish entry: " << entry << "\ttest entry: " << i << std::endl;
        if(empty_space[i] >= elements_to_place){
            start = start_space[i];
            end = ((start + empty_space[i] + HSIZE - 1) % HSIZE);
            std::cout << "\t\t[" << start << ", " << end << "]\n";
            

            size_t s2, e2;
            bool s_okay = false, e_okay = false;
            s2 = place;
            e2 = (s2 + elements_to_place - 1) % HSIZE;
            s_okay = in_interval(s2, start, end);
            e_okay = in_interval(e2, start, end);
            std::cout << "\t\t\t[" << s2 << ", " << e2 << "]\t" << s_okay << " " << e_okay << std::endl;

            while(!s_okay || !e_okay){
                if(s_okay && !e_okay){
                    s2 += (HSIZE - 1); //save way to roll over.
                }else{
                    s2 += 1;
                }
                s2 %= HSIZE;
                e2 = (s2 + elements_to_place - 1) % HSIZE;
                s_okay = in_interval(s2, start, end);
                e_okay = in_interval(e2, start, end);
                std::cout << "\t\t\t[" << s2 << ", " << e2 << "]\t" << s_okay << " " << e_okay << std::endl;
            }
            std::cout << "FOUND A PLACE: " << s2 << std::endl;
            return s2;
        }
        i = (i + 1) % empty_space.size();
    }while(i != entry);

    return 0; //some error occured
    // gives us all the sizes of the concecutive empty spaces.

}

size_t make_place(size_t*& member_count, bool*& blocked, size_t HSIZE, size_t place, size_t elements_to_place){
    // this method should only be used for the unaligned usecase. otherwise the alignment suffers.
    
    std::vector<size_t> used_space;
    std::vector<size_t> start_used_space;

    std::vector<size_t> empty_space;
    std::vector<size_t> start_space;

    size_t space_used_total = 0;
    for(size_t i = 0; i < HSIZE; i++){
        if(member_count[i] != 0){
            start_used_space.push_back(i);
            used_space.push_back(member_count[i]);
            space_used_total += member_count[i];
        }
    }

    if(used_space.size() <= 1 || elements_to_place + space_used_total > HSIZE){
        // Problem with the hashtable. Only one entry and this method got called.
        return HSIZE;
    }

    size_t start, end, empty, max = 0, max_pos;

    for(size_t i = 0; i < start_used_space.size() - 1; i++){
        start = start_used_space[i] + used_space[i];
        end = start_used_space[i + 1];
        empty = (((end + HSIZE) - start) + HSIZE) % HSIZE;
        
        if(max < empty){
            max = empty;
            max_pos = empty_space.size();
        }

        empty_space.push_back(empty);
        start_space.push_back(start);
    }
    
    size_t missing = elements_to_place - max;
    std::vector<size_t> place_to_steal;

    while(missing > 0){


    }




    return HSIZE;
}


void entry_place(size_t*& member_count, bool*& blocked, size_t HSIZE, size_t place, size_t elements_to_place){
    std::cout << std::endl << "Write to place: " << place << " with a size of " << elements_to_place << std::endl;
    std::cout << "start place:\t";
    for(size_t i = 0; i < HSIZE; i++){
        std::cout << blocked[i] << "  ";
    }

    size_t p = place%HSIZE;
    for(size_t i = 0; i < elements_to_place; i++){
        if(blocked[p]){
            std::cout << std::endl << p << std::endl;
            throw std::runtime_error("Bad entry location. Already used!");
        }else{
            blocked[p] = true;
        }
        p = (p+1) % HSIZE;
    }
    member_count[place] = elements_to_place;
    
    std::cout << std::endl << "end   place:\t";
    for(size_t i = 0; i < HSIZE; i++){
        std::cout << blocked[i] << "  ";
    }
    std::cout << std::endl << std::endl;
}


void place_collisions_unaligned(size_t*& individual_group_size, size_t*& member_count, bool*& blocked, size_t HSIZE, size_t groups, size_t to_place_total, size_t position){
    size_t unused = HSIZE - to_place_total;
    size_t groups_left = groups;
    size_t still_to_place = to_place_total;
    size_t running_pos = position % HSIZE;
    size_t elements_to_place = 0;

    size_t n_pos, k;
    for(size_t i = 0; i < groups; i++){
        elements_to_place = individual_group_size[i];
        n_pos = find_place(blocked, HSIZE, running_pos, elements_to_place);
        entry_place(member_count, blocked, HSIZE, n_pos, elements_to_place);

        k = unused / groups_left;

        running_pos = n_pos + elements_to_place + k;
        unused -= k;
        groups_left--;
    }
}

void place_collisions_aligned(size_t*& individual_group_size, size_t*& member_count, bool*& blocked, size_t HSIZE, size_t groups, size_t to_place_total, size_t alignment){
    size_t unused = HSIZE - to_place_total;
    size_t groups_left = groups;
    size_t still_to_place = to_place_total;
    size_t running_pos = 0;
    size_t elements_to_place = 0;
    
    size_t n_pos, k;
    for(size_t i = 0; i < groups; i++){
        elements_to_place = individual_group_size[i];
        n_pos = find_place(blocked, HSIZE, running_pos, elements_to_place);
        entry_place(member_count, blocked, HSIZE, n_pos, elements_to_place);

        k = unused / groups_left;

        running_pos = n_pos + elements_to_place + k;
        unused -= k;
        groups_left--;
    }

}


//returns 0 iff their was a problem during generation
// static perfect hashing (mod HSIZE)
template<typename T>
size_t generate_collision_data(
    T*& result, 
    size_t data_size,   // number of values to be generated
    size_t distinct_values, // number of distinct values
    size_t HSIZE, // hash_table_size
    size_t collision_groups = 0,  // 0 means no collisions!
    double collisions = 0.0, //[0, 1] 0 no collision 1 all collision
    // Generation gen = Generation::FLAT,
    Distribution dis = Distribution::UNIFORM,
    Alignment ali = Alignment::UNALIGNED,
    size_t allign_size = 16,
    size_t start = 0,   // starting offset for consecutive numbers (dense)
    size_t seed = 0     // for sparse number generation 0 true random, 1.. reproducible
){
    if(seed == 0){  
        srand(std::time(nullptr));
        seed = std::rand();
    }
    bool create_collisions = true;
    if(collision_groups == 0){
        collision_groups = 0;
        collisions = 0;
        create_collisions = false;
    }
    collisions = collisions <= 1 ? collisions : 1;
    size_t collisions_total = distinct_values * collisions;
    size_t non_collisions = distinct_values - collisions_total;

    if(create_collisions && collisions_total/collision_groups < 2){
        std::cout << "To many groups for the given collisions!\n";
        return 0;
    }

    size_t *member_count = new size_t[HSIZE];
    bool *blocked = new bool[HSIZE];

    for(size_t i = 0; i < HSIZE; i++){
        member_count[i] = 0;
        blocked[i] = false;
    }

    if(create_collisions){
        size_t *individual_group_size = new size_t[collision_groups];
        
        if(true){   //uniform
            size_t help = collisions_total;
            for(size_t i = collision_groups; i > 0; i--){
                size_t current = help / i;
                help -= current;
                individual_group_size[i-1] = current;
            }
        }else if(false){    //quadratic //important for not enough collisions quadratic might act like uniform
            size_t help = collisions_total;
            for(size_t i = collision_groups; i > 0; i--){
                size_t current = (help / ((1<<i) - 1.)) + 0.5;
                current = current > 2 ? current : 2;
                help -= current;
                individual_group_size[i-1] = current;
            }
        }else{
            return 0;
        }

        bool placed_all = false;
        size_t to_place_id = 0;
        size_t t = 0;

        std::cout << collisions_total  << " collisions in " << collision_groups << " Groups\n";
        for(size_t i = 0; i < collision_groups; i++){
            std::cout << individual_group_size[i] << "\t";
        }std::cout << std::endl << std::endl;

        size_t space_wo_collisions = HSIZE - collisions_total;
        size_t swo = space_wo_collisions, cgc = collision_groups;
        
        size_t elements_to_place = individual_group_size[to_place_id];
        size_t place, next_place;
            

        switch (ali)
        {
            case Alignment::UNALIGNED:
                place = noise(to_place_id + t, seed)  % HSIZE;

                place_collisions_unaligned(individual_group_size, member_count, blocked, HSIZE, collision_groups, collisions_total, place);

            case Alignment::GOOD:   // try to place the blocks as good as possible for avx/sse

                break;
            case Alignment::BAD:    // try to place the blocks as badly as possible for avx/sse
                break;
            default:
                break;
        }

        do{
            switch (ali)
            {
            case Alignment::UNALIGNED:
                place = noise(to_place_id + t, seed)  % HSIZE;
                next_place = find_place(blocked, HSIZE, place, elements_to_place);
                if(next_place >= HSIZE){ //we need to "defragment" our hash table other wise we might create more complex collisions
                    next_place = make_place(member_count, blocked, HSIZE, place, elements_to_place);
                }
                entry_place(member_count, blocked, HSIZE, next_place, elements_to_place);
                to_place_id++;
                break;

            case Alignment::GOOD:   // try to place the blocks as good as possible for avx/sse

                break;
            case Alignment::BAD:    // try to place the blocks as badly as possible for avx/sse
                break;
            default:
                break;
            }
            t++;
        }while(to_place_id < collision_groups);
        



    
    }







    return seed;
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