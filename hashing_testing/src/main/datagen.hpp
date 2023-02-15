#ifndef TUD_HASHING_TESTING_DATAGEN
#define TUD_HASHING_TESTING_DATAGEN

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>

#include <sstream>
#include <string>

#include "datagen_help.hpp"


// todo with one big cluster where we just place the data in a hashmap the same size of distinct data (so we just create collisions on the smallest size)
// todo cluster leaks into collisions!


/*
    Density sets the information about the data layout between the values.
    In the DENSE Case this should mean that the numbers are in an interval from [x:y]
        with a step size of 1.
    In the SPARSE case the numbers are taken at random from the full range of values.    
*/
enum class Density{DENSE, SPARSE};

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
enum class Distribution{NORMAL, UNIFORM};

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
enum class Generation{FLAT, GRID};

std::string generation_to_string(Generation z){
    switch(z){
        case Generation::FLAT:
            return "flat";
        case Generation::GRID:
            return "grid";
    }
    return "unknown";
}



/*
    How collisions groups should be aligned. BAD and GOOD come with another parameter on which it should be alignt 
*/
enum class Alignment{UNALIGNED, BAD, GOOD};

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

/*
    How collisions groups should be aligned. BAD and GOOD come with another parameter on which it should be alignt 
*/
enum Groupsize{UNIFORM, QUADRATIC};
std::string alignment_to_string(Groupsize x){
    
    switch(x){
        case Groupsize::UNIFORM:
            return "uniform";
        case Groupsize::QUADRATIC:
            return "quadratic";
    }
    
    return "unknown";
}


//Creates a GRID number described by the paper: A Seven Dimensional Analysis of Hashing Methods and its Implications on Query Processing
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
    




    // for(size_t i = 0; i < start_space.size(); i++){
    //     std::cout << i << ":\t" << start_space[i] << ",\t" << empty_space[i] << std::endl;
    // }


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
    // std::cout << "\nwish starting place: " << place << std::endl;
    do{
        // std::cout << "\twish entry: " << entry << "\ttest entry: " << i << std::endl;
        if(empty_space[i] >= elements_to_place){
            start = start_space[i];
            end = ((start + empty_space[i] + HSIZE - 1) % HSIZE);
            // std::cout << "\t\t[" << start << ", " << end << "]\n";
            

            size_t s2, e2;
            bool s_okay = false, e_okay = false;
            s2 = place;
            e2 = (s2 + elements_to_place - 1) % HSIZE;
            s_okay = in_interval(s2, start, end);
            e_okay = in_interval(e2, start, end);
            // std::cout << "\t\t\t[" << s2 << ", " << e2 << "]\t" << s_okay << " " << e_okay << std::endl;

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
                // std::cout << "\t\t\t[" << s2 << ", " << e2 << "]\t" << s_okay << " " << e_okay << std::endl;
            }
            // std::cout << "FOUND A PLACE: " << s2 << std::endl;
            return s2;
        }
        i = (i + 1) % empty_space.size();
    }while(i != entry);

    return 0; //some error occured
    // gives us all the sizes of the concecutive empty spaces.

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


//returns a number of unplaceable entries because of alignment conflicts. 
size_t place_collisions_aligned_good(size_t*& individual_group_size, size_t*& member_count, bool*& blocked, size_t HSIZE, size_t groups, size_t to_place_total, size_t alignment){
    size_t unused = HSIZE - to_place_total;
    size_t groups_left = groups;
    size_t still_to_place = to_place_total;
    size_t running_pos = 0;
    size_t elements_to_place = 0;
    size_t elements_placed = 0;
    
    size_t n_pos, k;
    size_t max_offset;
    size_t retries = 0;
    size_t max_retires = groups > HSIZE/alignment ? groups : HSIZE/alignment;

    size_t offset, bucket;
    for(size_t i = 0; i < groups; i++){
        retries = 0;
        elements_to_place = individual_group_size[i];
        max_offset = alignment - (size_t)(elements_to_place/2. + 0.5);

retry:
        n_pos = find_place(blocked, HSIZE, running_pos, elements_to_place);
        if(n_pos == HSIZE){
            return to_place_total - elements_placed;
        }

        if(retries >= max_retires){
            goto placing;
        }
        
        bucket = n_pos / alignment;
        offset = n_pos - bucket * alignment;
        
        if(!(offset <= max_offset)){
            running_pos = (running_pos + 1) % HSIZE;
            retries++;
            goto retry;
        }

placing:
        entry_place(member_count, blocked, HSIZE, n_pos, elements_to_place);
        elements_placed += elements_to_place;
        running_pos = (running_pos + elements_to_place) % HSIZE;
    }
    return 0;
}

size_t count_alignment_blocks(size_t HSIZE, size_t alignment, size_t position, size_t elements){
    std::cout << "\t\tcab\tpos: " << position;
    size_t count = 1;
    position = (position + 1) % HSIZE ;
    elements--;
    while(elements > 0){
        count += (position % alignment) == 0;
        position = (position + 1) % HSIZE; 
        elements --;

    }
    std::cout <<"\tcount: " << count << std::endl;
    return count;
}

size_t get_worst_alignment(size_t HSIZE, size_t alignment, size_t pos1, int64_t block_count1, size_t gap1, size_t pos2, int64_t block_count2, size_t gap2){
    
    //one has more blocks
    if(block_count1 > block_count2){
        return pos1;
    }else if(block_count2 > block_count1){
        return pos2;
    }

    //one is HSIZE and they have the same block count
    if(pos1 == HSIZE || pos2 == HSIZE){
        return HSIZE;
    }

    if(gap1 < gap2){
        return pos1;
    }else if(gap2 < gap1){
        return pos2;
    }

    //6 and 7 with 3 and 10
    //both have same block count. Search for the worse alignment.

    size_t elements_in_last_block = HSIZE % alignment;
    elements_in_last_block = elements_in_last_block == 0 ? alignment : elements_in_last_block;
    size_t patch = alignment - elements_in_last_block;
    
    bool last_block1 = (((pos1 / alignment) + 1) * alignment) % HSIZE < pos1; // checks if the block we are curretnly in is the last block. 
    bool last_block2 = (((pos2 / alignment) + 1) * alignment) % HSIZE < pos2; // checks if the block we are curretnly in is the last block. 

    size_t m_1 = pos1 % alignment;
    size_t m_2 = pos2 % alignment;
    if(last_block1){
        m_1 += patch;
    }
    if(last_block2){
        m_2 += patch;
    }

    if(m_2 > m_1){
        return pos2;
    }
    return pos1;
}

size_t get_gap_created(bool * blocked, size_t HSIZE, size_t pos, int64_t block_count){
    size_t gap = 0;
    size_t s = pos;
    size_t e = (pos + block_count) % HSIZE;
    size_t gap_help = 0;
    
    //search right
    for(size_t p = e; p != s; p = (p+1)%HSIZE){
        if(blocked[p]){
            break;
        }else{
            gap_help++;
        }
    }
    gap = gap_help;
    gap_help = 0;
    std::cout << "RIGHT: " << gap;
    //search left
    for(size_t p = (s+HSIZE-1)%HSIZE; p != e; p = (p+HSIZE-1)%HSIZE){
        if(blocked[p]){
            break;
        }else{
            gap_help++;
        }
    }
    
    std::cout << "\tLEFT: " << gap_help;
    if(gap_help < gap){
        gap = gap_help;
    }

    std::cout << "\tGAP: " << gap << std::endl;

    return gap;
}
 
size_t get_most_alignment_block_position(bool * blocked, size_t HSIZE, size_t alignment, size_t elements, size_t empty_space_allowed, bool override_gap = false){
    int64_t count[HSIZE];
    size_t gap[HSIZE];

    std::cout << "GET MOST\tGAP:" << empty_space_allowed << std::endl;
    for(size_t i = 0; i < HSIZE; i++){
        count[i] = -1;
        gap[i] = 0;
    }

    for(size_t i = 0; i < HSIZE; i++){
        if(count[i] == -1){
            size_t pos = find_place(blocked, HSIZE, i, elements);
            size_t gap_i = get_gap_created(blocked, HSIZE, pos, elements);
            

            if(pos > HSIZE){
                return HSIZE;
            }
            
            bool NOT_ALREADY_SET = count[pos] == -1;
            bool BIG_ENOUGH_GAP = gap_i >= elements; // the gap is big enough to contain another element of the same size
            bool SMALL_ENOUGH_GAP = gap_i <= empty_space_allowed; // the gap is smaller than the allowed free space

            if(NOT_ALREADY_SET && (BIG_ENOUGH_GAP || SMALL_ENOUGH_GAP || override_gap)){
                gap[pos] = gap_i;
                count[pos] = count_alignment_blocks(HSIZE, alignment, pos, elements);
                std::cout << "\t\t\t\t" << gap_i << std::endl;;
            }
        }
    }

    size_t max_id = HSIZE;
    int64_t max_val = 0;
    int64_t min_gap = 0;

    std::cout << "\t\t\tmax 1\t" << max_id << "\t" << max_val << std::endl;
    for(size_t i = 0; i < HSIZE; i++){

        max_id = get_worst_alignment(HSIZE, alignment, max_id, max_val, min_gap, i, count[i], gap[i]);
        if(i == max_id){
            max_val = count[i];
            min_gap = gap[i];
            std::cout << "\t\t\tmax\t" << max_id << "\t" << max_val << std::endl;
        }
    }
    return max_id;
}


//returns a number of unplaceable entries because of alignment conflicts. 
size_t place_collisions_aligned_bad(size_t*& individual_group_size, size_t*& member_count, bool*& blocked, size_t HSIZE, size_t groups, size_t to_place_total, size_t alignment){
    
    size_t groups_left = groups;
    size_t still_to_place = to_place_total;
    size_t running_pos = HSIZE - 1;
    size_t elements_to_place = 0;
    size_t elements_placed = 0;

    size_t empty_space = HSIZE - to_place_total;
    
    size_t n_pos, k;
    size_t max_loads, min_loads;    
    size_t min_offset;

    size_t retries = 0;
    size_t max_retires = groups > HSIZE/alignment ? groups : HSIZE/alignment;

    for(size_t i = 0; i < groups; i++){

        elements_to_place = individual_group_size[i];   //elements_to_place get smaller per iteration
        n_pos = get_most_alignment_block_position(blocked, HSIZE, alignment, elements_to_place, empty_space, i==0); 
        
        if(n_pos == HSIZE){
            return to_place_total - elements_placed;
        }

        size_t k = get_gap_created(blocked, HSIZE, n_pos, elements_to_place);
        if(k <= elements_to_place && k <= individual_group_size[(i+1) %HSIZE]){
            empty_space -= k;
        }else if(empty_space > 0){
            empty_space--; // due to unlucky placement we might get a "memory" leak.
        }
        
        entry_place(member_count, blocked, HSIZE, n_pos, elements_to_place);
        elements_placed += elements_to_place;
        running_pos = (running_pos + elements_to_place) % HSIZE;
    }

    std::cout << "ALL BLOCKS PLACED!\n";
    return 0;
}

























//returns 0 iff their was a problem during generation
// static perfect hashing (mod HSIZE)
// I COULD ADD THE FUNCTION TOO since it is just a simple search for values that hash to it. so it shouldn't take "that" long.
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
    Groupsize gro = Groupsize::UNIFORM,
    Alignment ali = Alignment::BAD,
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
        size_t help;

        switch(gro){
        case Groupsize::UNIFORM:
            help = collisions_total;
            for(size_t i = collision_groups; i > 0; i--){
                size_t current = help / i;
                help -= current;
                individual_group_size[i-1] = current;
            }

            break;
        case Groupsize::QUADRATIC:
            help = collisions_total;
            for(size_t i = collision_groups; i > 0; i--){
                size_t current = (help / ((1<<i) - 1.)) + 0.5;
                current = current > 2 ? current : 2;
                help -= current;
                individual_group_size[i-1] = current;
            }
            break;
        default:
            throw std::runtime_error("Unknown groupsize distribution");    

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
        std::cout << "point 1\n";

        size_t error = 0;
        switch (ali)
        {
            case Alignment::UNALIGNED:
                place = noise(to_place_id + t, seed + 1)  % HSIZE;

                place_collisions_unaligned(individual_group_size, member_count, blocked, HSIZE, collision_groups, collisions_total, place); //can't fail!
                break;
            case Alignment::GOOD:   // try to place the blocks as good as possible for avx/sse
                error = place_collisions_aligned_good(individual_group_size, member_count, blocked, HSIZE, collision_groups, collisions_total, allign_size);
                break;
            case Alignment::BAD:    // try to place the blocks as badly as possible for avx/sse
                error = place_collisions_aligned_bad(individual_group_size, member_count, blocked, HSIZE, collision_groups, collisions_total, allign_size);
                break;
            default:
                break;
        }

        if(error != 0){
            return 0; // we don't want to kill the execution due to an error during the creation. BUT we signal out that there was an error.
        }
    }

    // for(size_t i = 0; i < non_collisions; i++){
    //     size_t pos = find_place(blocked, HSIZE, noise(i, seed + 2) % HSIZE, 1);
    //     entry_place(member_count, blocked, HSIZE, pos, elements_to_place);
    // }


    //generate DATA HERE:

    return seed;
}



size_t p1_parameter_gen_max_collision_group(size_t distinct_value, size_t HSIZE){
    // std::cout << "col_group\n";
    return HSIZE/2;
}

size_t p1_parameter_gen_max_collisions(size_t distinct_value, size_t HSIZE, size_t collision_count){
    // std::cout << "cols\n";

    if(collision_count == 0){
        return 0;
    }

    size_t r =  (HSIZE/collision_count) - 1;
    
    if(r > HSIZE){
        return 0;
    }
    return r;
}

size_t p1_parameter_gen_max_cluster(size_t distinct_value, size_t HSIZE, size_t collision_count, size_t collisions){
    // std::cout << "clust\n";
    if(collisions == 0){
        collision_count = 0;
    }

    if(collision_count == 0){
        return HSIZE/2;
    }
    int64_t x = distinct_value - collision_count * collisions;
    if(x <= 0){
        return 0;
    }

    size_t left = HSIZE - collision_count * (collisions + 1);
    return left/2 + collision_count;
}

size_t p1_parameter_gen_max_cluster_length(size_t distinct_value, size_t HSIZE, size_t collision_count, size_t collisions, size_t cluster){
    // std::cout << "clust_len\n";
    // std::cout << distinct_value << "\t" << HSIZE << "\t" << collision_count << "\t" << collisions << "\t" << cluster << "\n";
    // 5 6 cc 1 cs 5 clu 1
    if(cluster == 0){
        return 0;
    }
    if(collisions == 0){
        collision_count = 0;
    }
    if(collision_count == 0){
        size_t r =  (HSIZE / cluster) - 1;
        if(r > distinct_value){
            // std::cout << "r0\n";
            return 0;
        }
        // std::cout << "r1\n";
        return r;
    }

    size_t x = (HSIZE / cluster) - 1;
    if(x > HSIZE){ 
        x = 0;
    }

    if(x > collisions){
        // for p1 the cluster can extend a collision groups size so that it achieves the necessary cluster length.
        // std::cout << "r2\n";
        return x;
    }

    size_t remaining_clusters = cluster - collision_count;
    if(remaining_clusters > cluster || remaining_clusters == 0){
        // std::cout << "r3\n";
        return 0;
    }

    size_t left = HSIZE - collision_count * (collisions + 1);
    // std::cout << "\t" << left << "\t" << remaining_clusters << "\t" << left/remaining_clusters - 1 << std::endl;
    // std::cout << "r4\n";
    return left/remaining_clusters - 1;
}


size_t p1_parameter_gen_distinct(size_t collision_count, size_t collisions, size_t cluster, size_t cluster_lenght){
    // std::cout << "distinct_calc\n";
    size_t a = collision_count;
    size_t a_l = collisions;
    size_t b = cluster;
    size_t b_l = cluster_lenght;

    if(a == 0){
        a_l = 0;
    }
    if(b == 0){
        b_l = 0;
    }

    size_t c = a < b ? a : b;

    a = a - c;
    b = b - c;

    size_t m = a_l > b_l ? a_l : b_l;

    size_t res = c * m + a * a_l + b * b_l;
    return res;
}

size_t p1_parameter_gen_hsize(size_t collision_count, size_t collisions = 0, size_t cluster = 0, size_t cluster_lenght = 0){
    // std::cout << "hsize_calc\n";
    size_t a = collision_count;
    size_t a_l = collisions;
    size_t b = cluster;
    size_t b_l = cluster_lenght;

    size_t c = a < b ? a : b;

    if(a == 0){
        a_l = 0;
    }
    if(b == 0){
        b_l = 0;
    }

    a = a - c;
    b = b - c;

    a_l += 1;
    b_l += 1;

    size_t m = a_l > b_l ? a_l : b_l;

    size_t res = c * m + a * a_l + b * b_l;
    return res;
}

size_t p0_parameter_gen_hsize(size_t collision_count, size_t collisions = 0){
    return collision_count * collisions;
}

std::string* p1_stringify_number(size_t max_val, size_t val){
    val--;
    max_val--;
    size_t m = max_val;
    size_t c = 0;
    while(m > 0){
        m /= 26;
        c++;
    }
    std::stringstream result;
    bool first = true;
    for(size_t i = 0; i < c; i++){
        if(i==0){
            result << (char)('A' + (val%26));
        }else{
            result << (char)('a' + (val%26));
        }
            val /= 26;
    }
    return new std::string(result.str());
}


std::string* p1_stringify( size_t HSIZE, size_t collision_count, size_t collisions, size_t cluster, size_t cluster_lenght){
    int64_t table [HSIZE];
    int64_t col_a = 0;
    int64_t clu_a = 0;
    size_t h_val = 1;
    size_t pos = 0;


    if(p1_parameter_gen_hsize(collision_count, collisions, cluster, cluster_lenght) > HSIZE){
        // *res = new std::string("");
        // return;
        return new std::string("");
    }

    for(size_t i = 0; i < HSIZE; i++){
        table[i] = 0;
    }

    for(; col_a < collision_count; col_a++, clu_a++){
        int64_t i = 0;
        for( ; i < collisions; i++){
            table[i + pos] = h_val;
        }
        pos += collisions;
        h_val++;
        for(; i < cluster_lenght && clu_a < cluster;  i++){
            table[pos] = h_val;
            pos++;
            h_val++;
        }
        pos++;
    }

    for(; clu_a < cluster; clu_a++){
        for(int64_t i = 0; i < cluster_lenght && clu_a < cluster;  i++){
            table[pos] = h_val;
            pos++;
            h_val++;
        }
        pos++;
    }
    // std::stringstream result;
    // result("");
    std::stringstream result;
    result << h_val << ":";
    for(size_t i = 0; i < HSIZE; i++){
        if(table[i] == 0){
            if((i+1) < HSIZE && table[i+1] != 0){
                result << "_";
            }else{
                break;
            }
        }else{
            std::string *helper;
            helper = p1_stringify_number(h_val, table[i]) ;
            result << *helper;
            delete helper; 
        }
        // if(i+1 < HSIZE){
        //     result << " ";
        // }
    }
    return new std::string(result.str());
}

//Data Generator that can create collision_groups FOR 512 bit vectors
// this works only really iff the colision_size is bigger then the vectorsize otherwise we don't really have an overflow.
template<typename T>
size_t generate_data_p1_SoAoV(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t HSIZE,
    size_t (*hash_function)(T, size_t),
    size_t collision_count = 0,
    size_t collision_size = 0,
    size_t cluster_count = 0,
    size_t cluster_size = 0,
    size_t seed = 0
){
    const size_t elements = (512 / 8) / sizeof(T);
    const size_t soaov_hsize = (HSIZE + elements - 1) / elements;

    size_t reserved_free = 0;//cluster_count > collision_count ? cluster_count: collision_count;
    size_t total_free = soaov_hsize * elements - distinct_values;
    size_t distributed_free = total_free <= reserved_free ? 1 : total_free - reserved_free;

    size_t mul = collision_size < 50? collision_size + 2 : 51;    // TODO: maybe use instead of plain 100 -> log(collision_size) * 50
    size_t number_of_values = soaov_hsize * elements * mul;
    
    std::multimap<size_t, T> all_numbers;
    std::vector<T> numbers;

    generate_random_values(all_numbers, hash_function, number_of_values, soaov_hsize, seed, elements * 3 + 1);

    size_t cluster_after_collition_length = cluster_size > collision_size ? cluster_size - collision_size : 0;
    size_t h_pos = noise(HSIZE * distinct_values, seed + 1) % soaov_hsize;
    size_t e_pos = 0;
    

    size_t i = 0;
    bool CREATE_CLUSTER = i < cluster_count;
    bool CREATE_COLLISION = i < collision_count;
    size_t remaining_cluster_length;

    while(CREATE_CLUSTER || CREATE_COLLISION){
        remaining_cluster_length = cluster_size;
        if(e_pos != 0 && distributed_free + e_pos >= elements && CREATE_COLLISION){
            size_t take = elements - e_pos;
            distributed_free -= take;
            e_pos = 0;
            h_pos++;
        }
        if(CREATE_COLLISION){
            generate_collision_soaov<T>(&numbers, &all_numbers, soaov_hsize, h_pos, e_pos, elements, collision_size);
            remaining_cluster_length = cluster_after_collition_length;
        }
        if(CREATE_CLUSTER){
            generate_cluster_soaov<T>(&numbers, &all_numbers, soaov_hsize, h_pos, e_pos, elements, remaining_cluster_length);
        }
        i++;
        CREATE_COLLISION = i < collision_count;
        CREATE_CLUSTER = i < cluster_count;
        
        if(e_pos != 0 && distributed_free > 0){
            distributed_free--;
            e_pos = (e_pos + 1) % elements;
            h_pos = (h_pos + (e_pos == 0)) % soaov_hsize;
        }
    }


    if(numbers.size() == 0){
        // std::cout << "NO DATA GENERATED!\n";
        // throw std::runtime_error("no data generated");    
        return 0;
    }
    generate_benchmark_data<T>(result, data_size, &numbers, seed+3);

    return numbers.size();
}

template<typename T>
size_t generate_data_p1(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t HSIZE,
    size_t (*hash_function)(T, size_t),
    size_t collision_count = 0,
    size_t collision_size = 0,
    size_t cluster_count = 0,
    size_t cluster_size = 0,
    size_t seed = 0,
    bool SoAoV = false
){

    size_t expected_hsize = p1_parameter_gen_hsize(collision_count, collision_size, cluster_count, cluster_size);
    if(expected_hsize > HSIZE || expected_hsize == 0){
        return 0; // HSIZE is to small for the given configuration to fit.
    }
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    }

    if(SoAoV){
        return generate_data_p1_SoAoV(result, data_size, distinct_values, HSIZE, hash_function, collision_count, collision_size, cluster_count, cluster_size, seed);
    }

    size_t total_free = HSIZE - distinct_values;
    size_t reserved_free = cluster_count> collision_count ? cluster_count: collision_count;
    reserved_free += 1;
    size_t distributed_free = total_free <= reserved_free ? 1 : total_free - reserved_free ;

    size_t mul = (collision_size + 1) * 2;
    if(mul > 30){
        mul = 30;
    }

    size_t number_of_values = (HSIZE + 1) * mul;
    
    size_t cluster_after_collition_length = cluster_size > collision_size ? cluster_size - collision_size : 0;
    size_t pos = noise(HSIZE * distinct_values, seed + 1) % HSIZE;

    std::multimap<size_t, T> all_numbers;
    std::vector<T> numbers;

    generate_random_values(all_numbers, hash_function, number_of_values, HSIZE, seed);

    // for(size_t i = 0; i < HSIZE; i++){
    //     std::cout << i << " -> " << all_numbers.count(i) << "\t\t";
    // }
    // std::cout << std::endl;

    size_t i = 0;
    bool CREATE_CLUSTER = i < cluster_count;
    bool CREATE_COLLISION = i < collision_count;
    size_t remaining_cluster_length;

    while(CREATE_CLUSTER || CREATE_COLLISION){
        remaining_cluster_length = cluster_size;
        if(CREATE_COLLISION){
            generate_collision<T>(&numbers, &all_numbers, HSIZE, pos, collision_size);
            pos = (pos + collision_size) % HSIZE;
            remaining_cluster_length = cluster_after_collition_length;
        }
        
        if(CREATE_CLUSTER){
            generate_cluster<T>(&numbers, &all_numbers, HSIZE, pos, remaining_cluster_length);
            pos = (pos + remaining_cluster_length) %HSIZE;
        }

        next_position(pos, distributed_free, HSIZE, seed + 2);

        i++;
        CREATE_COLLISION = i < collision_count;
        CREATE_CLUSTER = i < cluster_count;
    }

    // for(i = 0; i < numbers.size(); i++){
    //     std::cout << hash_function(numbers[i], HSIZE) << ":" << numbers[i] << "\t";
    //     if((i+1) % 20 == 0){
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    if(numbers.size() == 0){
        return 0;
    }
    // std::cout << std::endl;
    generate_benchmark_data<T>(result, data_size, &numbers, seed+3);    
    return numbers.size();
}



template<typename T>
size_t generate_data_p0(
    T*& result,
    size_t data_size,
    size_t distinct_values,
    size_t (*hash_function)(T, size_t),
    size_t collision_count = 0,
    size_t collision_size = 0,
    size_t seed = 0
){
    size_t expected_hsize = p0_parameter_gen_hsize(collision_count, collision_size);
    if(expected_hsize > distinct_values){
        return 0; // HSIZE is to small for the given configuration to fit.
    }
    if(seed == 0){
        srand(std::time(nullptr));
        seed = std::rand();
    }

    size_t free_space = distinct_values - expected_hsize;
    
    size_t mul = (collision_size + 1) * 2;
    if(mul > 30){
        mul = 30;
    }
    size_t number_of_values = (distinct_values + 1) * (mul + 2);
    size_t pos = noise(distinct_values * distinct_values, seed + 1) % distinct_values;

    std::multimap<size_t, T> all_numbers;
    std::vector<T> numbers;

    generate_random_values(all_numbers, hash_function, number_of_values, distinct_values, seed);

    // for(size_t i = 0; i < distinct_values; i++){
    //     std::cout << i << " -> " << all_numbers.count(i) << "\t\t";
    // }
    // std::cout << std::endl;

    size_t i = 0;
    bool CREATE_COLLISION = i < collision_count;
        
    while(CREATE_COLLISION){
        generate_collision<T>(&numbers, &all_numbers, distinct_values, pos, collision_size);
        pos = (pos + collision_size) % distinct_values;
        
        i++;
        CREATE_COLLISION = i < collision_count;
        // CREATE_CLUSTER = i < cluster_count;
    }

    generate_cluster<T>(&numbers, &all_numbers, distinct_values, pos, free_space);
    pos = (pos + free_space) % distinct_values;

    // for(i = 0; i < numbers.size(); i++){
    //     std::cout << hash_function(numbers[i], distinct_values) << ":" << numbers[i] << "\t";
    //     if((i+1) % 20 == 0){
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl << std::endl;

    if(numbers.size() == 0 || numbers.size() > distinct_values){
        return 0;
    }
    generate_benchmark_data<T>(result, data_size, &numbers, seed+3);    
    return numbers.size();
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























#endif //TUD_HASHING_TESTING_DATAGEN