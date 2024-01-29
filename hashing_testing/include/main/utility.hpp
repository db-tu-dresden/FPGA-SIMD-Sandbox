#ifndef TUD_HASHING_TESTING_AVX512_MAIN_UTILITY
#define TUD_HASHING_TESTING_AVX512_MAIN_UTILITY

#include <chrono>
#include <iostream>
#include <cstdint>

using time_stamp = std::chrono::high_resolution_clock::time_point;
std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

uint64_t duration_time_seconds (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
}

void status_output(size_t &runs_done, const size_t total_runs, double &percentage_done, const double percentage_print, std::chrono::high_resolution_clock::time_point time_begin){
    runs_done++;
    std::chrono::high_resolution_clock::time_point time_end;
    double current_percentage = (runs_done * 100.) / total_runs;
    if(current_percentage > percentage_done + percentage_print){
        while(current_percentage > percentage_done + percentage_print){
            percentage_done += percentage_print;
        }

        time_end = time_now();
        size_t meta_time = duration_time(time_begin, time_end);
        size_t meta_time_sec = (size_t)(meta_time / 1000000000.0 + 0.5);
        double work_done = (runs_done * 1. / total_runs);

        size_t meta_time_min = (size_t)(meta_time_sec / 60.0 + 0.5);
        size_t meta_time_left = (size_t)(meta_time_sec / work_done * (1 - work_done));
        if(meta_time_sec < 60){
            std::cout << "\t" <<((int32_t)(1000 * percentage_done))/1000. << "%\tit took ~" << meta_time_sec << " sec. Approx time left:\t" ;
        }else{
            std::cout << "\t" << ((int32_t)(1000 * percentage_done))/1000. << "%\tit took ~" << meta_time_min << " min. Approx time left:\t" ;
        }
        if(meta_time_left < 60){
            std::cout << meta_time_left << " sec" << std::endl;
        }else{
            meta_time_left = (size_t)(meta_time_left / 60.0 + 0.5);
            std::cout << meta_time_left << " min" << std::endl;
        }
    }
}


void print_time(size_t time_sec, bool new_line = true){
    size_t time_min = time_sec/60;
    time_sec -= time_min * 60;
    size_t time_hour = time_min/60;
    time_min -= time_hour * 60;
    if(time_hour < 10){
        std::cout << 0;
    }
    std::cout << time_hour << ":";
    if(time_min < 10){
        std::cout << 0;
    }
    std::cout << time_min << ":";
    if(time_sec < 10){
        std::cout << 0;
    }
    std::cout << time_sec;
    if(new_line){
        std::cout << std::endl;
    }
}

void print_time(std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end, bool new_line = true){
    uint64_t duration = duration_time_seconds(begin, end);
    print_time(duration, new_line);
}

void print_time(std::chrono::high_resolution_clock::time_point begin, bool new_line = true){
    std::chrono::high_resolution_clock::time_point end = time_now();
    print_time(begin, end, new_line);
}


void status_output(size_t runs_done, const size_t total_runs, const double percentage_print, std::chrono::high_resolution_clock::time_point time_begin, bool force = false){
    std::chrono::high_resolution_clock::time_point time_end;

    size_t runs_alt = runs_done -1;
    double p = (total_runs / 100.) * percentage_print;
    size_t val_c = runs_done / p;
    size_t val_a = runs_alt / p;

    if(val_c != val_a || force){
        double percent_done = runs_done * 1.0 / total_runs;
        time_end = time_now();
        size_t time = duration_time(time_begin, time_end);
        size_t time_sec = (size_t)(time / 1000000000.0 + 0.5);
        double time_per_percent = time_sec / percent_done;
        size_t time_left_sec = (1 - percent_done) * time_per_percent;

        std::cout << "\t" << (uint32_t)(percent_done * 10000)/100. << "%\tit took: \t";
        print_time(time_sec, false);
        std::cout << "\tApprox time left:\t";
        print_time(time_left_sec, false);
        std::cout << std::endl;
    }
}


// Different Vector Extention. SSE(128 Bit Vector), AVX2(256 Bit Vector), AVX512(512 Bit Vector)
enum Vector_Extention{
    SCALAR,
    SSE,
    AVX2,
    AVX512
};

// we don't need not unsigned integers, because the implementation are the same.
enum Base_Datatype{
    UI8,
    UI16,
    UI32,
    UI64
};

std::string vector_extention_to_string(Vector_Extention x){
    switch(x){
        case Vector_Extention::SCALAR:
            return "scalar";
        case Vector_Extention::SSE:
            return "sse";
        case Vector_Extention::AVX2:
            return "avx2";
        case Vector_Extention::AVX512:
            return "avx512";
    }
    std::stringstream error_stream;
    error_stream << "Unknown Vector_Extention Enum option. Please add it to vector_extention_to_string(...)\n";
    throw std::runtime_error(error_stream.str());
}

std::string base_datatype_to_string(Base_Datatype x){
    switch(x){
        case Base_Datatype::UI8:
            return "uint8";
        case Base_Datatype::UI16:
            return "uint16";
        case Base_Datatype::UI32:
            return "uint32";
        case Base_Datatype::UI64:
            return "uint64";
    }
    std::stringstream error_stream;
    error_stream << "Unknown Base_Datatype Enum option. Please add it to base_datatype_to_string(...)\n";
    throw std::runtime_error(error_stream.str());
}

#endif //TUD_HASHING_TESTING_AVX512_MAIN_UTILITY