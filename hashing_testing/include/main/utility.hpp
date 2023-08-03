#ifndef TUD_HASHING_TESTING_AVX512_MAIN_UTILITY
#define TUD_HASHING_TESTING_AVX512_MAIN_UTILITY

#include <chrono>
#include <iostream>

std::chrono::high_resolution_clock::time_point time_now(){
    return std::chrono::high_resolution_clock::now();
}

uint64_t duration_time (std::chrono::high_resolution_clock::time_point begin, std::chrono::high_resolution_clock::time_point end){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
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


void print_time(size_t time_sec){
    size_t time_min = time_sec/60;
    time_sec -= time_min * 60;
    size_t time_hour = time_min/60;
    time_min -= time_hour * 60;
    std::cout << time_hour << ":" << time_min << ":" << time_sec;
}



void status_output(size_t runs_done, const size_t total_runs, const double percentage_print, std::chrono::high_resolution_clock::time_point time_begin, bool force = false){
    std::chrono::high_resolution_clock::time_point time_end;

    size_t runs_alt = runs_done -1;
    double p = (total_runs / 100.) * percentage_print;
    size_t val_c = runs_done / p;
    size_t val_a = runs_alt /p;
    if(val_c != val_a || force){
        double percent_done = runs_done * 1.0 / total_runs;
        time_end = time_now();
        size_t time = duration_time(time_begin, time_end);
        size_t time_sec = (size_t)(time / 1000000000.0 + 0.5);
        double time_per_percent = time_sec / percent_done;
        size_t time_left_sec = (1 - percent_done) * time_per_percent;

        std::cout << "\t" << (uint32_t)(percent_done * 10000)/100. << "%\tit took ~";
        print_time(time_sec);
        std::cout << "\tApprox time left:\t";
        print_time(time_left_sec);
        std::cout << std::endl;
    }
}


#endif //TUD_HASHING_TESTING_AVX512_MAIN_UTILITY