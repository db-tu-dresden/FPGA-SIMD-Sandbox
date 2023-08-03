#include <stdexcept>

#include "operator/physical/group_count/group_count_handler/group_count_algorithms.hpp"

template <typename T>
void getGroupCount(Group_count<T> *& run, Group_Count_Algorithm test, size_t HSIZE, size_t (*function)(T, size_t)){
    if(run != nullptr){
        delete run;
        run = nullptr;
    }

    switch(test){
        case Group_Count_Algorithm::SCALAR_GROUP_COUNT_SOA:
            run = new Scalar_gc_SoA<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS:
            run = new Scalar_gc_AoS<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::SCALAR_GROUP_COUNT_AOS_V2:
            run = new Scalar_gc_AoS_V2<T>(HSIZE, function);
            break;
        // case Group_Count_Algorithm::CHAINED:
        //     run = new Chained<T>(HSIZE, function);
        //     break;
        // case Group_Count_Algorithm::CHAINED2:
        //     run = new Chained2<T>(HSIZE, function);
        //     break;
#ifdef USE_AVX512
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V1:
            run = new AVX512_gc_SoA_v1<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V2:
            run = new AVX512_gc_SoA_v2<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_V3:
            run = new AVX512_gc_SoA_v3<T>(HSIZE, function);
            break;        
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_V1:
            run = new AVX512_gc_AoS_v1<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V1:
            run = new AVX512_gc_SoAoV_v1<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V1:
            run = new AVX512_gc_AoSoV_v1<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOAOV_V2:
            run = new AVX512_gc_SoAoV_v2<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V2:
            run = new AVX512_gc_AoSoV_v2<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V3:
            run = new AVX512_gc_AoSoV_v3<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_AOSOV_V4:
            run = new AVX512_gc_AoSoV_v4<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V1:
            run = new AVX512_gc_SoA_conflict_v1<T>(HSIZE, function);
            break;
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_SOA_CONFLICT_V2:
            run = new AVX512_gc_SoA_conflict_v2<T>(HSIZE, function);
            break;        
        case Group_Count_Algorithm::AVX512_GROUP_COUNT_AOS_CONFLICT_V1:
            run = new AVX512_gc_AoS_conflict_v1<T>(HSIZE, function);
            break;
#endif
        default:
            throw std::runtime_error("The Algorithms isn't supported!");
    }
}


template void getGroupCount<>(Group_count<uint64_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint64_t, size_t));
template void getGroupCount<>(Group_count<uint32_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint32_t, size_t));
// template void getGroupCount<>(Group_count<uint16_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint16_t, size_t));
// template void getGroupCount<>(Group_count<uint8_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint8_t, size_t));
// template void getGroupCount<>(Group_count<int64_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int64_t, size_t));
// template void getGroupCount<>(Group_count<int32_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int32_t, size_t));
// template void getGroupCount<>(Group_count<int16_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int16_t, size_t));
// template void getGroupCount<>(Group_count<int8_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int8_t, size_t));
