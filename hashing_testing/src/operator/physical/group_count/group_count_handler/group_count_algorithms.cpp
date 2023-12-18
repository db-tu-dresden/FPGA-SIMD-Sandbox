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
            throw std::runtime_error("The Algorithm isn't supported!");
    }
}

// template void getGroupCount<>(Group_count<uint64_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint64_t, size_t));
template void getGroupCount<>(Group_count<uint32_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint32_t, size_t));
// template void getGroupCount<>(Group_count<uint16_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint16_t, size_t));
// template void getGroupCount<>(Group_count<uint8_t> *& , Group_Count_Algorithm , size_t , size_t (*)(uint8_t, size_t));
// template void getGroupCount<>(Group_count<int64_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int64_t, size_t));
// template void getGroupCount<>(Group_count<int32_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int32_t, size_t));
// template void getGroupCount<>(Group_count<int16_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int16_t, size_t));
// template void getGroupCount<>(Group_count<int8_t> *& , Group_Count_Algorithm , size_t , size_t (*)(int8_t, size_t));



template<class SimdT, typename T>
void getTSLGroupCount(Group_Count_TSL_SOA<T> *& run, Group_Count_Algorithm_TSL test, size_t HSIZE, size_t (*function)(T, size_t), size_t numa_node){
    if(run != nullptr){
        delete run;
        run = nullptr;
    }

    switch(test){
        // case Group_Count_Algorithm_TSL::LP_H_SOA:
        //     run = new TSL_gc_LP_H_SoA<SimdT, T>(HSIZE, function, numa_node);
        //     break;
        // case Group_Count_Algorithm_TSL::LP_V_SOA:
        //     run = new TSL_gc_LP_V_SoA<SimdT, T>(HSIZE, function, numa_node);
        //     break;
        case Group_Count_Algorithm_TSL::LCP_SOA:
            run = new TSL_gc_LCP_SoA<SimdT, T>(HSIZE, function, numa_node);
            break;
        default:
            throw std::runtime_error("The Algorithm isn't supported!");
    }
}

template void getTSLGroupCount<tsl::avx512, uint64_t>(Group_Count_TSL_SOA<uint64_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint64_t, size_t), size_t);
template void getTSLGroupCount<tsl::avx2, uint64_t>(Group_Count_TSL_SOA<uint64_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint64_t, size_t), size_t);
template void getTSLGroupCount<tsl::sse, uint64_t>(Group_Count_TSL_SOA<uint64_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint64_t, size_t), size_t);
template void getTSLGroupCount<tsl::scalar, uint64_t>(Group_Count_TSL_SOA<uint64_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint64_t, size_t), size_t);

template void getTSLGroupCount<tsl::avx512, uint32_t>(Group_Count_TSL_SOA<uint32_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint32_t, size_t), size_t);
template void getTSLGroupCount<tsl::avx2, uint32_t>(Group_Count_TSL_SOA<uint32_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint32_t, size_t), size_t);
template void getTSLGroupCount<tsl::sse, uint32_t>(Group_Count_TSL_SOA<uint32_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint32_t, size_t), size_t);
template void getTSLGroupCount<tsl::scalar, uint32_t>(Group_Count_TSL_SOA<uint32_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint32_t, size_t), size_t);

template void getTSLGroupCount<tsl::avx512, uint16_t>(Group_Count_TSL_SOA<uint16_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint16_t, size_t), size_t);
template void getTSLGroupCount<tsl::avx2, uint16_t>(Group_Count_TSL_SOA<uint16_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint16_t, size_t), size_t);
template void getTSLGroupCount<tsl::sse, uint16_t>(Group_Count_TSL_SOA<uint16_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint16_t, size_t), size_t);
template void getTSLGroupCount<tsl::scalar, uint16_t>(Group_Count_TSL_SOA<uint16_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint16_t, size_t), size_t);

template void getTSLGroupCount<tsl::avx512, uint8_t>(Group_Count_TSL_SOA<uint8_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint8_t, size_t), size_t);
template void getTSLGroupCount<tsl::avx2, uint8_t>(Group_Count_TSL_SOA<uint8_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint8_t, size_t), size_t);
template void getTSLGroupCount<tsl::sse, uint8_t>(Group_Count_TSL_SOA<uint8_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint8_t, size_t), size_t);
template void getTSLGroupCount<tsl::scalar, uint8_t>(Group_Count_TSL_SOA<uint8_t> *&, Group_Count_Algorithm_TSL, size_t, size_t(*)(uint8_t, size_t), size_t);

