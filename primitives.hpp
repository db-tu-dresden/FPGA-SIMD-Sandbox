#ifndef PRIMITIVES_HPP
#define PRIMITIVES_HPP

template<typename T>
struct fpvec {
    [[intel::fpga_register]] std::array<T, 64/sizeof(T)> elements;
};

template<typename T>
fpvec<T> load(T* p, int i_cnt) {
    auto reg = fpvec<T> {};
    #pragma unroll
    for (uint idx = 0; idx < 16; idx++) {
          reg.elements[idx] = p[idx + i_cnt*16];
    }
    return reg;
}

template<typename T>
fpvec<T> set1(T value) {
  auto reg = fpvec<T> {};
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    reg.elements[i] = value;
  }
  return reg;
}

template<typename T>
fpvec<T> add(fpvec<T>& a, fpvec<T>& b) {
  #pragma unroll
  for (uint idx = 0; idx < 16; idx++) {
          a.elements[idx] += b.elements[idx];
  }
  return a;
}

template<typename T>
T hadd(fpvec<T> a) {
  // Adder tree
  [[intel::fpga_register]] T add_1_1 = a.elements[0] + a.elements[1];
  [[intel::fpga_register]] T add_1_2 = a.elements[2] + a.elements[3];
  [[intel::fpga_register]] T add_1_3 = a.elements[4] + a.elements[5];
  [[intel::fpga_register]] T add_1_4 = a.elements[6] + a.elements[7];
  [[intel::fpga_register]] T add_1_5 = a.elements[8] + a.elements[9];
  [[intel::fpga_register]] T add_1_6 = a.elements[10] + a.elements[11];
  [[intel::fpga_register]] T add_1_7 = a.elements[12] + a.elements[13];
  [[intel::fpga_register]] T add_1_8 = a.elements[14] + a.elements[15];


  [[intel::fpga_register]] T add_2_1 = add_1_1 + add_1_2;
  [[intel::fpga_register]] T add_2_2 = add_1_3 + add_1_4;
  [[intel::fpga_register]] T add_2_3 = add_1_5 + add_1_6;
  [[intel::fpga_register]] T add_2_4 = add_1_7 + add_1_8;

  [[intel::fpga_register]] T add_3_1 = add_2_1 + add_2_2;
  [[intel::fpga_register]] T add_3_2 = add_2_3 + add_2_4;

   return add_3_1 + add_3_2;
}

#endif // PRIMITIVES_HPP