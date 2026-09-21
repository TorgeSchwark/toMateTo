// Compiled WITH vector-extension flags (-mavx2 -mavx512f -mavx512bw -mavx512vnni).
#include "../include/nnue/simd.hpp"
#include "simd_wrappers.hpp"

int32_t dot_i8_FAST(const uint8_t* input, const int8_t* weight, std::size_t n) {
    return nnue::simd::dot_i8(input, weight, n);
}
void add_row_i8_to_i16_FAST(int16_t* acc, const int8_t* row, std::size_t n) {
    nnue::simd::add_row_i8_to_i16(acc, row, n);
}
void sub_row_i8_from_i16_FAST(int16_t* acc, const int8_t* row, std::size_t n) {
    nnue::simd::sub_row_i8_from_i16(acc, row, n);
}
void crelu_i16_to_u8_FAST(const int16_t* in, uint8_t* out, std::size_t n, int shift) {
    nnue::simd::crelu_i16_to_u8(in, out, n, shift);
}
void crelu_i32_to_u8_FAST(const int32_t* in, uint8_t* out, std::size_t n, int shift) {
    nnue::simd::crelu_i32_to_u8(in, out, n, shift);
}
