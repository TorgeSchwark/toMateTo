// Declarations shared by simd_fast.cpp (compiled with -mavx2 -mavx512...) and
// simd_scalar.cpp (compiled with no vector-extension flags / NNUE_FORCE_SCALAR)
// so test_simd.cpp can link both and compare their outputs directly.
#pragma once
#include <cstdint>
#include <cstddef>

int32_t dot_i8_FAST(const uint8_t* input, const int8_t* weight, std::size_t n);
void add_row_i8_to_i16_FAST(int16_t* acc, const int8_t* row, std::size_t n);
void sub_row_i8_from_i16_FAST(int16_t* acc, const int8_t* row, std::size_t n);
void crelu_i16_to_u8_FAST(const int16_t* in, uint8_t* out, std::size_t n, int shift);
void crelu_i32_to_u8_FAST(const int32_t* in, uint8_t* out, std::size_t n, int shift);

int32_t dot_i8_SCALAR(const uint8_t* input, const int8_t* weight, std::size_t n);
void add_row_i8_to_i16_SCALAR(int16_t* acc, const int8_t* row, std::size_t n);
void sub_row_i8_from_i16_SCALAR(int16_t* acc, const int8_t* row, std::size_t n);
void crelu_i16_to_u8_SCALAR(const int16_t* in, uint8_t* out, std::size_t n, int shift);
void crelu_i32_to_u8_SCALAR(const int32_t* in, uint8_t* out, std::size_t n, int shift);
