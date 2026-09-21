#pragma once
// Low-level SIMD kernels used by the NNUE evaluator.
//
// Three tiers, selected at compile time by which macros the compiler defines:
//   - AVX512VNNI  (fastest: fused int8 dot product via _mm512_dpbusd_epi32)
//   - AVX2        (baseline: the classic maddubs+madd int8 dot-product trick)
//   - scalar      (portability / correctness reference, always available)
//
// Build with -mavx2 (baseline) and optionally -mavx512f -mavx512bw -mavx512vnni
// (or MSVC /arch:AVX2 / /arch:AVX512) to enable the fast paths. Without any of
// those flags everything still compiles and runs via the scalar fallback, so
// the same source works on non-x86 targets too.
//
// Define NNUE_FORCE_SCALAR before including this header to disable all
// intrinsics paths regardless of compiler flags (useful for correctness
// testing: compare the vectorized result against the scalar reference).

#include <cstdint>
#include <cstddef>

#if !defined(NNUE_FORCE_SCALAR)
  #if defined(__AVX512VNNI__)
    #include <immintrin.h>
    #define NNUE_HAVE_AVX512VNNI 1
  #endif
  #if defined(__AVX2__)
    #include <immintrin.h>
    #define NNUE_HAVE_AVX2 1
  #endif
#endif

namespace nnue::simd {

inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

#if defined(NNUE_HAVE_AVX2)
inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}
#endif

// sum_j input_u8[j] * weight_s8[j]  for j in [0, n)
// `input` must hold values in [0, 127] (guaranteed by ClippedReLU output),
// so it is safe to reinterpret as unsigned - that is what makes the AVX2
// maddubs_epi16 (unsigned x signed -> int16) trick applicable.
inline int32_t dot_i8(const uint8_t* input, const int8_t* weight, std::size_t n) {
#if defined(NNUE_HAVE_AVX512VNNI)
    __m512i acc = _mm512_setzero_si512();
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i in = _mm512_loadu_si512(reinterpret_cast<const void*>(input + i));
        __m512i w  = _mm512_loadu_si512(reinterpret_cast<const void*>(weight + i));
        acc = _mm512_dpbusd_epi32(acc, in, w); // acc += sum(in * w) as 4-wide int8 dot per int32 lane
    }
    int32_t sum = _mm512_reduce_add_epi32(acc);
    for (; i < n; ++i) sum += int32_t(input[i]) * int32_t(weight[i]);
    return sum;
#elif defined(NNUE_HAVE_AVX2)
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
        __m256i w  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + i));
        // unsigned(in) * signed(w), horizontally paired into 16 x int16
        __m256i prod16 = _mm256_maddubs_epi16(in, w);
        // widen the 16 pairs to int32 (madd with all-ones is a widening sum, not a second multiply)
        __m256i prod32 = _mm256_madd_epi16(prod16, ones16);
        acc = _mm256_add_epi32(acc, prod32);
    }
    int32_t sum = hsum_epi32(acc);
    for (; i < n; ++i) sum += int32_t(input[i]) * int32_t(weight[i]);
    return sum;
#else
    int32_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += int32_t(input[i]) * int32_t(weight[i]);
    return sum;
#endif
}

// acc[i] += row[i], row is int8 sign-extended to int16 before adding.
// Used for the sparse feature-transformer's incremental accumulator update.
inline void add_row_i8_to_i16(int16_t* acc, const int8_t* row, std::size_t n) {
#if defined(NNUE_HAVE_AVX2)
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i r8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i));
        __m256i r16 = _mm256_cvtepi8_epi16(r8); // sign-extend 16 x int8 -> 16 x int16
        __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        a16 = _mm256_add_epi16(a16, r16);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), a16);
    }
    for (; i < n; ++i) acc[i] = int16_t(acc[i] + int16_t(row[i]));
#else
    for (std::size_t i = 0; i < n; ++i) acc[i] = int16_t(acc[i] + int16_t(row[i]));
#endif
}

// acc[i] -= row[i]
inline void sub_row_i8_from_i16(int16_t* acc, const int8_t* row, std::size_t n) {
#if defined(NNUE_HAVE_AVX2)
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i r8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i));
        __m256i r16 = _mm256_cvtepi8_epi16(r8);
        __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        a16 = _mm256_sub_epi16(a16, r16);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), a16);
    }
    for (; i < n; ++i) acc[i] = int16_t(acc[i] - int16_t(row[i]));
#else
    for (std::size_t i = 0; i < n; ++i) acc[i] = int16_t(acc[i] - int16_t(row[i]));
#endif
}

// ClippedReLU + requantization: int16 accumulator -> uint8, clamp(x >> shift, 0, 127).
// Feeds the feature-transformer output into the first int8 affine layer.
inline void crelu_i16_to_u8(const int16_t* in, uint8_t* out, std::size_t n, int shift) {
#if defined(NNUE_HAVE_AVX2)
    std::size_t i = 0;
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 16));
        a = _mm256_srai_epi16(a, shift);
        b = _mm256_srai_epi16(b, shift);
        __m256i packed = _mm256_packs_epi16(a, b);          // saturating int16 -> int8, lane-interleaved
        packed = _mm256_permute4x64_epi64(packed, 0xD8);    // undo AVX2's 128-bit lane interleave
        packed = _mm256_max_epi8(packed, zero);              // clamp lower bound to 0
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
    }
    for (; i < n; ++i) {
        int32_t v = int32_t(in[i]) >> shift;
        out[i] = uint8_t(clamp_i32(v, 0, 127));
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        int32_t v = int32_t(in[i]) >> shift;
        out[i] = uint8_t(clamp_i32(v, 0, 127));
    }
#endif
}

// ClippedReLU + requantization: int32 layer output -> uint8, clamp(x >> shift, 0, 127).
// Feeds one hidden affine layer's output into the next.
inline void crelu_i32_to_u8(const int32_t* in, uint8_t* out, std::size_t n, int shift) {
#if defined(NNUE_HAVE_AVX2)
    std::size_t i = 0;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i fixlanes = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8));
        __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 16));
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 24));
        a = _mm256_srai_epi32(a, shift);
        b = _mm256_srai_epi32(b, shift);
        c = _mm256_srai_epi32(c, shift);
        d = _mm256_srai_epi32(d, shift);
        __m256i ab16 = _mm256_packs_epi32(a, b);   // saturating int32 -> int16
        __m256i cd16 = _mm256_packs_epi32(c, d);
        __m256i packed = _mm256_packs_epi16(ab16, cd16); // saturating int16 -> int8
        // two pack stages leave the 32 bytes permuted as 8 dword-groups in
        // {0,2,4,6 (from ab16/cd16 low halves), 1,3,5,7 (high halves)} order;
        // fixlanes restores the original element order.
        packed = _mm256_permutevar8x32_epi32(packed, fixlanes);
        packed = _mm256_max_epi8(packed, zero);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
    }
    for (; i < n; ++i) {
        int32_t v = in[i] >> shift;
        out[i] = uint8_t(clamp_i32(v, 0, 127));
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        int32_t v = in[i] >> shift;
        out[i] = uint8_t(clamp_i32(v, 0, 127));
    }
#endif
}

} // namespace nnue::simd
