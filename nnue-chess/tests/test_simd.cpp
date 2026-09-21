// Correctness self-test: compares the AVX2/AVX512 kernels in simd.hpp
// (linked from simd_fast.cpp, compiled with vector-extension flags) against
// the scalar reference (linked from simd_scalar.cpp, compiled without them)
// on randomized inputs.
//
// Deliberately uses only <cstdio>/<cstdint>/<cstddef> (no <vector>/<random>/
// <algorithm>) plus a tiny hand-rolled LCG, to stay independent of the
// C++ standard library implementation.
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include "simd_wrappers.hpp"

namespace {

uint32_t g_rng_state = 0xC0FFEEu;
uint32_t next_rand() {
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return g_rng_state;
}
int randint(int lo, int hi) { // inclusive
    uint32_t span = uint32_t(hi - lo + 1);
    return lo + int(next_rand() % span);
}

constexpr std::size_t MAXN = 512;
int g_failures = 0;

void check(const char* name, bool ok) {
    if (!ok) { std::printf("FAIL: %s\n", name); ++g_failures; }
}

} // namespace

int main() {
    // dot_i8
    for (int trial = 0; trial < 200; ++trial) {
        std::size_t n = 32 + (trial % 5) * 32 + (trial % 7);
        uint8_t in[MAXN]; int8_t w[MAXN];
        for (std::size_t i = 0; i < n; ++i) { in[i] = uint8_t(randint(0, 127)); w[i] = int8_t(randint(-128, 127)); }
        int32_t a = dot_i8_FAST(in, w, n);
        int32_t b = dot_i8_SCALAR(in, w, n);
        check("dot_i8", a == b);
    }

    // add_row_i8_to_i16 / sub_row_i8_from_i16
    for (int trial = 0; trial < 200; ++trial) {
        std::size_t n = 16 + (trial % 5) * 16 + (trial % 3);
        int16_t acc1[MAXN], acc2[MAXN]; int8_t row[MAXN];
        for (std::size_t i = 0; i < n; ++i) {
            acc1[i] = int16_t(randint(-20000, 20000));
            acc2[i] = acc1[i];
            row[i] = int8_t(randint(-128, 127));
        }
        add_row_i8_to_i16_FAST(acc1, row, n);
        add_row_i8_to_i16_SCALAR(acc2, row, n);
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) if (acc1[i] != acc2[i]) ok = false;
        check("add_row_i8_to_i16", ok);

        sub_row_i8_from_i16_FAST(acc1, row, n);
        sub_row_i8_from_i16_SCALAR(acc2, row, n);
        ok = true;
        for (std::size_t i = 0; i < n; ++i) if (acc1[i] != acc2[i]) ok = false;
        check("sub_row_i8_from_i16", ok);
    }

    // crelu_i16_to_u8
    for (int trial = 0; trial < 200; ++trial) {
        std::size_t n = 32 + (trial % 4) * 32 + (trial % 5);
        int shift = trial % 7;
        int16_t in[MAXN]; uint8_t a[MAXN], b[MAXN];
        for (std::size_t i = 0; i < n; ++i) in[i] = int16_t(randint(-20000, 20000));
        crelu_i16_to_u8_FAST(in, a, n, shift);
        crelu_i16_to_u8_SCALAR(in, b, n, shift);
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) if (a[i] != b[i]) ok = false;
        check("crelu_i16_to_u8", ok);
    }

    // crelu_i32_to_u8
    for (int trial = 0; trial < 200; ++trial) {
        std::size_t n = 32 + (trial % 4) * 32 + (trial % 5);
        int shift = trial % 9;
        int32_t in[MAXN]; uint8_t a[MAXN], b[MAXN];
        for (std::size_t i = 0; i < n; ++i) in[i] = randint(-2000000, 2000000);
        crelu_i32_to_u8_FAST(in, a, n, shift);
        crelu_i32_to_u8_SCALAR(in, b, n, shift);
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) if (a[i] != b[i]) ok = false;
        if (!ok) {
            std::printf("  n=%zu shift=%d\n", n, shift);
            for (std::size_t i = 0; i < n; ++i)
                if (a[i] != b[i]) std::printf("   i=%zu fast=%d slow=%d in=%d\n", i, a[i], b[i], in[i]);
        }
        check("crelu_i32_to_u8", ok);
    }

    std::printf("%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT", g_failures);
    return g_failures == 0 ? 0 : 1;
}
