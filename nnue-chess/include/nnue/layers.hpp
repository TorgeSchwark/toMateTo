#pragma once
// Dense int8 affine layers used for everything after the sparse feature
// transformer. These layers are small (tens of neurons), so unlike the
// feature transformer they are simply evaluated as a full matrix multiply
// on every call - no incremental update needed or possible here.

#include <cstdint>
#include <cstddef>
#include <array>
#include "simd.hpp"

namespace nnue {

// weights[o][i] is int8, biases[o] is int32 (wide enough to hold IN * 127 *
// 127 without overflow for any IN this architecture uses).
// `forward` expects `input` to already be uint8 in [0,127] - i.e. the
// output of a ClippedReLU - and writes raw (pre-activation) int32 sums to
// `output`; call clipped_relu<OUT>() on that before passing it to the next
// layer, except on the final (score) layer.
template <std::size_t IN, std::size_t OUT>
struct AffineLayer {
    alignas(64) std::array<std::array<int8_t, IN>, OUT> weights{};
    alignas(64) std::array<int32_t, OUT> biases{};

    void forward(const uint8_t* input, int32_t* output) const {
        for (std::size_t o = 0; o < OUT; ++o) {
            output[o] = biases[o] + simd::dot_i8(input, weights[o].data(), IN);
        }
    }
};

// ClippedReLU + requantization: clamp(x >> shift, 0, 127), narrowing back
// down to uint8 for the next layer's input. `shift` is the fixed-point
// scale factor chosen when the network was quantized (see
// training/train_network.hpp); it must match what training used or the
// int8 network will not reproduce the float network's behaviour.
template <std::size_t N>
inline void clipped_relu(const int32_t* in, uint8_t* out, int shift) {
    simd::crelu_i32_to_u8(in, out, N, shift);
}

// Same, but for the feature transformer's int16 accumulator output
// (the very first activation in the network).
template <std::size_t N>
inline void clipped_relu(const int16_t* in, uint8_t* out, int shift) {
    simd::crelu_i16_to_u8(in, out, N, shift);
}

} // namespace nnue
