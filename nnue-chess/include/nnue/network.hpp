#pragma once
// Assembles the full network: sparse feature transformer -> accumulator ->
// three dense int8 hidden layers -> scalar output.
//
// All layer widths are template parameters, so trying a different
// architecture is "change the numbers in the type and recompile", e.g.:
//
//   nnue::NNUE<256, 128, 32, 32> net;   // Stockfish-ish "HalfKP-256x2-32-32"
//   nnue::NNUE<128, 64,  32, 16> small; // a smaller net to experiment with
//
// The accumulator holds one ACC_SIZE-wide vector per perspective; the first
// dense layer sees both concatenated (2 * ACC_SIZE inputs, own perspective
// first), which is the standard NNUE "the net always evaluates from the
// side-to-move's point of view" trick.

#include <cstdint>
#include <cstddef>
#include <array>
#include <fstream>
#include <memory>

#include "halfkp.hpp"
#include "feature_transformer.hpp"
#include "layers.hpp"

namespace nnue {

template <std::size_t ACC_SIZE, std::size_t H1, std::size_t H2, std::size_t H3>
class NNUE {
public:
    FeatureTransformer<ACC_SIZE> feature_transformer;
    AffineLayer<2 * ACC_SIZE, H1> layer1;
    AffineLayer<H1, H2> layer2;
    AffineLayer<H2, H3> layer3;
    AffineLayer<H3, 1> output_layer;

    // Right-shifts that requantize each layer's wider accumulation back
    // down into the uint8 [0,127] range the next layer's int8 weights
    // expect. Fixed by the quantization scheme used when the network was
    // exported (see training/train_network.hpp::quantize_into); stored here
    // (not hardcoded) so a network file is self-describing.
    //
    // Defaults match train_network.hpp's scheme (QA=127 feature-transformer
    // scale, QB=64 hidden-layer scale): the accumulator is already in "x127"
    // units so needs no shift (ft_shift=0), each hidden layer multiplies in
    // another factor of QB=64 that log2(64)=6 undoes (hidden_shift=6).
    int ft_shift = 0;
    int hidden_shift = 6;
    // Final int32 output -> centipawns. Divide, don't shift, since this
    // factor is a training hyperparameter (see training's OUTPUT_SCALE),
    // not necessarily a power of two. Default matches QA*QB for the same
    // scheme, i.e. "the float output layer's raw value is already
    // centipawns" (see train_network.hpp).
    float output_scale = 127.0f * 64.0f;

    // Evaluates the position `acc` currently represents, from
    // `side_to_move`'s point of view (positive = good for side_to_move).
    int32_t evaluate(const Accumulator<ACC_SIZE>& acc, Color side_to_move) const {
        const Color other = static_cast<Color>(static_cast<int>(side_to_move) ^ 1);

        alignas(64) uint8_t input[2 * ACC_SIZE];
        clipped_relu<ACC_SIZE>(acc.values[static_cast<std::size_t>(side_to_move)].data(), input, ft_shift);
        clipped_relu<ACC_SIZE>(acc.values[static_cast<std::size_t>(other)].data(), input + ACC_SIZE, ft_shift);

        alignas(64) int32_t out1[H1];
        alignas(64) uint8_t relu1[H1];
        layer1.forward(input, out1);
        clipped_relu<H1>(out1, relu1, hidden_shift);

        alignas(64) int32_t out2[H2];
        alignas(64) uint8_t relu2[H2];
        layer2.forward(relu1, out2);
        clipped_relu<H2>(out2, relu2, hidden_shift);

        alignas(64) int32_t out3[H3];
        alignas(64) uint8_t relu3[H3];
        layer3.forward(relu2, out3);
        clipped_relu<H3>(out3, relu3, hidden_shift);

        int32_t raw = 0;
        output_layer.forward(relu3, &raw);
        return raw;
    }

    // Convenience: evaluate() rescaled to centipawns as a float.
    float evaluate_cp(const Accumulator<ACC_SIZE>& acc, Color side_to_move) const {
        return static_cast<float>(evaluate(acc, side_to_move)) / output_scale;
    }

    static std::unique_ptr<NNUE> make() { return std::make_unique<NNUE>(); }

    // --- Serialization -----------------------------------------------
    // Plain, self-describing little-endian binary layout: a small header
    // (magic, layer sizes, shifts, output scale) followed by every layer's
    // raw weight/bias bytes in declaration order. Deliberately simple - no
    // versioning/compression - so training/train_network.hpp's exporter and
    // this loader are trivially kept in sync.

    bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        write_header(f);
        write_raw(f, feature_transformer.weights);
        write_raw(f, feature_transformer.biases);
        write_raw(f, layer1.weights);
        write_raw(f, layer1.biases);
        write_raw(f, layer2.weights);
        write_raw(f, layer2.biases);
        write_raw(f, layer3.weights);
        write_raw(f, layer3.biases);
        write_raw(f, output_layer.weights);
        write_raw(f, output_layer.biases);
        return bool(f);
    }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        if (!read_header(f)) return false;
        read_raw(f, feature_transformer.weights);
        read_raw(f, feature_transformer.biases);
        read_raw(f, layer1.weights);
        read_raw(f, layer1.biases);
        read_raw(f, layer2.weights);
        read_raw(f, layer2.biases);
        read_raw(f, layer3.weights);
        read_raw(f, layer3.biases);
        read_raw(f, output_layer.weights);
        read_raw(f, output_layer.biases);
        return bool(f);
    }

private:
    static constexpr uint32_t kMagic = 0x4E4E5545u; // "NNUE"

    void write_header(std::ofstream& f) const {
        uint32_t header[7] = {
            kMagic,
            static_cast<uint32_t>(ACC_SIZE), static_cast<uint32_t>(H1),
            static_cast<uint32_t>(H2), static_cast<uint32_t>(H3),
            static_cast<uint32_t>(ft_shift), static_cast<uint32_t>(hidden_shift)
        };
        f.write(reinterpret_cast<const char*>(header), sizeof(header));
        f.write(reinterpret_cast<const char*>(&output_scale), sizeof(output_scale));
    }

    bool read_header(std::ifstream& f) {
        uint32_t header[7];
        f.read(reinterpret_cast<char*>(header), sizeof(header));
        if (!f) return false;
        if (header[0] != kMagic) return false;
        if (header[1] != ACC_SIZE || header[2] != H1 || header[3] != H2 || header[4] != H3) {
            return false; // file was exported for a different architecture
        }
        ft_shift = static_cast<int>(header[5]);
        hidden_shift = static_cast<int>(header[6]);
        f.read(reinterpret_cast<char*>(&output_scale), sizeof(output_scale));
        return bool(f);
    }

    template <typename T>
    static void write_raw(std::ofstream& f, const T& data) {
        f.write(reinterpret_cast<const char*>(&data), sizeof(T));
    }
    template <typename T>
    static void read_raw(std::ifstream& f, T& data) {
        f.read(reinterpret_cast<char*>(&data), sizeof(T));
    }
};

} // namespace nnue
