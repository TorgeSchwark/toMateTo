#pragma once
// Float-precision training network that mirrors nnue::NNUE<ACC_SIZE,H1,H2,H3>
// layer-for-layer, plus forward/backward passes and a quantize_into()
// exporter that produces the int8/int16 inference network from
// include/nnue/network.hpp.
//
// Quantization scheme (fixed-point convention used throughout):
//   QA = 127  - feature-transformer weight & activation scale
//   QB = 64   - every hidden/output layer's weight scale (a power of two,
//               so requantization is an exact right-shift, not a rounded
//               division)
// Every ClippedReLU output (the feature transformer's, and every hidden
// layer's) is an int in [0,127] representing (float_activation * QA); every
// hidden/output layer's raw pre-bias-add... pre-activation sum lands in the
// (QA*QB) fixed-point domain and is shifted right by log2(QB)=6 bits to get
// back to the QA domain for the next layer. See quantize_into() and
// include/nnue/network.hpp's ft_shift/hidden_shift for the inference side
// of this same scheme.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>

#include "../include/nnue/halfkp.hpp"
#include "../include/nnue/network.hpp"
#include "sparse_optimizer.hpp"

namespace nnue::train {

constexpr float QA = 127.0f;
constexpr float QB = 64.0f;
// By convention the (unscaled) output-layer activation is trained to equal
// centipawns/CP_SCALE, i.e. pred_wdl = sigmoid(output_pre) corresponds to
// the usual chess-engine sigmoid(cp/400) win-probability model.
constexpr float CP_SCALE = 400.0f;

inline float relu01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
// Gradient of clamp(x,0,1) used for backprop: 1 strictly inside the
// interval, but only a small "leak" (not a hard 0) outside it. A hard 0
// outside (0,1) is the textbook clamp derivative, but combined with Adam's
// per-parameter-normalized step size it is an absorbing trap in practice: a
// unit that gets pushed outside (0,1) - easy to happen to many units at
// once early in training, before the input distribution has stabilized -
// then receives exactly zero gradient forever after, with nothing (no
// weight decay on biases, no gradient) to pull it back. This straight-
// through-style leak (forward pass stays a hard clamp - unaffected by this,
// so it still matches the quantized int8 network's behaviour bit-for-bit)
// gives a dying unit a way back into the live region if the loss wants it
// there.
constexpr float kReluLeak = 0.05f;
inline float relu01_grad(float pre) { return (pre > 0.f && pre < 1.f) ? 1.f : kReluLeak; }

// Quantization-aware-training helper: simulates quantize_i8's rounding to
// int8 and back (forward pass only - see below), so training feels the same
// precision loss export will apply instead of finding it out afterwards.
// Without this, plain float SGD/Adam has no reason to prefer a weight that
// survives quantization over one that doesn't: MSE loss can be reduced by
// an arbitrarily tiny weight nudge just as well as by a larger one, so nothing
// stops the optimizer from settling on weights smaller than 1/QA (127) -
// which quantize_i8's std::lround rounds straight to 0, silently discarding
// everything the feature transformer "learned" for that dimension. Applying
// the same rounding during training means a weight that small contributes
// *nothing* to the forward pass either, so the optimizer is pushed to either
// grow it past the rounding threshold (if it actually helps the loss) or
// let weight decay carry it to true 0 (if it doesn't) - never leave it
// stranded in between, invisible in float but silently vaporized at export.
//
// This is a straight-through estimator (STE): the backward pass treats
// fake_quantize as the identity (d(fake_quantize(x))/dx = 1), which is why
// no gradient-computation code needs to change anywhere this is used - the
// existing dL/d(row weight) formulas are already exactly the STE gradient.
inline float fake_quantize(float x, float scale) {
    float q = static_cast<float>(std::lround(x * scale));
    if (q > 127.0f) q = 127.0f;
    if (q < -127.0f) q = -127.0f;
    return q / scale;
}
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct TrainingSample {
    std::vector<int> white_features; // active HalfKP indices, White's perspective
    std::vector<int> black_features; // active HalfKP indices, Black's perspective
    Color side_to_move;
    float target_wdl; // in [0,1]; caller decides how to blend search eval / game result into this
};

// Dense float layer, row-major weight[OUT][IN].
template <std::size_t IN, std::size_t OUT>
struct DenseLayerF {
    std::vector<float> weight;
    std::vector<float> bias;
    DenseLayerF() : weight(OUT * IN), bias(OUT, 0.0f) {}
    float* w(std::size_t o) { return &weight[o * IN]; }
    const float* w(std::size_t o) const { return &weight[o * IN]; }
};

template <std::size_t ACC_SIZE, std::size_t H1, std::size_t H2, std::size_t H3>
class TrainNetwork {
public:
    // --- Parameters -----------------------------------------------------
    LazySparseAdamW<ACC_SIZE> ft_weight_opt; // owns the NUM_FEATURES x ACC_SIZE embedding table
    std::vector<float> ft_bias;              // ACC_SIZE - dense (every sample uses it), no laziness needed
    DenseAdamW ft_bias_opt;

    DenseLayerF<2 * ACC_SIZE, H1> layer1; DenseAdamW layer1_w_opt, layer1_b_opt;
    DenseLayerF<H1, H2> layer2;           DenseAdamW layer2_w_opt, layer2_b_opt;
    DenseLayerF<H2, H3> layer3;           DenseAdamW layer3_w_opt, layer3_b_opt;
    DenseLayerF<H3, 1> output;            DenseAdamW output_w_opt, output_b_opt;

    explicit TrainNetwork(float lr, float ft_weight_decay = 1e-6f, float hidden_weight_decay = 1e-5f)
        : ft_weight_opt(NUM_FEATURES, lr, 0.9f, 0.999f, 1e-8f, ft_weight_decay),
          // Every bias that feeds a ClippedReLU also gets a (small) weight
          // decay, unlike the usual "no decay on biases" convention: it is
          // touched every single step (dense), so over a long run its Adam
          // updates - each individually bounded to roughly +/-lr, but able
          // to compound in one direction for many consecutive steps if the
          // loss signal stays consistently signed early in training - can
          // otherwise drift arbitrarily far outside the activation's [0,1]
          // range and get permanently stuck there (see relu01_grad). Decay
          // gives it a restoring pull back toward 0 that plain "no gradient
          // once saturated" cannot provide on its own. output.bias has no
          // activation after it, so it is exempt (see randomize_dense_weights).
          ft_bias(ACC_SIZE, 0.0f), ft_bias_opt(ACC_SIZE, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          layer1_w_opt(H1 * 2 * ACC_SIZE, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          layer1_b_opt(H1, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          layer2_w_opt(H2 * H1, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          layer2_b_opt(H2, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          layer3_w_opt(H3 * H2, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          layer3_b_opt(H3, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay),
          output_w_opt(1 * H3, lr, 0.9f, 0.999f, 1e-8f, hidden_weight_decay), output_b_opt(1, lr) {
        randomize_dense_weights();
    }

    // --- Forward pass (cached, for backward()) --------------------------
    struct Cache {
        std::vector<float> acc[2];       // pre-activation accumulator, per perspective
        std::vector<float> relu_acc[2];  // post-ClippedReLU accumulator, per perspective
        std::vector<float> input;        // concat(relu_acc[stm], relu_acc[other]) -> layer1 input
        std::vector<float> l1_pre, l1_post;
        std::vector<float> l2_pre, l2_post;
        std::vector<float> l3_pre, l3_post;
        float out_pre = 0.0f;
        float pred_wdl = 0.0f;
        Cache()
            : input(2 * ACC_SIZE), l1_pre(H1), l1_post(H1), l2_pre(H2), l2_post(H2), l3_pre(H3), l3_post(H3) {
            acc[0].assign(ACC_SIZE, 0.0f); acc[1].assign(ACC_SIZE, 0.0f);
            relu_acc[0].assign(ACC_SIZE, 0.0f); relu_acc[1].assign(ACC_SIZE, 0.0f);
        }
    };

    float forward(const TrainingSample& s, Cache& c) const {
        compute_accumulator(s.white_features, WHITE, c);
        compute_accumulator(s.black_features, BLACK, c);

        const std::size_t stm = static_cast<std::size_t>(s.side_to_move);
        const std::size_t other = 1 - stm;
        for (std::size_t i = 0; i < ACC_SIZE; ++i) c.input[i] = c.relu_acc[stm][i];
        for (std::size_t i = 0; i < ACC_SIZE; ++i) c.input[ACC_SIZE + i] = c.relu_acc[other][i];

        dense_forward<2 * ACC_SIZE, H1>(layer1, c.input.data(), c.l1_pre.data());
        for (std::size_t i = 0; i < H1; ++i) c.l1_post[i] = relu01(c.l1_pre[i]);

        dense_forward<H1, H2>(layer2, c.l1_post.data(), c.l2_pre.data());
        for (std::size_t i = 0; i < H2; ++i) c.l2_post[i] = relu01(c.l2_pre[i]);

        dense_forward<H2, H3>(layer3, c.l2_post.data(), c.l3_pre.data());
        for (std::size_t i = 0; i < H3; ++i) c.l3_post[i] = relu01(c.l3_pre[i]);

        float raw = output.bias[0];
        const float* wo = output.w(0);
        for (std::size_t i = 0; i < H3; ++i) raw += wo[i] * c.l3_post[i];
        c.out_pre = raw;
        c.pred_wdl = sigmoid(raw);
        return c.pred_wdl;
    }

    // --- Backward pass + optimizer step, one sample -----------------------
    // MSE loss in win-probability space: L = (pred_wdl - target)^2. Simple,
    // well-behaved, and the standard NNUE training loss's shape (a fancier
    // blend of search-eval-derived and game-result-derived targets just
    // changes what target_wdl *is*, not this function).
    float train_step(const TrainingSample& s) {
        Cache c;
        const float pred = forward(s, c);
        const float loss = (pred - s.target_wdl) * (pred - s.target_wdl);

        const float dL_dpred = 2.0f * (pred - s.target_wdl);
        const float dpred_dpre = pred * (1.0f - pred); // sigmoid'
        const float grad_out_pre = dL_dpred * dpred_dpre;

        // output layer
        std::vector<float> grad_output_w(H3);
        for (std::size_t i = 0; i < H3; ++i) grad_output_w[i] = grad_out_pre * c.l3_post[i];
        float grad_output_b = grad_out_pre;
        std::vector<float> grad_l3_post(H3);
        for (std::size_t i = 0; i < H3; ++i) grad_l3_post[i] = grad_out_pre * output.w(0)[i];

        // layer3
        std::vector<float> grad_l3_pre(H3);
        for (std::size_t i = 0; i < H3; ++i) grad_l3_pre[i] = grad_l3_post[i] * relu01_grad(c.l3_pre[i]);
        std::vector<float> grad_l3_w(H3 * H2), grad_l3_b(H3);
        std::vector<float> grad_l2_post(H2, 0.0f);
        backprop_dense<H2, H3>(layer3, c.l2_post.data(), grad_l3_pre.data(), grad_l3_w.data(), grad_l3_b.data(),
                                grad_l2_post.data());

        // layer2
        std::vector<float> grad_l2_pre(H2);
        for (std::size_t i = 0; i < H2; ++i) grad_l2_pre[i] = grad_l2_post[i] * relu01_grad(c.l2_pre[i]);
        std::vector<float> grad_l2_w(H2 * H1), grad_l2_b(H2);
        std::vector<float> grad_l1_post(H1, 0.0f);
        backprop_dense<H1, H2>(layer2, c.l1_post.data(), grad_l2_pre.data(), grad_l2_w.data(), grad_l2_b.data(),
                                grad_l1_post.data());

        // layer1
        std::vector<float> grad_l1_pre(H1);
        for (std::size_t i = 0; i < H1; ++i) grad_l1_pre[i] = grad_l1_post[i] * relu01_grad(c.l1_pre[i]);
        std::vector<float> grad_l1_w(H1 * 2 * ACC_SIZE), grad_l1_b(H1);
        std::vector<float> grad_input(2 * ACC_SIZE, 0.0f);
        backprop_dense<2 * ACC_SIZE, H1>(layer1, c.input.data(), grad_l1_pre.data(), grad_l1_w.data(),
                                          grad_l1_b.data(), grad_input.data());

        // accumulator / feature transformer
        const std::size_t stm = static_cast<std::size_t>(s.side_to_move);
        const std::size_t other = 1 - stm;
        std::vector<float> grad_acc[2] = {std::vector<float>(ACC_SIZE, 0.0f), std::vector<float>(ACC_SIZE, 0.0f)};
        for (std::size_t i = 0; i < ACC_SIZE; ++i)
            grad_acc[stm][i] = grad_input[i] * relu01_grad(c.acc[stm][i]);
        for (std::size_t i = 0; i < ACC_SIZE; ++i)
            grad_acc[other][i] = grad_input[ACC_SIZE + i] * relu01_grad(c.acc[other][i]);

        std::vector<float> grad_ft_bias(ACC_SIZE);
        for (std::size_t i = 0; i < ACC_SIZE; ++i) grad_ft_bias[i] = grad_acc[0][i] + grad_acc[1][i];

        // acc[p][d] = bias[d] + sum over active features of ft_weight[f][d];
        // every active feature of perspective p gets the SAME gradient
        // vector grad_acc[p]. A given absolute row index could (rarely, by
        // coincidence) be active in both perspectives' lists, so accumulate
        // per-row before calling the optimizer once per distinct row -
        // this is also exactly the "sparse gradient" the optimizer expects.
        std::unordered_map<int, std::vector<float>> row_grads;
        accumulate_row_grads(s.white_features, grad_acc[0], row_grads);
        accumulate_row_grads(s.black_features, grad_acc[1], row_grads);

        // --- apply updates ---
        ft_weight_opt.begin_step();
        for (auto& [feature, grad] : row_grads) {
            ft_weight_opt.apply_gradient(static_cast<std::size_t>(feature), grad.data());
        }
        ft_bias_opt.step(ft_bias.data(), grad_ft_bias.data());

        layer1_w_opt.step(layer1.weight.data(), grad_l1_w.data());
        layer1_b_opt.step(layer1.bias.data(), grad_l1_b.data());
        layer2_w_opt.step(layer2.weight.data(), grad_l2_w.data());
        layer2_b_opt.step(layer2.bias.data(), grad_l2_b.data());
        layer3_w_opt.step(layer3.weight.data(), grad_l3_w.data());
        layer3_b_opt.step(layer3.bias.data(), grad_l3_b.data());
        output_w_opt.step(output.weight.data(), grad_output_w.data());
        output_b_opt.step(output.bias.data(), &grad_output_b);

        return loss;
    }

    // Must be called once after the last train_step() so every feature-
    // transformer row (even ones idle since their last activation) ends up
    // exactly where a dense optimizer would have left it - see
    // sparse_optimizer.hpp's LazySparseAdamW::finalize().
    void finalize_training() { ft_weight_opt.finalize(); }

    // --- Export to the quantized int8/int16 inference network -----------
    void quantize_into(NNUE<ACC_SIZE, H1, H2, H3>& out) const {
        for (std::size_t f = 0; f < NUM_FEATURES; ++f) {
            const float* src = ft_weight_opt.row(f);
            for (std::size_t d = 0; d < ACC_SIZE; ++d)
                out.feature_transformer.weights[f][d] = quantize_i8(src[d] * QA);
        }
        for (std::size_t d = 0; d < ACC_SIZE; ++d)
            out.feature_transformer.biases[d] = static_cast<int16_t>(std::lround(ft_bias[d] * QA));

        quantize_dense_layer<2 * ACC_SIZE, H1>(layer1, out.layer1);
        quantize_dense_layer<H1, H2>(layer2, out.layer2);
        quantize_dense_layer<H2, H3>(layer3, out.layer3);
        quantize_dense_layer<H3, 1>(output, out.output_layer);

        out.ft_shift = 0;       // accumulator is already in the QA domain, see file header
        out.hidden_shift = 6;   // log2(QB), QB = 64
        out.output_scale = (QA * QB) / CP_SCALE;
    }

private:
    void compute_accumulator(const std::vector<int>& active_features, Color perspective, Cache& c) const {
        const std::size_t p = static_cast<std::size_t>(perspective);
        for (std::size_t d = 0; d < ACC_SIZE; ++d) c.acc[p][d] = fake_quantize(ft_bias[d], QA);
        for (int f : active_features) {
            const float* row = ft_weight_opt.row(static_cast<std::size_t>(f));
            for (std::size_t d = 0; d < ACC_SIZE; ++d) c.acc[p][d] += fake_quantize(row[d], QA);
        }
        for (std::size_t d = 0; d < ACC_SIZE; ++d) c.relu_acc[p][d] = relu01(c.acc[p][d]);
    }

    static void accumulate_row_grads(const std::vector<int>& features, const std::vector<float>& grad_acc,
                                      std::unordered_map<int, std::vector<float>>& out) {
        for (int f : features) {
            auto it = out.find(f);
            if (it == out.end()) {
                out.emplace(f, grad_acc); // copy: this row's gradient starts as grad_acc
            } else {
                for (std::size_t d = 0; d < grad_acc.size(); ++d) it->second[d] += grad_acc[d];
            }
        }
    }

    template <std::size_t IN, std::size_t OUT>
    static void dense_forward(const DenseLayerF<IN, OUT>& layer, const float* input, float* out_pre) {
        for (std::size_t o = 0; o < OUT; ++o) {
            float sum = layer.bias[o];
            const float* w = layer.w(o);
            for (std::size_t i = 0; i < IN; ++i) sum += w[i] * input[i];
            out_pre[o] = sum;
        }
    }

    // grad_pre[OUT] -> grad_w[OUT*IN], grad_b[OUT], grad_input[IN] (accumulated, caller must zero-init)
    template <std::size_t IN, std::size_t OUT>
    static void backprop_dense(const DenseLayerF<IN, OUT>& layer, const float* input, const float* grad_pre,
                                float* grad_w, float* grad_b, float* grad_input) {
        for (std::size_t o = 0; o < OUT; ++o) {
            const float g = grad_pre[o];
            grad_b[o] = g;
            const float* w = layer.w(o);
            float* gw = grad_w + o * IN;
            for (std::size_t i = 0; i < IN; ++i) {
                gw[i] = g * input[i];
                grad_input[i] += g * w[i];
            }
        }
    }

    static int8_t quantize_i8(float x) {
        float r = static_cast<float>(std::lround(x));
        if (r > 127.0f) r = 127.0f;
        if (r < -127.0f) r = -127.0f;
        return static_cast<int8_t>(r);
    }

    template <std::size_t IN, std::size_t OUT>
    static void quantize_dense_layer(const DenseLayerF<IN, OUT>& src, AffineLayer<IN, OUT>& dst) {
        for (std::size_t o = 0; o < OUT; ++o) {
            for (std::size_t i = 0; i < IN; ++i) dst.weights[o][i] = quantize_i8(src.w(o)[i] * QB);
            dst.biases[o] = static_cast<int32_t>(std::lround(src.bias[o] * QA * QB));
        }
    }

    void randomize_dense_weights() {
        std::mt19937 rng(1);
        auto init = [&](std::vector<float>& w, std::size_t fan_in) {
            // A conservative fraction of 1/sqrt(fan_in): with a hard-clamped
            // [0,1] activation, the textbook Xavier/Kaiming scale (aimed at
            // keeping pre-activation *variance* near 1) still leaves plenty
            // of neurons starting outside the live region once summed over
            // a wide fan-in - and once a neuron is outside, gradient signal
            // back to it is weak-to-none (see relu01_grad's leak). Starting
            // narrower keeps most units alive at step one; the optimizer
            // still has full freedom to grow any of them later.
            const float bound = 0.25f / std::sqrt(float(fan_in));
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (float& x : w) x = dist(rng);
        };
        init(layer1.weight, 2 * ACC_SIZE);
        init(layer2.weight, H1);
        init(layer3.weight, H2);
        init(output.weight, H3);
        // ft weights start at 0 (handled by LazySparseAdamW's zero-init) -
        // a huge sparse embedding table is conventionally zero- or
        // near-zero-initialized so untouched rows contribute nothing.

        // Every bias that feeds a ClippedReLU (ft_bias, layer1/2/3's) must
        // NOT start at exactly 0: clamp(x,0,1)'s gradient is defined as 0
        // outside the open interval (0,1) (see relu01_grad), so with the ft
        // weights still all zero, acc[d] == ft_bias[d] exactly at step one -
        // if that were also 0, every accumulator dimension would sit right
        // on the dead boundary and NOTHING downstream (not even the ft
        // weights, whose only gradient path runs back through this same
        // ClippedReLU) would ever receive a gradient. Starting these biases
        // in the middle of the live region sidesteps that "everything is
        // dead at initialization" trap. output.bias has no activation after
        // it, so it is exempt and keeps its default 0.
        std::uniform_real_distribution<float> mid_dist(0.4f, 0.6f);
        for (float& x : ft_bias) x = mid_dist(rng);
        for (float& x : layer1.bias) x = mid_dist(rng);
        for (float& x : layer2.bias) x = mid_dist(rng);
        for (float& x : layer3.bias) x = mid_dist(rng);
    }
};

} // namespace nnue::train
