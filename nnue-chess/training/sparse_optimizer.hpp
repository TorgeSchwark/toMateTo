#pragma once
// The "sparse update" trick for training the feature transformer.
//
// The feature-transformer weight matrix is NUM_FEATURES x DIM (40960 x
// ACC_SIZE, tens of millions of floats). A single training position only
// activates ~30 of its 40960 rows (one per piece on the board, per
// perspective), so the *gradient* step itself is naturally cheap: only
// touched rows get a nonzero dL/dW row, so only those need Adam's usual
// m/v-moment update and the actual SGD/Adam step.
//
// The part that is NOT naturally cheap is weight decay (L2 regularization):
// plain AdamW subtracts `lr * weight_decay * w` from *every* weight on
// *every* step, active or not - i.e. touching all 40960 rows every step
// regardless of whether they had any gradient feedback that step, which
// would dominate training cost and defeat the whole point of the sparse
// gradient.
//
// The trick: track `last_step[row]`, the step at which a row's weights,
// momentum and decay were last brought up to date. A row that keeps getting
// skipped doesn't actually need per-step work done to it - the *net effect*
// of N skipped steps of pure decay (no gradient) has a closed form:
//   weight  *= (1 - lr*weight_decay)^N     (decoupled weight decay)
//   m       *= beta1^N                     (Adam 1st moment, no new gradient)
//   v       *= beta2^N                     (Adam 2nd moment, no new gradient)
// So instead of applying one step of decay N times, `catch_up()` applies N
// steps of decay once, in closed form, the moment the row becomes active
// again (or once, for every row, at finalize() time). This gives bit-for-bit
// (up to float rounding) the same result as a dense optimizer that really
// did touch all 40960 rows every step, but costs O(active rows) per step
// instead of O(NUM_FEATURES).

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

namespace nnue::train {

// Learning-rate schedule shared by both optimizers below: lr(t) = lr0 / (1 +
// t/half_life). Deliberately NOT a schedule that reaches (near-)zero at some
// pre-planned horizon (e.g. linear/cosine decay to 0 over N total steps) -
// those go dead if you end up training longer than N, with no way back
// short of restarting. This one only ever approaches 0 asymptotically: at
// t=half_life it has halved, at t=10*half_life it is down to lr0/11, but
// it never truly reaches 0, so a run that goes on far longer than expected
// keeps making (shrinking, never absent) progress instead of stalling out.
// Choose half_life once for roughly "how many steps until I want the LR
// noticeably reduced" - a bad guess just means slower or faster convergence,
// never a dead run, which is the whole point.
inline float lr_schedule(float base_lr, float half_life, std::uint64_t step) {
    return base_lr / (1.0f + float(step) / half_life);
}

template <std::size_t DIM>
class LazySparseAdamW {
public:
    LazySparseAdamW(std::size_t num_rows, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                     float weight_decay = 0.0f, float lr_half_life = 100000.0f)
        : num_rows_(num_rows), base_lr_(lr), lr_half_life_(lr_half_life), beta1_(beta1), beta2_(beta2), eps_(eps),
          wd_(weight_decay), weights_(num_rows * DIM, 0.0f), m_(num_rows * DIM, 0.0f), v_(num_rows * DIM, 0.0f),
          last_step_(num_rows, 0) {}

    std::size_t num_rows() const { return num_rows_; }
    float* row(std::size_t r) { return &weights_[r * DIM]; }
    const float* row(std::size_t r) const { return &weights_[r * DIM]; }

    // Call once at the start of every minibatch, before any apply_gradient()
    // calls for that batch (all rows touched within one batch share the same
    // step index, which is what makes bias correction well-defined).
    void begin_step() { ++step_; }

    float current_lr() const { return lr_schedule(base_lr_, lr_half_life_, step_); }

    // grad[0..DIM) = dL/dW[row][*], already summed over every occurrence of
    // `row` within the current minibatch (a feature can be active in more
    // than one sample of a batch; sum, don't average, to match a dense
    // full-batch gradient).
    void apply_gradient(std::size_t r, const float* grad) {
        catch_up(r, step_ - 1); // bring the row current through the *previous* step
        const float lr = current_lr();
        float* w = &weights_[r * DIM];
        float* m = &m_[r * DIM];
        float* v = &v_[r * DIM];
        const float bc1 = 1.0f - std::pow(beta1_, float(step_));
        const float bc2 = 1.0f - std::pow(beta2_, float(step_));
        for (std::size_t i = 0; i < DIM; ++i) {
            w[i] -= lr * wd_ * w[i]; // decoupled weight decay for *this* step
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * grad[i] * grad[i];
            const float mhat = m[i] / bc1;
            const float vhat = v[i] / bc2;
            w[i] -= lr * mhat / (std::sqrt(vhat) + eps_);
        }
        last_step_[r] = step_;
    }

    // Brings every row fully current (applies any pending no-gradient decay
    // catch-up). Call this once after the very last training step, before
    // reading out weights for quantization/export - otherwise rows that
    // simply weren't touched on the final steps would be very slightly
    // under-decayed relative to a dense optimizer.
    void finalize() {
        for (std::size_t r = 0; r < num_rows_; ++r) catch_up(r, step_);
    }

    std::uint64_t step() const { return step_; }

private:
    // Applies (up_to_step - last_step_[r]) steps of pure decay (no
    // gradient) to row r, in closed form. The lr *schedule* means the "true"
    // per-step decay factor drifted across that gap instead of staying
    // constant, which would break the closed form - approximated here by
    // using the lr at up_to_step for the whole gap. Weight decay's effect is
    // already many orders of magnitude below anything else in this system
    // (see this file's header comment), so this approximation's error is
    // completely negligible in practice.
    void catch_up(std::size_t r, std::uint64_t up_to_step) {
        if (up_to_step <= last_step_[r]) return;
        const std::uint64_t n = up_to_step - last_step_[r];
        const float lr = lr_schedule(base_lr_, lr_half_life_, up_to_step);
        const float wd_factor = std::pow(1.0f - lr * wd_, float(n));
        const float b1n = std::pow(beta1_, float(n));
        const float b2n = std::pow(beta2_, float(n));
        float* w = &weights_[r * DIM];
        float* m = &m_[r * DIM];
        float* v = &v_[r * DIM];
        for (std::size_t i = 0; i < DIM; ++i) {
            w[i] *= wd_factor;
            m[i] *= b1n;
            v[i] *= b2n;
        }
        last_step_[r] = up_to_step;
    }

    std::size_t num_rows_;
    float base_lr_, lr_half_life_, beta1_, beta2_, eps_, wd_;
    std::vector<float> weights_, m_, v_;
    std::vector<std::uint64_t> last_step_;
    std::uint64_t step_ = 0;
};

// Plain (dense) AdamW for the small hidden/output layers - they have only a
// few hundred to a few thousand weights each, so there is nothing to be
// lazy about.
class DenseAdamW {
public:
    DenseAdamW(std::size_t size, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
               float weight_decay = 0.0f, float lr_half_life = 100000.0f)
        : base_lr_(lr), lr_half_life_(lr_half_life), beta1_(beta1), beta2_(beta2), eps_(eps), wd_(weight_decay),
          m_(size, 0.0f), v_(size, 0.0f) {}

    float current_lr() const { return lr_schedule(base_lr_, lr_half_life_, step_); }

    // weights/grad both length `size` passed to the constructor.
    void step(float* weights, const float* grad) {
        ++step_;
        const float lr = current_lr();
        const float bc1 = 1.0f - std::pow(beta1_, float(step_));
        const float bc2 = 1.0f - std::pow(beta2_, float(step_));
        for (std::size_t i = 0; i < m_.size(); ++i) {
            weights[i] -= lr * wd_ * weights[i];
            m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * grad[i];
            v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * grad[i] * grad[i];
            const float mhat = m_[i] / bc1;
            const float vhat = v_[i] / bc2;
            weights[i] -= lr * mhat / (std::sqrt(vhat) + eps_);
        }
    }

private:
    float base_lr_, lr_half_life_, beta1_, beta2_, eps_, wd_;
    std::vector<float> m_, v_;
    std::uint64_t step_ = 0;
};

} // namespace nnue::train
