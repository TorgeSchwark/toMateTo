#pragma once
// The sparse first layer plus the incrementally-updated accumulator that
// sits on top of it.
//
// For every one of the NUM_FEATURES = 40960 HalfKP features there is one
// row of ACC_SIZE int8 weights. A real chess position only activates ~30 of
// them (one per piece on the board, per perspective), so this layer is
// never evaluated as a matrix multiply: the "forward pass" is just "add
// together the rows for the active features", and because a make_move only
// flips a handful of features on/off, the running sum (the accumulator) can
// be updated incrementally in O(moved pieces) instead of recomputed in
// O(NUM_FEATURES) every move.

#include <cstdint>
#include <cstddef>
#include <array>
#include <cstring>
#include "halfkp.hpp"
#include "simd.hpp"

namespace nnue {

// NOTE: sizeof(FeatureTransformer<ACC_SIZE>) is NUM_FEATURES * ACC_SIZE
// bytes (~10.5 MB for ACC_SIZE=256). Always heap-allocate this
// (e.g. std::make_unique<FeatureTransformer<256>>()) - it is far too big
// for the stack.
template <std::size_t ACC_SIZE>
struct FeatureTransformer {
    static constexpr std::size_t kAccSize = ACC_SIZE;

    alignas(64) std::array<std::array<int8_t, ACC_SIZE>, NUM_FEATURES> weights{};
    alignas(64) std::array<int16_t, ACC_SIZE> biases{};

    // acc := biases  (the state before any feature is added)
    void refresh(int16_t* acc) const {
        std::memcpy(acc, biases.data(), sizeof(int16_t) * ACC_SIZE);
    }

    // acc += weights[feature]   (a piece appeared for this perspective)
    void add_feature(int16_t* acc, int feature_index) const {
        simd::add_row_i8_to_i16(acc, weights[static_cast<std::size_t>(feature_index)].data(), ACC_SIZE);
    }

    // acc -= weights[feature]   (a piece disappeared for this perspective)
    void remove_feature(int16_t* acc, int feature_index) const {
        simd::sub_row_i8_from_i16(acc, weights[static_cast<std::size_t>(feature_index)].data(), ACC_SIZE);
    }
};

// Holds both perspectives' running accumulators (index by Color: WHITE=0,
// BLACK=1). This is the piece of state a search node carries forward from
// its parent and patches in place on make_move / unmakes by restoring the
// parent's copy - it is cheap to copy (2 * ACC_SIZE int16) precisely so that
// pattern works.
template <std::size_t ACC_SIZE>
struct Accumulator {
    alignas(64) std::array<std::array<int16_t, ACC_SIZE>, 2> values{};

    // Full recompute for one perspective from scratch. Needed once at the
    // start of a search, and any time that perspective's own king moves -
    // king_square is multiplied into *every* active feature's index (see
    // halfkp.hpp), so a king move changes literally every feature that
    // perspective sees, not just the king's own square.
    template <typename FeatureIndexRange>
    void refresh(const FeatureTransformer<ACC_SIZE>& ft, Color perspective,
                 const FeatureIndexRange& active_features) {
        int16_t* acc = values[static_cast<std::size_t>(perspective)].data();
        ft.refresh(acc);
        for (int f : active_features) ft.add_feature(acc, f);
    }

    // Incremental update for one perspective: apply the small list of
    // features that actually turned on/off because of one move. A normal
    // non-capture move touches 2 features (remove source, add destination)
    // per unaffected perspective; a capture adds one more removal; castling
    // touches the rook the same way. Do NOT call this for the perspective
    // whose own king just moved - use refresh() for that one instead.
    void apply_add(const FeatureTransformer<ACC_SIZE>& ft, Color perspective, int feature_index) {
        ft.add_feature(values[static_cast<std::size_t>(perspective)].data(), feature_index);
    }
    void apply_remove(const FeatureTransformer<ACC_SIZE>& ft, Color perspective, int feature_index) {
        ft.remove_feature(values[static_cast<std::size_t>(perspective)].data(), feature_index);
    }
};

} // namespace nnue
