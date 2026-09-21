// Minimal usage example / integration test: builds a toy position, computes
// its HalfKP feature set, evaluates it, makes a move incrementally, and
// checks that the incrementally-updated accumulator matches a from-scratch
// refresh. Weights are random here (no trained network available) - this is
// about exercising the wiring, not about eval quality.
//
// Plug this into a real engine by replacing `Piece`/`Position` below with
// your own board representation and calling refresh()/apply_add()/
// apply_remove() from make_move()/unmake_move() the same way.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "../include/nnue/network.hpp"

using namespace nnue;

namespace {

struct Piece {
    int square;
    PieceType type;
    Color color;
};

// Active HalfKP feature indices, one list per perspective, for a given set
// of pieces and king squares. Kings themselves are not features (see
// halfkp.hpp); they select the p_idx offset via king_square instead.
std::vector<int> active_features(Color perspective, const std::vector<Piece>& pieces,
                                  int white_king, int black_king) {
    const int king_sq = perspective == WHITE ? white_king : black_king;
    std::vector<int> out;
    out.reserve(pieces.size());
    for (const Piece& p : pieces) {
        out.push_back(feature_index(perspective, p.square, p.type, p.color, king_sq));
    }
    return out;
}

template <typename Net>
void randomize_weights(Net& net) {
    std::srand(42);
    auto rnd_i8 = []() { return int8_t(std::rand() % 41 - 20); }; // small range so nothing saturates immediately
    for (auto& row : net.feature_transformer.weights)
        for (auto& w : row) w = rnd_i8();
    for (auto& b : net.feature_transformer.biases) b = int16_t(std::rand() % 21 - 10);

    auto fill_layer = [&](auto& layer) {
        for (auto& row : layer.weights)
            for (auto& w : row) w = rnd_i8();
        for (auto& b : layer.biases) b = std::rand() % 21 - 10;
    };
    fill_layer(net.layer1);
    fill_layer(net.layer2);
    fill_layer(net.layer3);
    fill_layer(net.output_layer);
}

} // namespace

int main() {
    // A Stockfish-ish "HalfKP-256x2-32-32-32" sized network.
    using Net = NNUE</*ACC_SIZE=*/256, /*H1=*/128, /*H2=*/32, /*H3=*/32>;
    auto net = Net::make(); // heap-allocated: the feature transformer alone is ~10 MB
    randomize_weights(*net);

    // A handful of pieces plus both kings. Squares are 0..63, a1=0 .. h8=63.
    int white_king = 4;   // e1
    int black_king = 60;  // e8
    std::vector<Piece> pieces = {
        {8,  PAWN,   WHITE}, // a2
        {9,  PAWN,   WHITE}, // b2
        {1,  KNIGHT, WHITE}, // b1
        {52, PAWN,   BLACK}, // e7
        {57, KNIGHT, BLACK}, // b8
    };

    nnue::Accumulator<256> accumulator;
    accumulator.refresh(net->feature_transformer, WHITE, active_features(WHITE, pieces, white_king, black_king));
    accumulator.refresh(net->feature_transformer, BLACK, active_features(BLACK, pieces, white_king, black_king));

    int32_t eval_before = net->evaluate(accumulator, WHITE);
    std::printf("eval before move: %d (%.2f cp)\n", eval_before, net->evaluate_cp(accumulator, WHITE));

    // Move the white knight b1 -> c3 (square 1 -> 18). Not a king move and
    // not a capture, so this is a pure incremental update: for each
    // perspective, remove the feature for (b1, knight, white) and add the
    // feature for (c3, knight, white).
    const int from_sq = 1, to_sq = 18;
    for (Color perspective : {WHITE, BLACK}) {
        const int king_sq = perspective == WHITE ? white_king : black_king;
        const int f_from = feature_index(perspective, from_sq, KNIGHT, WHITE, king_sq);
        const int f_to   = feature_index(perspective, to_sq,   KNIGHT, WHITE, king_sq);
        accumulator.apply_remove(net->feature_transformer, perspective, f_from);
        accumulator.apply_add(net->feature_transformer, perspective, f_to);
    }
    pieces[2].square = to_sq;

    int32_t eval_after_incremental = net->evaluate(accumulator, WHITE);

    // Sanity check: a full refresh from the updated piece list must give the
    // exact same accumulator (and therefore eval) as the incremental update.
    nnue::Accumulator<256> refreshed;
    refreshed.refresh(net->feature_transformer, WHITE, active_features(WHITE, pieces, white_king, black_king));
    refreshed.refresh(net->feature_transformer, BLACK, active_features(BLACK, pieces, white_king, black_king));
    int32_t eval_after_refresh = net->evaluate(refreshed, WHITE);

    std::printf("eval after move (incremental): %d\n", eval_after_incremental);
    std::printf("eval after move (full refresh): %d\n", eval_after_refresh);
    std::printf("%s\n", eval_after_incremental == eval_after_refresh
                             ? "OK: incremental update matches full refresh"
                             : "MISMATCH: incremental update bug!");

    // Now move the white king e1 -> f1 (square 4 -> 5). This changes the
    // king_square term of *every* white-perspective feature, so White's
    // accumulator must be refreshed from scratch; Black's accumulator only
    // sees "the white king piece" if kings were features (they are not), so
    // nothing needs to change on Black's side for this particular move.
    white_king = 5;
    accumulator.refresh(net->feature_transformer, WHITE, active_features(WHITE, pieces, white_king, black_king));

    std::printf("eval after king move: %d\n", net->evaluate(accumulator, WHITE));
    return eval_after_incremental == eval_after_refresh ? 0 : 1;
}
