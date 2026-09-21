// Demonstrates the training loop end-to-end on synthetic data: generates
// random "positions" (random legal-ish piece placements, not real chess
// legality - swap generate_sample() for a real PGN/binpack data loader),
// trains a small NNUE with the sparse optimizer, quantizes it, and confirms
// the quantized int8 network's evaluation tracks the float training
// network's prediction.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#include "train_network.hpp"

using namespace nnue;
using namespace nnue::train;

namespace {

// A random small set of pieces plus two king squares, and a synthetic
// "target_wdl" that is a deterministic function of material count - this
// gives the training loop something learnable to chase without needing a
// real evaluation function or labeled dataset.
//
// Squares are drawn from a small fixed pool rather than all 64: with the
// full HalfKP space (40960 features) and positions that are each
// independently uniformly random, essentially every training sample would
// activate a set of feature rows that no other sample - training or
// held-out test - ever touches again, so there would be nothing for the
// network to generalize from in a demo-sized run. Real training data does
// not have this problem (millions of games naturally revisit the same
// squares/king positions over and over); this pool just reproduces that
// repetition cheaply for a short demo.
TrainingSample generate_sample(std::mt19937& rng) {
    static const int kSquarePool[] = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19};
    constexpr int kPoolSize = int(sizeof(kSquarePool) / sizeof(kSquarePool[0]));
    std::uniform_int_distribution<int> pool_dist(0, kPoolSize - 1);
    auto sq_dist = [&](std::mt19937& r) { return kSquarePool[pool_dist(r)]; };
    std::uniform_int_distribution<int> piece_dist(0, 4);
    std::uniform_int_distribution<int> color_dist(0, 1);
    std::uniform_int_distribution<int> count_dist(2, 8);

    int white_king = sq_dist(rng);
    int black_king = sq_dist(rng);
    while (black_king == white_king) black_king = sq_dist(rng);

    struct P { int sq; PieceType t; Color c; };
    std::vector<P> pieces;
    int material = 0;
    static const int piece_value[5] = {1, 3, 3, 5, 9}; // P,N,B,R,Q
    int n = count_dist(rng);
    for (int i = 0; i < n; ++i) {
        int sq = sq_dist(rng);
        if (sq == white_king || sq == black_king) continue;
        PieceType t = static_cast<PieceType>(piece_dist(rng));
        Color c = static_cast<Color>(color_dist(rng));
        pieces.push_back({sq, t, c});
        material += (c == WHITE ? 1 : -1) * piece_value[t];
    }

    TrainingSample s;
    s.side_to_move = static_cast<Color>(color_dist(rng));
    for (const P& p : pieces) {
        s.white_features.push_back(feature_index(WHITE, p.sq, p.t, p.c, white_king));
        s.black_features.push_back(feature_index(BLACK, p.sq, p.t, p.c, black_king));
    }
    // material advantage -> win probability via a logistic curve, matching
    // the same sigmoid(cp/400)-shaped target the network itself predicts.
    float cp = float(material) * 100.0f; // 1 pawn ~= 100cp
    if (s.side_to_move == BLACK) cp = -cp;
    s.target_wdl = sigmoid(cp / CP_SCALE);
    return s;
}

} // namespace

int main() {
    using Net = TrainNetwork</*ACC_SIZE=*/64, /*H1=*/128, /*H2=*/32, /*H3=*/32>;
    Net net(/*lr=*/0.0001f, /*ft_weight_decay=*/1e-6f, /*hidden_weight_decay=*/1e-4f);

    std::mt19937 rng(7);
    const int kSteps = 150000;
    float running_loss = 0.0f;
    for (int step = 1; step <= kSteps; ++step) {
        TrainingSample sample = generate_sample(rng);
        float loss = net.train_step(sample);
        running_loss += loss;
        if (step % 15000 == 0) {
            std::printf("step %6d  avg loss (last 15000) = %.5f\n", step, running_loss / 15000.0f);
            running_loss = 0.0f;
        }
    }
    net.finalize_training(); // catch up every feature-transformer row's pending weight decay before export

    // NNUE<64,128,32,32> is ~2.6 MB (the feature-transformer table alone) -
    // always heap-allocate it, never declare it as a local/stack variable.
    auto quantized = NNUE<64, 128, 32, 32>::make();
    net.quantize_into(*quantized);
    std::printf("quantized network ready (ft_shift=%d hidden_shift=%d output_scale=%.3f)\n", quantized->ft_shift,
                quantized->hidden_shift, quantized->output_scale);

    // Spot-check: float training net's prediction vs. the quantized int8
    // net's evaluation should agree in sign and rough magnitude (exact
    // equality isn't expected - that's the whole point/cost of int8
    // quantization), and both should track target_wdl's direction.
    std::mt19937 test_rng(123);
    int agree = 0, total = 20;
    for (int i = 0; i < total; ++i) {
        TrainingSample s = generate_sample(test_rng);
        typename Net::Cache cache;
        float pred = net.forward(s, cache);
        float train_cp = std::log(pred / (1.0f - pred + 1e-9f) + 1e-9f) * CP_SCALE;

        Accumulator<64> acc;
        acc.refresh(quantized->feature_transformer, WHITE, s.white_features);
        acc.refresh(quantized->feature_transformer, BLACK, s.black_features);
        float quant_cp = quantized->evaluate_cp(acc, s.side_to_move);

        bool same_sign = (train_cp > 0) == (quant_cp > 0) || (std::abs(train_cp) < 5 && std::abs(quant_cp) < 20);
        agree += same_sign ? 1 : 0;
        std::printf("sample %2d: target_wdl=%.3f  train_cp=%8.1f  quant_cp=%8.1f  %s\n", i, s.target_wdl, train_cp,
                    quant_cp, same_sign ? "ok" : "DIVERGED");
    }
    std::printf("%d/%d samples agree in sign\n", agree, total);

    quantized->save("trained_net.nnue");
    std::printf("saved trained_net.nnue\n");
    return 0;
}
