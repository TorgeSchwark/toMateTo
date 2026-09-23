#pragma once
// Automated regression test for the exact failure mode this project spent a
// long time chasing by hand: a network that predicts (near-)the same value
// for every position, or predicts values so small they carry no real
// information. Computes mean|eval| and the standard deviation of eval over
// a batch of positions and flags both directly, instead of relying on
// someone noticing a suspiciously flat loss curve or a near-zero
// correlation number after the fact.
//
// Generic on purpose (no chess_board dependency, see leaderboard.hpp /
// validation.hpp's header comments for why full_cycle.cpp and
// self_play.cpp can't share board-touching code): callers collect a
// std::vector<float> of raw (pre-sigmoid) evaluations themselves and pass
// it in.

#include <cmath>
#include <cstdio>
#include <vector>

struct EvalSanityStats {
    double mean = 0.0;
    double mean_abs = 0.0;
    double stddev = 0.0;
    bool suspiciously_flat = false; // stddev near 0: "always predicts about the same value"
    bool suspiciously_tiny = false; // mean|eval| near 0: "predictions barely move off the baseline"
};

// Thresholds are heuristic, not derived from any formal criterion - picked
// to flag the specific degenerate case seen during development (stddev/
// mean|eval| both several orders of magnitude below these) without
// tripping on a merely-not-great-yet network. In raw "cp/400" units (see
// train_network.hpp): a stddev of 0.02 corresponds to ~8 centipawns of
// spread, well below what any remotely useful evaluator should show across
// a diverse batch of positions.
constexpr double kMinHealthyStddev = 0.02;
constexpr double kMinHealthyMeanAbs = 0.01;

inline EvalSanityStats compute_eval_sanity_stats(const std::vector<float>& evals) {
    EvalSanityStats s;
    if (evals.empty()) return s;
    double sum = 0.0, sum_abs = 0.0;
    for (float e : evals) {
        sum += e;
        sum_abs += std::abs(double(e));
    }
    const double n = double(evals.size());
    s.mean = sum / n;
    s.mean_abs = sum_abs / n;
    double var = 0.0;
    for (float e : evals) {
        const double d = double(e) - s.mean;
        var += d * d;
    }
    s.stddev = std::sqrt(var / n);
    s.suspiciously_flat = s.stddev < kMinHealthyStddev;
    s.suspiciously_tiny = s.mean_abs < kMinHealthyMeanAbs;
    return s;
}

inline void print_eval_sanity_stats(const EvalSanityStats& s, std::size_t n) {
    std::printf("== eval sanity check (%zu positions) ==\n", n);
    std::printf("  mean|eval|=%.4f  stddev=%.4f  (raw units, ~cp/400 - see train_network.hpp)\n", s.mean_abs,
                s.stddev);
    if (s.suspiciously_flat) {
        std::printf(
            "  WARNING: standard deviation is very low - the network may be predicting nearly\n"
            "  the same value for every position regardless of what's on the board. This is\n"
            "  the exact failure mode this project spent a long time chasing by hand (see\n"
            "  randomize_ft_weights()'s and search_net's comments in self_play.cpp) - treat\n"
            "  this run's output with suspicion.\n");
    }
    if (s.suspiciously_tiny) {
        std::printf("  WARNING: average |eval| is very close to 0 - predictions may be\n"
                    "  uninformatively small across the board.\n");
    }
    if (!s.suspiciously_flat && !s.suspiciously_tiny) {
        std::printf("  looks healthy: predictions vary meaningfully across positions.\n");
    }
    std::printf("\n");
}
