#pragma once
// Scale-robust validation against Stockfish.
//
// A network's raw (pre-sigmoid) evaluation is only meaningful on an
// absolute "centipawns" scale if something anchors it there. Stockfish-
// supervised training (full_cycle.cpp) has that anchor built in: every
// training target is literally sigmoid(stockfish_cp / CP_SCALE), so the
// network's own raw output is pulled toward matching real centipawns as a
// side effect of minimizing that loss. Self-play (self_play.cpp) has no
// such anchor - its targets come from the network's own search, so the
// network is free to converge to "correct move preferences, but on some
// arbitrary internal scale" (e.g. what it calls 1.0 might be what
// Stockfish would call 150cp, or 600cp - nothing forces those to match).
//
// That means comparing self-play's training loss directly against
// full_cycle's is comparing different units. This header fixes that: it
// evaluates a network on a small held-out set of freshly Stockfish-scored
// positions, fits the best-fit affine correction (scale + offset) between
// the network's raw evals and Stockfish's, and reports the error *after*
// that correction - so a network whose judgments are essentially "correct
// but on the wrong scale" scores just as well as one already on the right
// scale, and only genuine disagreement about which positions are good
// shows up in the numbers. Also reports the raw (uncalibrated) Pearson
// correlation, which is scale/offset-invariant on its own and a useful
// sanity check independent of the fit.

#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "train_network.hpp"

struct ValidationResult {
    float correlation = 0.0f;          // Pearson r between raw net/Stockfish evals, scale-invariant by itself
    float scale = 1.0f, offset = 0.0f; // best-fit: stockfish_cp/CP_SCALE ~= scale * net_raw + offset
    float calibrated_prob_mse = 0.0f;  // MSE in win-probability space AFTER applying the fit - the number to
                                        // compare against full_cycle's training loss / another run's validation_loss
};

// gen_fen_and_cp: produces one (FEN, stockfish_cp) pair for a fresh random
// legal position - supplied by the caller since only it knows the concrete
// board type and move generator in use (full_cycle.cpp and self_play.cpp
// each link a *different* copy of the toMateTo engine, so this header
// deliberately has no direct dependency on either).
// raw_eval_for_fen: the network's raw (pre-sigmoid) evaluation for a
// position given by FEN - likewise supplied by the caller.
inline ValidationResult validate_against_stockfish(
    int n_positions, const std::function<std::pair<std::string, int>()>& gen_fen_and_cp,
    const std::function<float(const std::string&)>& raw_eval_for_fen) {
    std::vector<float> net_raw, sf_raw;
    net_raw.reserve(std::size_t(n_positions));
    sf_raw.reserve(std::size_t(n_positions));
    for (int i = 0; i < n_positions; ++i) {
        auto [fen, cp] = gen_fen_and_cp();
        net_raw.push_back(raw_eval_for_fen(fen));
        sf_raw.push_back(float(cp) / nnue::train::CP_SCALE);
    }

    const double n = double(net_raw.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < net_raw.size(); ++i) {
        sx += net_raw[i];
        sy += sf_raw[i];
        sxx += double(net_raw[i]) * net_raw[i];
        sxy += double(net_raw[i]) * sf_raw[i];
    }
    const double denom = n * sxx - sx * sx;
    const float scale = denom != 0.0 ? float((n * sxy - sx * sy) / denom) : 1.0f;
    const float offset = float((sy - scale * sx) / n);

    const double mean_x = sx / n, mean_y = sy / n;
    double cov = 0, varx = 0, vary = 0;
    for (std::size_t i = 0; i < net_raw.size(); ++i) {
        const double dx = net_raw[i] - mean_x, dy = sf_raw[i] - mean_y;
        cov += dx * dy;
        varx += dx * dx;
        vary += dy * dy;
    }
    const float correlation = (varx > 0 && vary > 0) ? float(cov / std::sqrt(varx * vary)) : 0.0f;

    double mse = 0;
    for (std::size_t i = 0; i < net_raw.size(); ++i) {
        const float calibrated = scale * net_raw[i] + offset;
        const float p_net = nnue::train::sigmoid(calibrated);
        const float p_sf = nnue::train::sigmoid(sf_raw[i]);
        const double d = double(p_net) - double(p_sf);
        mse += d * d;
    }
    mse /= n;

    ValidationResult r;
    r.correlation = correlation;
    r.scale = scale;
    r.offset = offset;
    r.calibrated_prob_mse = float(mse);
    return r;
}
