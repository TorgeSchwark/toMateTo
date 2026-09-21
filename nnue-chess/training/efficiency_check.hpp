#pragma once
// Benchmarks the operations that matter for search speed, for whichever
// NNUE<ACC,H1,H2,H3> architecture the calling tool was compiled for: a full
// accumulator refresh (paid once per search root / whenever a king moves),
// a single incremental add/remove (paid on every other make_move), and one
// full evaluate() call (the actual per-leaf cost during search - see the
// full_cycle.cpp/self_play.cpp header comments for why this is a very
// different number from "time to generate one Stockfish-labeled training
// sample" or "time to play one self-play search").

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/nnue/network.hpp"

template <std::size_t ACC_SIZE, std::size_t H1, std::size_t H2, std::size_t H3>
void run_efficiency_check() {
    using Inference = nnue::NNUE<ACC_SIZE, H1, H2, H3>;

    std::printf("== efficiency check (ACC_SIZE=%zu) ==\n", ACC_SIZE);
    auto ft = std::make_unique<nnue::FeatureTransformer<ACC_SIZE>>();
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> wdist(-20, 20);
    for (auto& row : ft->weights)
        for (auto& w : row) w = int8_t(wdist(rng));
    for (auto& b : ft->biases) b = int16_t(wdist(rng));

    std::uniform_int_distribution<int> fdist(0, int(nnue::NUM_FEATURES) - 1);
    std::vector<int> features;
    for (int i = 0; i < 32; ++i) features.push_back(fdist(rng));

    nnue::Accumulator<ACC_SIZE> acc;
    constexpr int kFullIters = 20000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kFullIters; ++i) acc.refresh(*ft, nnue::WHITE, features);
    auto t1 = std::chrono::high_resolution_clock::now();
    double full_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / kFullIters;

    constexpr int kUpdateIters = 200000;
    int f = features.front();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kUpdateIters; ++i) {
        acc.apply_add(*ft, nnue::WHITE, f);
        acc.apply_remove(*ft, nnue::WHITE, f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double update_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (kUpdateIters * 2);

    auto net = Inference::make();
    for (auto& row : net->feature_transformer.weights)
        for (auto& w : row) w = int8_t(wdist(rng));
    for (auto& b : net->feature_transformer.biases) b = int16_t(wdist(rng));
    auto rand_dense = [&](auto& layer) {
        for (auto& row : layer.weights)
            for (auto& w : row) w = int8_t(wdist(rng));
        for (auto& b : layer.biases) b = wdist(rng);
    };
    rand_dense(net->layer1);
    rand_dense(net->layer2);
    rand_dense(net->layer3);
    rand_dense(net->output_layer);
    acc.refresh(net->feature_transformer, nnue::WHITE, features);
    acc.refresh(net->feature_transformer, nnue::BLACK, features);

    constexpr int kEvalIters = 200000;
    volatile int32_t sink = 0; // prevents the optimizer from deleting the loop
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kEvalIters; ++i) sink = net->evaluate(acc, nnue::WHITE);
    t1 = std::chrono::high_resolution_clock::now();
    double eval_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / kEvalIters;
    (void)sink;

    std::printf("  full refresh (%zu active features): %8.1f ns/call\n", features.size(), full_ns);
    std::printf("  incremental update (add or remove):  %8.1f ns/call  (%.1fx faster than full)\n", update_ns,
                full_ns / update_ns);
    std::printf("  full evaluate() call:                %8.1f ns/call  (%.4f ms)\n\n", eval_ns, eval_ns / 1e6);
}
