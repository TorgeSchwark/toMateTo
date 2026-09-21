#pragma once
// Shared results leaderboard, used by both full_cycle.cpp (Stockfish-
// supervised training) and self_play.cpp (self-play TD-learning): every
// run appends one row to training_results.csv (relative to the current
// working directory, same place the .dataset/.nnue files land) - plain
// append-only CSV instead of a database, so results survive across runs,
// tools and architectures with nothing extra to install.

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct TrainingResult {
    std::string name;
    std::string method; // "stockfish" (full_cycle) or "selfplay" (self_play)
    std::string timestamp;
    int acc = 0, h1 = 0, h2 = 0, h3 = 0;
    int depth = 0; // Stockfish search depth, or self-play search depth - same column, meaning depends on method
    long samples = 0; // positions trained on, or games played (see games field) - see method
    long games = 0;   // self-play only; 0 for Stockfish-supervised runs
    float best_loss = 0.0f;  // lowest reported windowed-average loss seen during training
    float final_loss = 0.0f; // the last reported windowed-average loss
    double total_time_ms = 0.0; // data generation/self-play + training, not counting engine/table init

    // Held-out comparison against Stockfish (see validation.hpp), scale-
    // corrected so self-play's internally-bootstrapped evaluation scale
    // (which has nothing anchoring it to real centipawns) is still
    // meaningfully comparable to full_cycle's Stockfish-supervised loss:
    // validation_loss is calibrated_prob_mse, directly comparable across
    // methods/runs; validation_correlation is the raw (uncalibrated)
    // Pearson correlation, -1 if not computed for this run.
    float validation_loss = -1.0f;
    float validation_correlation = -1.0f;

    std::string architecture() const {
        return std::to_string(acc) + "," + std::to_string(h1) + "," + std::to_string(h2) + "," + std::to_string(h3);
    }
};

inline std::string results_csv_path() { return "training_results.csv"; }

inline std::string current_timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

inline void append_result(const TrainingResult& r) {
    const std::string path = results_csv_path();
    bool need_header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("could not open " + path + " for writing");
    if (need_header)
        f << "name,method,timestamp,acc,h1,h2,h3,depth,samples,games,best_loss,final_loss,total_time_ms,"
             "validation_loss,validation_correlation\n";
    f << r.name << ',' << r.method << ',' << r.timestamp << ',' << r.acc << ',' << r.h1 << ',' << r.h2 << ','
      << r.h3 << ',' << r.depth << ',' << r.samples << ',' << r.games << ',' << r.best_loss << ',' << r.final_loss
      << ',' << r.total_time_ms << ',' << r.validation_loss << ',' << r.validation_correlation << '\n';
}

inline std::vector<TrainingResult> load_results() {
    std::vector<TrainingResult> out;
    std::ifstream f(results_csv_path());
    if (!f) return out;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        TrainingResult r;
        auto next = [&]() -> std::string {
            std::getline(ss, field, ',');
            return field;
        };
        r.name = next();
        r.method = next();
        r.timestamp = next();
        r.acc = std::stoi(next());
        r.h1 = std::stoi(next());
        r.h2 = std::stoi(next());
        r.h3 = std::stoi(next());
        r.depth = std::stoi(next());
        r.samples = std::stol(next());
        r.games = std::stol(next());
        r.best_loss = std::stof(next());
        r.final_loss = std::stof(next());
        r.total_time_ms = std::stod(next());
        r.validation_loss = std::stof(next());
        r.validation_correlation = std::stof(next());
        out.push_back(std::move(r));
    }
    return out;
}

// Ranked by validation_loss when available (the scale-corrected,
// cross-method-comparable number - see validation.hpp) so a self-play run's
// self-referential training loss never gets compared apples-to-oranges
// against a Stockfish-supervised run's; falls back to best_loss for older
// rows/runs that skipped validation.
inline float ranking_key(const TrainingResult& r) { return r.validation_loss >= 0.0f ? r.validation_loss : r.best_loss; }

// Prints every recorded run, ranked best (lowest ranking_key) first:
// "Platz 1 / Platz 2 / ..." with method, architecture and computation time
// on each line, plus the validation numbers when present.
inline void print_leaderboard(int top_n = 10) {
    std::vector<TrainingResult> results = load_results();
    if (results.empty()) {
        std::printf("(noch keine Trainingsergebnisse in %s)\n", results_csv_path().c_str());
        return;
    }
    std::sort(results.begin(), results.end(),
              [](const TrainingResult& a, const TrainingResult& b) { return ranking_key(a) < ranking_key(b); });

    std::printf("\n== Leaderboard (%s, %zu Läufe insgesamt) ==\n", results_csv_path().c_str(), results.size());
    int n = std::min<int>(top_n, int(results.size()));
    for (int i = 0; i < n; ++i) {
        const TrainingResult& r = results[i];
        std::string val_str = r.validation_loss >= 0.0f
                                   ? "  val_loss=" + std::to_string(r.validation_loss) +
                                         " corr=" + std::to_string(r.validation_correlation)
                                   : "";
        std::string rank_label = (i == 0) ? "momentan bester (val_loss)" : ("Platz " + std::to_string(i + 1));
        std::printf("%s: \"%s\" [%s]  train_loss=%.5f%s  computation time: %.0fms  architecture=%s\n",
                    rank_label.c_str(), r.name.c_str(), r.method.c_str(), r.best_loss, val_str.c_str(),
                    r.total_time_ms, r.architecture().c_str());
    }
    std::printf("\n");
}
