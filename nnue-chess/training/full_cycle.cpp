// Full training cycle: generates Stockfish-evaluated random legal positions
// using the toMateTo engine's own board representation and move generator,
// benchmarks the feature-transformer accumulator, trains an NNUE, and
// writes out both the dataset and the trained network under one shared
// name.
//
// Runtime flags:
//   --depth N     Stockfish search depth per generated position
//   --samples N   total number of positions to generate AND train on
//   --name NAME   names both <NAME>.dataset and <NAME>.nnue
//   --report      print the results leaderboard and exit (no training)
//
// Every run appends its (name, architecture, depth, samples, best/final
// loss, wall-clock time) to training_results.csv in the current working
// directory, and prints a "Platz 1 / Platz 2 / ..." ranking (best loss
// first) both at the end of that run and on demand via --report.
//
// Architecture (accumulator + hidden-layer widths) is fixed at COMPILE
// time via the NNUE_ACC_SIZE/NNUE_H1/NNUE_H2/NNUE_H3 preprocessor defines
// (set by CMakeLists.txt's matching cache variables) - a C++ template's
// size parameters can't come from a runtime value. Use ../train.sh, which
// reconfigures and rebuilds this target for you whenever you pass a
// different --arch, so day to day it still feels like an ordinary flag; or
// pass -DNNUE_ACC_SIZE=... etc. to cmake yourself. --arch is also accepted
// here directly, purely to sanity-check against whatever this particular
// binary was actually compiled for.
//
// POSIX/Linux only: reuses toMateTo_engine's move generator (bitboards,
// no OS dependency) but talks to Stockfish via fork+exec (see
// uci_engine.hpp), the same approach testing/stockfish_perft.cpp already
// uses in this repo.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "uci_engine.hpp"
#include "train_network.hpp"
#include "leaderboard.hpp"
#include "efficiency_check.hpp"
#include "validation.hpp"

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"

namespace fs = std::filesystem;

#ifndef NNUE_ACC_SIZE
#define NNUE_ACC_SIZE 256
#endif
#ifndef NNUE_H1
#define NNUE_H1 128
#endif
#ifndef NNUE_H2
#define NNUE_H2 32
#endif
#ifndef NNUE_H3
#define NNUE_H3 32
#endif

using Net = nnue::train::TrainNetwork<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>;
using Inference = nnue::NNUE<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>;

namespace {

struct Args {
    int depth = 10;
    long samples = 20000;
    std::string name = "run";
    bool report_only = false;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
                  "Usage: %s --depth N --samples N --name NAME [--arch ACC,H1,H2,H3]\n"
                  "       %s --report\n"
                  "  --depth    Stockfish search depth per generated position\n"
                  "  --samples  total number of positions to generate AND train on\n"
                  "  --name     names both <NAME>.dataset and <NAME>.nnue, and the run's\n"
                  "             entry in the results leaderboard (training_results.csv)\n"
                  "  --arch     sanity-checked, not applied: this binary was compiled for\n"
                  "             (%d,%d,%d,%d); to actually change it use ../train.sh or\n"
                  "             recompile with -DNNUE_ACC_SIZE=.. -DNNUE_H1=.. -DNNUE_H2=..\n"
                  "             -DNNUE_H3=..\n"
                  "  --report   print the results leaderboard (every run ever recorded in\n"
                  "             training_results.csv, best loss first) and exit - no training\n",
                  prog, prog, NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--depth") {
            a.depth = std::stoi(next());
        } else if (arg == "--samples") {
            a.samples = std::stol(next());
        } else if (arg == "--name") {
            a.name = next();
        } else if (arg == "--arch") {
            std::string spec = next();
            unsigned acc, h1, h2, h3;
            if (std::sscanf(spec.c_str(), "%u,%u,%u,%u", &acc, &h1, &h2, &h3) == 4) {
                if (int(acc) != NNUE_ACC_SIZE || int(h1) != NNUE_H1 || int(h2) != NNUE_H2 || int(h3) != NNUE_H3) {
                    std::fprintf(stderr,
                                 "WARNING: --arch %s does not match the architecture this binary was\n"
                                 "compiled for (%d,%d,%d,%d). This run WILL train a (%d,%d,%d,%d)\n"
                                 "network regardless - use ../train.sh to actually change it.\n",
                                 spec.c_str(), NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3, NNUE_ACC_SIZE, NNUE_H1,
                                 NNUE_H2, NNUE_H3);
                }
            }
        } else if (arg == "--report") {
            a.report_only = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return a;
}

// Searches upward from the current working directory for a
// "testing/stockfish_bin" folder (the toMateTo repo's convention) and
// returns the first file under it that looks like a Stockfish binary.
std::string find_stockfish() {
    fs::path dir = fs::current_path();
    for (int hops = 0; hops < 6; ++hops) {
        fs::path candidate = dir / "testing" / "stockfish_bin";
        if (fs::exists(candidate)) {
            for (auto& entry : fs::recursive_directory_iterator(candidate)) {
                if (!entry.is_regular_file()) continue;
                std::string name = entry.path().filename().string();
                if (name.rfind("stockfish", 0) == 0 && name.find('.') == std::string::npos) {
                    fs::permissions(entry.path(),
                                     fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                                     fs::perm_options::add);
                    return entry.path().string();
                }
            }
        }
        if (!dir.has_parent_path() || dir == dir.parent_path()) break;
        dir = dir.parent_path();
    }
    throw std::runtime_error(
        "could not find a Stockfish binary under any 'testing/stockfish_bin' directory "
        "above the current working directory - run this from somewhere inside the toMateTo repo");
}

// ---- board -> NNUE feature extraction ----------------------------------
// Reads pieces directly out of the toMateTo chess_board's bitboards - no
// FEN round trip needed since we already have the live board object.

void collect_features(const chess_board& board, nnue::Color perspective, int king_sq, std::vector<int>& out) {
    auto scan_side = [&](const one_side& side, nnue::Color color) {
        auto scan = [&](Bitboard bb, nnue::PieceType pt) {
            while (bb) {
                int sq = __builtin_ctzll(bb);
                bb &= bb - 1;
                out.push_back(nnue::feature_index(perspective, sq, pt, color, king_sq));
            }
        };
        scan(side.pawns, nnue::PAWN);
        scan(side.knights, nnue::KNIGHT);
        scan(side.bishop, nnue::BISHOP);
        scan(side.rooks, nnue::ROOK);
        scan(side.queen, nnue::QUEEN);
    };
    scan_side(board.white, nnue::WHITE);
    scan_side(board.black, nnue::BLACK);
}

nnue::train::TrainingSample sample_from_board(const chess_board& board, int stockfish_cp) {
    nnue::train::TrainingSample s;
    int white_king = __builtin_ctzll(board.white.king);
    int black_king = __builtin_ctzll(board.black.king);
    collect_features(board, nnue::WHITE, white_king, s.white_features);
    collect_features(board, nnue::BLACK, black_king, s.black_features);
    s.side_to_move = board.whites_turn ? nnue::WHITE : nnue::BLACK;
    s.target_wdl = nnue::train::sigmoid(float(stockfish_cp) / nnue::train::CP_SCALE);
    return s;
}

// ---- random legal position generation ----------------------------------
// Plays a random number of random legal plies from the start position.
// Restarts from scratch whenever a walk runs into checkmate/stalemate
// before reaching its target ply count - simple and correct, if
// occasionally wasteful; move generation is cheap next to the Stockfish
// call that follows each accepted position.
chess_board random_position(std::mt19937& rng) {
    std::uniform_int_distribution<int> ply_dist(0, 60);
    for (;;) {
        // chess_board{} (not chess_board board;): setup_chess_board() only
        // fills in white/black/whites_turn - castling_rights, ep_square and
        // the move counters are plain uninitialized members otherwise, so a
        // stack-allocated board here would start from whatever garbage the
        // previous loop iteration (or an unrelated earlier stack frame)
        // left behind. {} zero-initializes everything first; the explicit
        // assignments below then set the three fields a fresh start
        // position actually needs that setup_chess_board() doesn't touch.
        chess_board board{};
        board.setup_chess_board();
        board.castling_rights = ANY_CASTLING;
        board.ep_square = SQ_NONE;
        board.halve_move_counter = 0;
        board.full_move_counter = 1;
        int target_plies = ply_dist(rng);
        bool dead_end = false;
        for (int ply = 0; ply < target_plies; ++ply) {
            MoveStacks ms;
            find_all_moves(&ms, &board);
            int total = ms.normal_size() + ms.capture_size();
            if (total == 0) {
                dead_end = true;
                break;
            }
            std::uniform_int_distribution<int> move_dist(0, total - 1);
            int idx = move_dist(rng);
            Move mv = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];
            StateInfo st;
            make_move(&board, mv, st);
        }
        if (dead_end) continue;
        // The final position itself must have a legal move - otherwise
        // there is nothing meaningful for Stockfish (or the network) to
        // evaluate about it.
        MoveStacks ms;
        find_all_moves(&ms, &board);
        if (ms.empty()) continue;
        return board;
    }
}

void init_engine_tables() {
    init_knight_table();
    init_magic_rook_or_bishop("rook");
    init_magic_rook_or_bishop("bishop");
    init_squares_in_between_table();
    init_square_on_the_line_table();
    init_king_mask();
    init_pinned_tables_rook_and_bishop();
    init_pawn_attack_lookup();
    init_direction_rays();
    init_rows();
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.report_only) {
        print_leaderboard();
        return 0;
    }

    std::printf("architecture: NNUE<%d,%d,%d,%d> (fixed at compile time)\n", NNUE_ACC_SIZE, NNUE_H1, NNUE_H2,
                NNUE_H3);
    std::printf("depth=%d samples=%ld name=\"%s\"\n\n", args.depth, args.samples, args.name.c_str());

    run_efficiency_check<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>();

    init_engine_tables();
    std::string sf_path = find_stockfish();
    std::printf("using stockfish: %s\n", sf_path.c_str());
    UciEngine engine(sf_path);

    const std::string dataset_path = args.name + ".dataset";
    std::ofstream dataset_file(dataset_path);
    if (!dataset_file) throw std::runtime_error("could not open " + dataset_path + " for writing");

    std::mt19937 rng(std::random_device{}());
    Net net(/*lr=*/1e-4f, /*ft_weight_decay=*/1e-6f, /*hidden_weight_decay=*/1e-4f);

    std::printf("== generating %ld Stockfish-evaluated positions (depth %d) and training on them ==\n",
                args.samples, args.depth);
    auto gen_start = std::chrono::steady_clock::now();
    float running_loss = 0.0f;
    float best_window_loss = std::numeric_limits<float>::infinity();
    float last_window_loss = 0.0f;
    const long report_every = std::max<long>(1, args.samples / 20);
    for (long i = 0; i < args.samples; ++i) {
        chess_board board = random_position(rng);
        std::string fen = board_to_fen(board);
        int cp = engine.evaluate(fen, args.depth);
        dataset_file << fen << ';' << cp << ';' << (board.whites_turn ? 'w' : 'b') << '\n';

        nnue::train::TrainingSample sample = sample_from_board(board, cp);
        running_loss += net.train_step(sample);

        if ((i + 1) % report_every == 0 || i + 1 == args.samples) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - gen_start).count();
            double per_sample = elapsed / double(i + 1);
            double eta = per_sample * double(args.samples - (i + 1));
            float window_loss = running_loss / float((i + 1) % report_every == 0 ? report_every : (i + 1) % report_every);
            best_window_loss = std::min(best_window_loss, window_loss);
            last_window_loss = window_loss;
            std::printf("[%6ld/%6ld] avg loss=%.5f  %.1f ms/sample  ETA %.0fs\n", i + 1, args.samples, window_loss,
                        per_sample * 1000.0, eta);
            running_loss = 0.0f;
        }
    }
    dataset_file.close();
    std::printf("dataset written to %s\n", dataset_path.c_str());

    net.finalize_training();
    auto training_end = std::chrono::steady_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(training_end - gen_start).count();

    auto quantized = Inference::make();
    net.quantize_into(*quantized);
    const std::string net_path = args.name + ".nnue";
    quantized->save(net_path);
    std::printf("network written to %s\n", net_path.c_str());

    // Held-out validation: for full_cycle this is mostly a sanity check
    // (train/held-out gap), since the training loss is already Stockfish-
    // anchored - but computing it the same way self_play.cpp does keeps
    // both methods' leaderboard entries on one directly comparable scale.
    std::printf("== validation against a fresh Stockfish-labeled held-out set ==\n");
    constexpr int kValidationPositions = 200;
    auto gen_fen_and_cp = [&]() -> std::pair<std::string, int> {
        chess_board vboard = random_position(rng);
        std::string vfen = board_to_fen(vboard);
        int vcp = engine.evaluate(vfen, args.depth);
        return {vfen, vcp};
    };
    auto raw_eval_for_fen = [&](const std::string& fen) -> float {
        chess_board vboard{};
        setup_fen_position(vboard, fen);
        nnue::train::TrainingSample s = sample_from_board(vboard, 0);
        typename Net::Cache cache;
        net.forward(s, cache);
        return cache.out_pre;
    };
    ValidationResult validation = validate_against_stockfish(kValidationPositions, gen_fen_and_cp, raw_eval_for_fen);
    std::printf("  correlation=%.4f  calibrated_prob_mse=%.5f  (fit: scale=%.3f offset=%.3f)\n\n",
                validation.correlation, validation.calibrated_prob_mse, validation.scale, validation.offset);

    TrainingResult result;
    result.name = args.name;
    result.method = "stockfish";
    result.timestamp = current_timestamp();
    result.acc = NNUE_ACC_SIZE;
    result.h1 = NNUE_H1;
    result.h2 = NNUE_H2;
    result.h3 = NNUE_H3;
    result.depth = args.depth;
    result.samples = args.samples;
    result.best_loss = best_window_loss;
    result.final_loss = last_window_loss;
    result.total_time_ms = total_time_ms;
    result.validation_loss = validation.calibrated_prob_mse;
    result.validation_correlation = validation.correlation;
    append_result(result);
    std::printf("result recorded in %s\n", results_csv_path().c_str());

    print_leaderboard();

    return 0;
}
