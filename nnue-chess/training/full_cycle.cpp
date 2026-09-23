// Full training cycle: generates Stockfish-evaluated random legal positions
// using the toMateTo engine's own board representation and move generator,
// benchmarks the feature-transformer accumulator, trains an NNUE, and
// writes out both the dataset and the trained network under one shared
// name.
//
// Runtime flags:
//   --depth N     Stockfish search depth per generated position
//   --samples N   total number of positions to generate AND train on
//   --threads N   parallel worker threads generating+evaluating positions;
//                 default: all logical cores (see the "multithreading"
//                 section below)
//   --name NAME   names both <NAME>.dataset and <NAME>.nnue
//   --report      print the results leaderboard and exit (no training)
//
// Multithreading: generating a training position (random walk + a Stockfish
// call) is completely independent from generating any other one, so that
// part parallelizes trivially across --threads worker threads, each with
// its own Stockfish subprocess (forced to 1 internal thread each - see
// uci_engine.hpp - so N workers don't oversubscribe cores N times over).
// Training itself (net.train_step()) stays on the main thread: it mutates
// one shared model's Adam optimizer state (including the lazy catch-up
// bookkeeping in sparse_optimizer.hpp, which assumes a single strictly-
// increasing step counter), so making it concurrent would need a much more
// involved design (e.g. a lock per feature row, or accepting slightly stale/
// racy updates) for comparatively little benefit - training a single sample
// takes low tens of microseconds (see the efficiency check), several
// orders of magnitude faster than the Stockfish call that produced it, so
// one consumer thread never becomes the bottleneck even with many producers
// feeding it.
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
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "uci_engine.hpp"
#include "train_network.hpp"
#include "leaderboard.hpp"
#include "efficiency_check.hpp"
#include "validation.hpp"
#include "eval_sanity_check.hpp"
#include "progress_bar.hpp"

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

enum class Mode { Full, GenData, OnlyTrain };

struct Args {
    Mode mode = Mode::Full;
    int depth = 10;
    long samples = 20000;
    int threads = 0; // 0 = resolved to hardware_concurrency() in main()
    long batch_size = 0; // 0 = resolved to threads*64 in main() (only_train's parallel-gradient batch size)
    int epochs = 1;      // only_train: passes over the dataset file
    std::string dataset; // only_train: input dataset path (required in that mode)
    float lr = 1e-4f;
    float lr_half_life = 100000.0f;
    std::string name = "run";
    bool report_only = false;
    // only_train: train only on positions with piece_min <= piece count <=
    // piece_max (both inclusive, whole-board count including both kings -
    // same convention as the "Datensatz-Statistiken" block). Default is the
    // full range (no filtering) - see the README's bucket-NNUE section for
    // why you'd narrow this: a net trained only on e.g. 2-15 pieces can
    // specialize on endgames instead of splitting capacity across every
    // game phase.
    int piece_min = 2;
    int piece_max = 32;
    // Held-out positions used for the final "validation against a fresh
    // Stockfish-labeled held-out set" step (both full and only_train). Was
    // a hardcoded 200 - too small a sample for val_loss/correlation to be
    // trustworthy run-to-run (each run draws a FRESH random set, so two
    // runs' numbers are already comparing different samples on top of
    // whatever real difference exists between the nets); default raised to
    // 1000, still overridable for a quicker/rougher check.
    int validation_positions = 1000;
};

void print_usage(const char* prog) {
    std::fprintf(
        stderr,
        "Usage: %s [--mode full|gen_data|only_train] --name NAME [options] [--arch ACC,H1,H2,H3]\n"
        "       %s --report\n"
        "  --mode         full (default): generate + train in one pass, like before.\n"
        "                 gen_data: generate <NAME>.dataset only, no network at all.\n"
        "                 only_train: skip Stockfish/generation entirely, train on an\n"
        "                 existing dataset (--dataset) - see the README's \"Ausführung\" section\n"
        "  --depth        Stockfish search depth per generated position (full, gen_data)\n"
        "  --samples      total number of positions to generate AND/OR train on (full, gen_data)\n"
        "  --dataset      path to a <NAME>.dataset file to train on (only_train, required)\n"
        "  --epochs       passes over --dataset (only_train); default 1\n"
        "  --threads      parallel worker threads; default: all logical cores.\n"
        "                 full/gen_data: each thread runs its own Stockfish process - this is\n"
        "                 where nearly all the wall-clock time goes, so this is what matters.\n"
        "                 only_train: threads compute gradients for different positions of the\n"
        "                 same batch in parallel (network weights are read-only during that\n"
        "                 phase); one Adam step is then applied per batch, sequentially - see\n"
        "                 train_network.hpp's compute_gradients()/apply_gradients() comment for\n"
        "                 why this split, not e.g. one thread per weight\n"
        "  --batch-size   only_train: positions per parallel-gradient batch; default threads*64\n"
        "  --piece-min    only_train: skip positions with fewer than this many pieces on the\n"
        "                 board (2..32, both kings counted); default 2 (no filtering). Use with\n"
        "                 --piece-max to train a bucket-specialized net on one game-phase slice\n"
        "                 of --dataset instead of the whole thing - see the README's bucket-NNUE\n"
        "                 section, and strength_match's --net-buckets for using several such\n"
        "                 nets together in search\n"
        "  --piece-max    only_train: skip positions with more than this many pieces; default 32\n"
        "  --validation-positions  size of the held-out, freshly-Stockfish-labeled set used for\n"
        "                 the final val_loss/correlation numbers (full, only_train); default 1000.\n"
        "                 Larger = more trustworthy comparison across runs, but each position is\n"
        "                 one more real Stockfish call at --depth, so also slower\n"
        "  --lr           base learning rate; default 1e-4\n"
        "  --lr-half-life steps until the learning rate has halved (lr/(1+step/half_life),\n"
        "                 never reaches 0 - see sparse_optimizer.hpp's lr_schedule());\n"
        "                 default 100000. Bigger = slower/gentler decay - if in doubt,\n"
        "                 prefer too slow (just train longer) over too fast (can't undo)\n"
        "  --name         names <NAME>.dataset and/or <NAME>.nnue (mode-dependent) and the\n"
        "                 run's entry in the results leaderboard (training_results.csv)\n"
        "  --arch         sanity-checked, not applied: this binary was compiled for\n"
        "                 (%d,%d,%d,%d); to actually change it use ../train.sh or\n"
        "                 recompile with -DNNUE_ACC_SIZE=.. -DNNUE_H1=.. -DNNUE_H2=..\n"
        "                 -DNNUE_H3=..\n"
        "  --report       print the results leaderboard (every run ever recorded in\n"
        "                 training_results.csv, best loss first) and exit - no training\n",
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
        if (arg == "--mode") {
            std::string mode = next();
            if (mode == "full") a.mode = Mode::Full;
            else if (mode == "gen_data") a.mode = Mode::GenData;
            else if (mode == "only_train") a.mode = Mode::OnlyTrain;
            else {
                std::fprintf(stderr, "Unknown --mode: %s (expected 'full', 'gen_data' or 'only_train')\n",
                             mode.c_str());
                std::exit(1);
            }
        } else if (arg == "--depth") {
            a.depth = std::stoi(next());
        } else if (arg == "--samples") {
            a.samples = std::stol(next());
        } else if (arg == "--dataset") {
            a.dataset = next();
        } else if (arg == "--epochs") {
            a.epochs = std::stoi(next());
        } else if (arg == "--threads") {
            a.threads = std::stoi(next());
        } else if (arg == "--batch-size") {
            a.batch_size = std::stol(next());
        } else if (arg == "--piece-min") {
            a.piece_min = std::stoi(next());
        } else if (arg == "--piece-max") {
            a.piece_max = std::stoi(next());
        } else if (arg == "--validation-positions") {
            a.validation_positions = std::stoi(next());
        } else if (arg == "--lr") {
            a.lr = std::stof(next());
        } else if (arg == "--lr-half-life") {
            a.lr_half_life = std::stof(next());
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
    if (a.mode == Mode::OnlyTrain && a.dataset.empty()) {
        std::fprintf(stderr, "--mode only_train requires --dataset PATH\n");
        print_usage(argv[0]);
        std::exit(1);
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
//
// Two knobs matter for what game PHASES actually end up in the resulting
// dataset - found the hard way (see README): the original version used
// ply_dist(0,60) with plain uniform move selection (never preferring
// captures), and a piece-count analysis of the datasets it produced showed
// >99.9% of positions still had 16+ of the 32 starting pieces on the board -
// essentially zero real endgame coverage, no matter how many samples you
// generated, because uniform-random moves almost never trade pieces off (a
// captures needs BOTH a capturing move to exist AND to get picked, and it's
// usually a small minority of the legal move list). A net trained on that
// then has no idea what a real endgame looks like, which shows up exactly
// as "loses every long game" - long games are the ones that reach reduced
// material, i.e. the one regime the dataset never covered.
//
// Fix: bias move selection toward captures when available (kCaptureBias),
// and widen the ply range (kMaxPlies) so walks can actually run long enough
// for that bias to matter. Together these produce a real spread from
// opening through deep endgame - verify with the "Datensatz-Statistiken"
// block main() prints after generation (piece-count histogram) rather than
// assuming - a distribution is easy to get subtly wrong and hard to eyeball
// from a few sample FENs.
constexpr int kMaxPlies = 140;
constexpr float kCaptureBias = 0.25f; // fraction of plies that prefer a capture when one exists

// chess_board{} (not chess_board board;): setup_chess_board() only fills in
// white/black/whites_turn - castling_rights, ep_square and the move
// counters are plain uninitialized members otherwise, so a stack-allocated
// board would start from whatever garbage an earlier stack frame left
// behind without this. {} zero-initializes everything first; the explicit
// assignments then set the three fields a fresh start position actually
// needs that setup_chess_board() doesn't touch.
void reset_to_start_position(chess_board& board) {
    board = chess_board{};
    board.setup_chess_board();
    board.castling_rights = ANY_CASTLING;
    board.ep_square = SQ_NONE;
    board.halve_move_counter = 0;
    board.full_move_counter = 1;
}

inline int piece_count_of(const chess_board& board) {
    return __builtin_popcountll(board.white.side_all | board.black.side_all);
}

chess_board random_position(std::mt19937& rng) {
    std::uniform_int_distribution<int> ply_dist(0, kMaxPlies);
    std::uniform_real_distribution<float> bias_roll(0.0f, 1.0f);
    for (;;) {
        chess_board board;
        reset_to_start_position(board);
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
            int idx;
            if (ms.capture_size() > 0 && bias_roll(rng) < kCaptureBias) {
                std::uniform_int_distribution<int> cap_dist(0, ms.capture_size() - 1);
                idx = ms.normal_size() + cap_dist(rng);
            } else {
                std::uniform_int_distribution<int> move_dist(0, total - 1);
                idx = move_dist(rng);
            }
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

// ---- multithreaded position generation -----------------------------------
// One item = one Stockfish-evaluated position, fully formed (both the
// TrainingSample for training and the fen/cp/stm for the dataset file) so
// the consumer (main thread) never has to touch the board/engine again.
struct GeneratedItem {
    nnue::train::TrainingSample sample;
    std::string fen;
    int cp = 0;
    char stm = 'w';
};

// Simple bounded mutex+condvar queue. Bounded (not unbounded) so a slow
// consumer can't let producers race arbitrarily far ahead and blow up
// memory - in practice the consumer (one train_step() call, low tens of
// microseconds) is so much faster than any producer (one Stockfish call,
// milliseconds+) that the queue should rarely if ever actually fill up;
// the cap is a safety net, not an expected steady state.
class WorkQueue {
public:
    explicit WorkQueue(std::size_t max_size) : max_size_(max_size) {}

    void push(GeneratedItem item) {
        std::unique_lock<std::mutex> lock(m_);
        not_full_.wait(lock, [&] { return items_.size() < max_size_ || closed_; });
        items_.push_back(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
    }

    // Returns false once the queue is closed AND drained - the consumer's
    // signal to stop.
    bool pop(GeneratedItem& out) {
        std::unique_lock<std::mutex> lock(m_);
        not_empty_.wait(lock, [&] { return !items_.empty() || closed_; });
        if (items_.empty()) return false; // closed and drained
        out = std::move(items_.front());
        items_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }

    // Called once every producer thread has finished (no more pushes will
    // happen) - lets pop() return false once the last queued item is taken
    // instead of blocking forever.
    void close() {
        {
            std::lock_guard<std::mutex> lock(m_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::deque<GeneratedItem> items_;
    std::size_t max_size_;
    bool closed_ = false;
};

// How many pieces (both colors, both kings included) a generated position
// may have - 2 (bare kings) to 32 (nothing captured yet) - and how many
// distinct counts that spans. Used to give every piece count an equal
// --samples/kNumPieceBuckets quota during generation (see producer_worker())
// instead of leaving the resulting mix wherever a random walk's move-
// selection happens to land it - see this file's "Datensatz-Statistiken"
// block and the README for why an *explicit* quota, not just capture-bias
// tuning, is what actually gets a flat distribution rather than just a less
// skewed one.
constexpr int kMinPieces = 2;
constexpr int kMaxPieces = 32;
constexpr int kNumPieceBuckets = kMaxPieces - kMinPieces + 1; // 31
constexpr int kWalkMaxPlies = 400;
// Rough estimate of how many plies it takes to see one more capture -
// roughly 1/kCaptureBias, a bit generous since captures aren't always
// *available* even when preferred (especially early on, before pieces
// come into contact). Used only to aim a walk's length at a piece count
// that still needs samples (see pick_needy_bucket()) - it doesn't need to
// be accurate, just roughly in the right ballpark, since wherever a walk
// actually lands still goes through the same bucket-quota check below.
constexpr float kPliesPerCapture = 4.0f;

int estimate_plies_for_piece_count(int target_pieces) {
    return int(float(kMaxPieces - target_pieces) * kPliesPerCapture);
}

// +/- range added to a walk-length estimate, so repeated attempts at the
// same target spread out instead of all taking the identical path. Scales
// with the estimate itself, not a flat constant: a flat +/-15 completely
// swamps small estimates (e.g. base=0 for "32 pieces, no captures yet" -
// +/-15 plies of jitter there means the walk usually captures at least
// once before stopping, missing 32 almost every time) while barely
// mattering for large ones. Found this the hard way: with a flat jitter,
// generation got stuck at 87% (2700/3100) twice in a row, always the same
// handful of near-32-piece buckets never filling.
int ply_jitter_for(int base_plies) { return std::max(2, base_plies / 3); }

// Lower bound on target_plies regardless of the estimate/jitter above.
// Without this, buckets right next to kMaxPieces (whose base estimate is
// near 0, since "no captures yet" needs ~0 plies) collapse: jitter clamped
// to >=0 means a walk aimed at 32 pieces takes 0 plies most of the time,
// which is the literal unmoved start position every single time - not a
// diverse *sample* of near-full-board positions, the same one repeated.
// Confirmed at production scale: 20%+ of all "32 pieces" rows in a 100M
// run were the exact start FEN before this floor existed. 6 plies is
// small enough that most such walks still land back on 32 pieces (real
// captures are rare in the first few plies of quasi-random play, well
// below the nominal kCaptureBias since ms.capture_size() is usually 0
// this early), while giving genuine move-order variety instead of none.
constexpr int kMinWalkPlies = 6;

// Picks a random still-under-quota bucket (uniform among all of them, not
// weighted - every worker thread calls this independently and they should
// spread across whatever's needed, not all pile onto the single neediest
// one). Returns -1 if every bucket is already at quota (shouldn't normally
// happen - target_per_bucket is rounded up to give every bucket enough
// headroom to reach --samples - but kept as a defined fallback).
int pick_needy_bucket(std::vector<std::atomic<long>>& bucket_counts, long target_per_bucket, std::mt19937& rng) {
    thread_local std::vector<int> needy;
    needy.clear();
    for (int b = 0; b < kNumPieceBuckets; ++b)
        if (bucket_counts[std::size_t(b)].load(std::memory_order_relaxed) < target_per_bucket) needy.push_back(b);
    if (needy.empty()) return -1;
    std::uniform_int_distribution<std::size_t> pick(0, needy.size() - 1);
    return needy[pick(rng)];
}

// One worker thread's whole job: repeatedly (1) pick a piece-count bucket
// that still needs samples, (2) walk a fresh capture-biased random game
// (reset_to_start_position() + kCaptureBias) for roughly the number of
// plies estimate_plies_for_piece_count() thinks reaches that material
// level, (3) evaluate+push whatever position it actually landed on - which
// goes through the SAME bucket-quota check regardless of which bucket it
// was aiming for, so an inaccurate estimate never breaks correctness, only
// wastes a (cheap - it's move generation, not a Stockfish call) attempt.
//
// Why aim walks instead of just sampling every ply of one long walk (the
// first version of this): tried that, and it doesn't actually produce a
// flat distribution - it produces one that's flatTER than the original
// (still much better than nothing), but the *rare* piece counts get
// starved for a structural reason: a long walk spends many more of its
// plies at whatever material level captures happen to leave it sitting at
// for a while, and very little time at the two extremes (32 pieces only
// exists for the first few plies of a walk, before any capture; a specific
// low count like 2 is a brief passing moment too) - so common counts hit
// their quota fast while 29-32 and the very bottom stay stuck near zero,
// and once the *global* emitted-count target is reached from the common
// buckets alone, generation stops before the rare ones ever catch up.
// Concretely: a 3100-sample test run of that version produced zero
// positions above 28 pieces. Aiming each walk explicitly at whatever's
// currently short doesn't have that failure mode - every bucket gets
// actively sought out, not just passively hoped for.
void producer_worker(int thread_id, std::atomic<long>& emitted, long total_samples, long target_per_bucket, int depth,
                      const std::string& stockfish_path, WorkQueue& queue,
                      std::vector<std::atomic<long>>& bucket_counts) {
    std::mt19937 rng(std::random_device{}() ^ (0x9e3779b9u * std::uint32_t(thread_id + 1)));
    std::uniform_real_distribution<float> bias_roll(0.0f, 1.0f);
    UciEngine engine(stockfish_path);

    while (emitted.load(std::memory_order_relaxed) < total_samples) {
        const int target_bucket = pick_needy_bucket(bucket_counts, target_per_bucket, rng);
        int target_plies;
        if (target_bucket < 0) {
            target_plies = std::uniform_int_distribution<int>(kMinWalkPlies, kWalkMaxPlies)(rng); // fallback - see pick_needy_bucket()
        } else {
            const int base = estimate_plies_for_piece_count(target_bucket + kMinPieces);
            const int half_range = ply_jitter_for(base);
            std::uniform_int_distribution<int> jitter(-half_range, half_range);
            target_plies = std::max(kMinWalkPlies, std::min(kWalkMaxPlies, base + jitter(rng)));
        }

        chess_board board;
        reset_to_start_position(board);
        bool dead_end = false;
        for (int ply = 0; ply < target_plies; ++ply) {
            MoveStacks ms;
            find_all_moves(&ms, &board);
            const int total = ms.normal_size() + ms.capture_size();
            if (total == 0) {
                dead_end = true;
                break;
            }
            int idx;
            if (ms.capture_size() > 0 && bias_roll(rng) < kCaptureBias) {
                std::uniform_int_distribution<int> cap_dist(0, ms.capture_size() - 1);
                idx = ms.normal_size() + cap_dist(rng);
            } else {
                std::uniform_int_distribution<int> move_dist(0, total - 1);
                idx = move_dist(rng);
            }
            Move mv = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];
            StateInfo st;
            make_move(&board, mv, st);
        }
        if (dead_end) continue;

        MoveStacks ms;
        find_all_moves(&ms, &board);
        if (ms.empty()) continue; // landed on checkmate/stalemate - nothing to evaluate

        const int bucket = piece_count_of(board) - kMinPieces;
        bool accept = true;
        if (bucket >= 0 && bucket < kNumPieceBuckets) {
            const long before = bucket_counts[std::size_t(bucket)].fetch_add(1, std::memory_order_relaxed);
            accept = before < target_per_bucket;
            if (!accept) bucket_counts[std::size_t(bucket)].fetch_sub(1, std::memory_order_relaxed); // full - give the slot back
        }
        if (!accept) continue;

        std::string fen = board_to_fen(board);
        int cp = engine.evaluate(fen, depth);
        GeneratedItem item;
        item.sample = sample_from_board(board, cp);
        item.fen = std::move(fen);
        item.cp = cp;
        item.stm = board.whites_turn ? 'w' : 'b';
        queue.push(std::move(item));
        emitted.fetch_add(1, std::memory_order_relaxed);
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

// ---- dataset statistics (game-phase / eval coverage) ----------------------
// Printed after generation (--mode full/gen_data) so a skewed dataset - the
// original random_position() produced almost no endgame coverage at all,
// see that function's comment - is visible immediately instead of only
// showing up later as "the net loses every long game" once something has
// already been trained on it.

int count_pieces_in_fen(const std::string& fen) {
    int n = 0;
    for (char c : fen) {
        if (c == ' ') break; // only the piece-placement field
        if (std::isalpha(static_cast<unsigned char>(c))) ++n;
    }
    return n;
}

struct DatasetStats {
    std::vector<int> piece_counts;
    std::vector<int> cps;
    long white_to_move = 0;

    void record(const std::string& fen, int cp, char stm) {
        piece_counts.push_back(count_pieces_in_fen(fen));
        cps.push_back(cp);
        if (stm == 'w') ++white_to_move;
    }
};

void print_dataset_stats(DatasetStats& s) {
    if (s.piece_counts.empty()) return;
    std::sort(s.piece_counts.begin(), s.piece_counts.end());
    std::sort(s.cps.begin(), s.cps.end());
    const std::size_t n = s.piece_counts.size();
    auto pc_pct = [&](double p) { return s.piece_counts[std::min(n - 1, std::size_t(p * double(n)))]; };
    auto cp_pct = [&](double p) { return s.cps[std::min(n - 1, std::size_t(p * double(n)))]; };

    long opening = 0, midgame = 0, endgame = 0, deep_endgame = 0;
    for (int c : s.piece_counts) {
        if (c >= 28) ++opening;
        else if (c >= 16) ++midgame;
        else if (c >= 8) ++endgame;
        else ++deep_endgame;
    }
    auto pct_of = [&](long x) { return 100.0 * double(x) / double(n); };

    std::printf("\n== Datensatz-Statistiken (%zu Stellungen) ==\n", n);
    std::printf("Figuren pro Stellung: min=%d  p10=%d  p25=%d  p50=%d  p75=%d  p90=%d  max=%d\n",
                s.piece_counts.front(), pc_pct(0.10), pc_pct(0.25), pc_pct(0.50), pc_pct(0.75), pc_pct(0.90),
                s.piece_counts.back());
    std::printf("  Eroeffnung      (28-32 Figuren): %6ld  (%.1f%%)\n", opening, pct_of(opening));
    std::printf("  Mittelspiel     (16-27 Figuren): %6ld  (%.1f%%)\n", midgame, pct_of(midgame));
    std::printf("  Endspiel        ( 8-15 Figuren): %6ld  (%.1f%%)\n", endgame, pct_of(endgame));
    std::printf("  Tiefes Endspiel ( 2- 7 Figuren): %6ld  (%.1f%%)\n", deep_endgame, pct_of(deep_endgame));
    std::printf("Stockfish-Eval (cp): min=%d  p10=%d  p50=%d  p90=%d  max=%d\n", s.cps.front(), cp_pct(0.10),
                cp_pct(0.50), cp_pct(0.90), s.cps.back());
    std::printf("Seite am Zug: weiss %.1f%%  schwarz %.1f%%\n\n", pct_of(s.white_to_move),
                100.0 - pct_of(s.white_to_move));
}

// ---- --mode only_train: multithreaded gradient computation ----------------
// Parses one "fen;cp;stm" line (the format both full_cycle and self_play
// write their .dataset files in) into a FEN + cp pair. The board itself
// isn't reconstructed here - that happens per-thread in run_only_train(),
// since setup_fen_position() populates a caller-supplied chess_board and
// each worker thread needs its own (thread-unsafe to share one).
bool parse_dataset_line(const std::string& line, std::string& fen, int& cp) {
    if (line.empty()) return false;
    const auto p1 = line.find(';');
    if (p1 == std::string::npos) return false;
    const auto p2 = line.find(';', p1 + 1);
    if (p2 == std::string::npos) return false;
    fen = line.substr(0, p1);
    cp = std::stoi(line.substr(p1 + 1, p2 - p1 - 1));
    return true;
}

// best/final loss across the run, for the leaderboard - windowed the same
// way the Full-mode generation loop below does (one point per batch here,
// since a batch already IS the natural unit of work for --mode only_train).
struct TrainStats {
    float best_loss = std::numeric_limits<float>::infinity();
    float final_loss = 0.0f;
};

// Trains on an already-generated dataset file, no Stockfish involved at
// all. This is where multithreaded training actually pays off (see this
// file's header comment): with no per-sample Stockfish call to hide behind,
// gradient computation - forward + backward pass through the network,
// read-only w.r.t. its weights (see train_network.hpp's Gradients/
// compute_gradients()/apply_gradients() split) - is the whole cost, and it
// parallelizes the same way the Stockfish calls above do: independent work
// per position, merged into one sequential Adam step per batch.
TrainStats run_only_train(const Args& args, Net& net) {
    std::ifstream in(args.dataset);
    if (!in) throw std::runtime_error("could not open --dataset file: " + args.dataset);

    std::vector<std::pair<std::string, int>> positions; // (fen, cp)
    std::string line;
    while (std::getline(in, line)) {
        std::string fen;
        int cp = 0;
        if (parse_dataset_line(line, fen, cp)) positions.emplace_back(std::move(fen), cp);
    }
    if (positions.empty()) throw std::runtime_error("--dataset file has no usable lines: " + args.dataset);
    std::printf("loaded %zu positions from %s\n", positions.size(), args.dataset.c_str());

    if (args.piece_min > kMinPieces || args.piece_max < kMaxPieces) {
        std::vector<std::pair<std::string, int>> filtered;
        filtered.reserve(positions.size());
        for (auto& p : positions) {
            const int pc = count_pieces_in_fen(p.first);
            if (pc >= args.piece_min && pc <= args.piece_max) filtered.push_back(std::move(p));
        }
        std::printf("filtered to %zu/%zu positions with %d-%d pieces (bucket-specialized training)\n",
                    filtered.size(), positions.size(), args.piece_min, args.piece_max);
        if (filtered.empty())
            throw std::runtime_error("no positions in --dataset fall within --piece-min/--piece-max - "
                                      "check the range against the dataset's own Datensatz-Statistiken");
        positions = std::move(filtered);
    }

    // Shuffle once now, and again before every epoch below - datasets from
    // gen_data/full aren't written in any random order (bucket-balanced
    // generation in particular tends to cluster whichever piece-count
    // bucket was hardest to fill - typically the rarest, most extreme
    // endgames - toward the END of the file, since easy buckets fill their
    // quota and drop out of contention early), and without shuffling, every
    // epoch sees that same lopsided tail as its OWN last few batches, every
    // single time. That's not a hypothetical: it's exactly what produced
    // the "loss craters to ~0 in phases" reports - the fixed tail of
    // whichever run this was turned out to be a cluster of near-identical,
    // trivially-easy positions (e.g. simple drawn/won endgames), so the
    // network's loss on THAT specific fixed batch sequence crashed toward
    // zero once per epoch, exactly at the epoch boundary, every time -
    // visible as a suspiciously exact, perfectly repeating pattern in the
    // loss curve rather than randomly-distributed noise.
    std::mt19937 shuffle_rng(std::random_device{}());
    std::shuffle(positions.begin(), positions.end(), shuffle_rng);

    const int num_threads = args.threads > 0 ? args.threads : std::max(1u, std::thread::hardware_concurrency());
    const long batch_size = args.batch_size > 0 ? args.batch_size : long(num_threads) * 64;
    std::printf("== training: %d epoch(s) x %zu positions, batch size %ld across %d threads ==\n", args.epochs,
                positions.size(), batch_size, num_threads);

    const long total_steps = long(positions.size()) * args.epochs;
    long done = 0;
    float running_loss = 0.0f;
    long loss_batches = 0;
    auto start_time = std::chrono::steady_clock::now();
    TrainStats stats;

    // One row per batch (step = samples processed so far, loss = that
    // batch's average loss) - the GUI's "Loss-Verlauf" chart reads this
    // directly (see gui/server.py's /api/loss endpoint). Truncates any
    // earlier file of the same name - a fresh run overwrites its own curve.
    const std::string loss_csv_path = args.name + "_loss.csv";
    std::ofstream loss_csv(loss_csv_path);
    if (loss_csv) loss_csv << "step,loss\n";

    for (int epoch = 0; epoch < args.epochs; ++epoch) {
        if (epoch > 0) std::shuffle(positions.begin(), positions.end(), shuffle_rng); // new batch composition every epoch
        for (long batch_start = 0; batch_start < long(positions.size()); batch_start += batch_size) {
            const long batch_end = std::min(batch_start + batch_size, long(positions.size()));
            const long n = batch_end - batch_start;
            // One Gradients PER THREAD (a running sum over that thread's
            // whole chunk), not one per sample: apply_gradients()'s merge
            // step is itself O(entries * network_size) and single-threaded,
            // so merging batch_size (e.g. 1024) individual per-sample
            // Gradients there - rather than num_threads (e.g. 16) pre-summed
            // partials - made it, not the parallel forward/backward passes,
            // the actual bottleneck (that's also why CPU utilization looked
            // low despite --threads being fully honored: most wall-clock
            // time was one thread doing this merge while the others sat
            // idle). See train_network.hpp's compute_gradients_range().
            const std::size_t n_threads = std::size_t(num_threads); // named - avoids the "most vexing parse"
            std::vector<typename Net::Gradients> partials(n_threads);
            const long per_thread = n / num_threads;
            const long remainder = n % num_threads;
            std::vector<std::thread> pool;
            pool.reserve(std::size_t(num_threads));
            for (int t = 0; t < num_threads; ++t) {
                const long lo = long(t) * per_thread + std::min<long>(t, remainder);
                const long hi = lo + per_thread + (t < remainder ? 1 : 0);
                pool.emplace_back([&, t, lo, hi]() {
                    typename Net::Gradients local;
                    for (long i = lo; i < hi; ++i) {
                        const auto& [fen, cp] = positions[std::size_t(batch_start + i)];
                        chess_board board{};
                        setup_fen_position(board, fen);
                        nnue::train::TrainingSample sample = sample_from_board(board, cp);
                        Net::add_gradients_into(local, net.compute_gradients(sample)); // const, safe concurrently
                    }
                    partials[std::size_t(t)] = std::move(local);
                });
            }
            for (auto& th : pool) th.join();

            const float batch_loss = net.apply_gradients(partials); // the one sequential part - now O(threads), not O(batch)
            done += n;
            stats.best_loss = std::min(stats.best_loss, batch_loss);
            stats.final_loss = batch_loss;
            if (loss_csv) loss_csv << done << ',' << batch_loss << '\n';

            // Exponential moving average, not a cumulative since-epoch-start
            // average: the latter (this file's earlier version) lags behind
            // the CSV's raw per-batch loss more and more as an epoch
            // progresses - since loss trends down, that average stays
            // stuck noticeably ABOVE the current, actual loss for most of
            // an epoch (exactly the "chart shows half the log's number"
            // mismatch this replaced). An EMA tracks the current loss
            // closely while still smoothing out per-batch noise, and
            // doesn't reset (and jump) at epoch boundaries either.
            running_loss = (loss_batches == 0) ? batch_loss : 0.98f * running_loss + 0.02f * batch_loss;
            ++loss_batches;

            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            const double rate_ms = elapsed * 1000.0 / double(done);
            const double eta = (total_steps - done) * (elapsed / double(done));
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), "epoch %d/%d  loss=%.5f", epoch + 1, args.epochs, running_loss);
            print_progress_bar(done, total_steps, rate_ms, eta, suffix);
        }
    }
    return stats;
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
    const char* mode_name =
        args.mode == Mode::Full ? "full" : (args.mode == Mode::GenData ? "gen_data" : "only_train");
    std::printf("mode=%s name=\"%s\"\n\n", mode_name, args.name.c_str());

    init_engine_tables();
    std::mt19937 rng(std::random_device{}());

    std::unique_ptr<Net> net;
    float best_loss = std::numeric_limits<float>::infinity();
    float final_loss = 0.0f;
    auto phase_start = std::chrono::steady_clock::now();

    // --- Phase 1a: generate Stockfish-evaluated positions (full, gen_data) -
    if (args.mode == Mode::Full || args.mode == Mode::GenData) {
        std::printf("depth=%d samples=%ld\n", args.depth, args.samples);
        if (args.mode == Mode::Full) run_efficiency_check<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>();

        std::string sf_path = find_stockfish();
        std::printf("using stockfish: %s\n", sf_path.c_str());

        const std::string dataset_path = args.name + ".dataset";
        std::ofstream dataset_file(dataset_path);
        if (!dataset_file) throw std::runtime_error("could not open " + dataset_path + " for writing");

        if (args.mode == Mode::Full) net = std::make_unique<Net>(args.lr, /*ft_weight_decay=*/1e-6f,
                                                                   /*hidden_weight_decay=*/1e-4f, args.lr_half_life);
        std::ofstream loss_csv;
        if (net) {
            loss_csv.open(args.name + "_loss.csv");
            if (loss_csv) loss_csv << "step,loss\n";
        }

        const int num_threads = args.threads > 0 ? args.threads : std::max(1u, std::thread::hardware_concurrency());
        std::printf("== generating %ld Stockfish-evaluated positions (depth %d) across %d threads%s ==\n",
                    args.samples, args.depth, num_threads, net ? ", training on them" : "");

        // Queue capacity: a handful of items per thread is plenty of slack for
        // producers to run slightly ahead of the consumer without either side
        // stalling on the other under normal conditions (see WorkQueue's
        // comment on why the consumer essentially never falls behind).
        WorkQueue queue(std::size_t(num_threads) * 8);
        std::atomic<long> emitted{0};
        // One quota shared by all worker threads (see producer_worker()) -
        // std::vector, not std::array<std::atomic<long>, N>: atomics aren't
        // copyable, and a vector can still default-construct N of them
        // in place (each starts at 0) without needing that.
        std::vector<std::atomic<long>> bucket_counts(kNumPieceBuckets);
        // Round UP: integer division would otherwise leave a few samples'
        // worth of total quota capacity short of args.samples (e.g.
        // 2000/31=64, 31*64=1984 - the last 16 would never be reachable,
        // since every bucket would already report "full" once the walks
        // that CAN still make progress run out). Rounding up gives every
        // bucket a hair more headroom than strictly needed; the
        // `emitted < total_samples` loop condition below still stops
        // generation at exactly args.samples regardless.
        const long target_per_bucket = std::max<long>(1, (args.samples + kNumPieceBuckets - 1) / kNumPieceBuckets);
        std::vector<std::thread> workers;
        workers.reserve(std::size_t(num_threads));
        for (int t = 0; t < num_threads; ++t)
            workers.emplace_back(producer_worker, t, std::ref(emitted), args.samples, target_per_bucket, args.depth,
                                  std::cref(sf_path), std::ref(queue), std::ref(bucket_counts));

        phase_start = std::chrono::steady_clock::now();
        float running_loss = 0.0f;
        const long report_every = std::max<long>(1, args.samples / 20);
        long consumed = 0;
        DatasetStats dataset_stats;
        dataset_stats.piece_counts.reserve(std::size_t(args.samples));
        dataset_stats.cps.reserve(std::size_t(args.samples));
        GeneratedItem item;
        while (consumed < args.samples && queue.pop(item)) {
            dataset_file << item.fen << ';' << item.cp << ';' << item.stm << '\n';
            dataset_stats.record(item.fen, item.cp, item.stm);
            if (net) running_loss += net->train_step(item.sample);
            ++consumed;

            if (net && consumed % report_every == 0) {
                final_loss = running_loss / float(report_every);
                best_loss = std::min(best_loss, final_loss);
                running_loss = 0.0f;
                if (loss_csv) loss_csv << consumed << ',' << final_loss << '\n';
            }
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_start).count();
            const double rate_ms = elapsed * 1000.0 / double(consumed);
            const double eta = rate_ms / 1000.0 * double(args.samples - consumed);
            char suffix[64] = "";
            if (net) std::snprintf(suffix, sizeof(suffix), "loss=%.5f", final_loss);
            print_progress_bar(consumed, args.samples, rate_ms, eta, suffix);
        }
        queue.close(); // in case the loop exited via consumed==args.samples before every worker noticed
        for (auto& w : workers) w.join();
        dataset_file.close();
        std::printf("dataset written to %s\n", dataset_path.c_str());
        print_dataset_stats(dataset_stats);

        if (args.mode == Mode::GenData) return 0; // no network at all in this mode - nothing left to do
    }

    // --- Phase 1b: train on an already-generated dataset (only_train) -----
    if (args.mode == Mode::OnlyTrain) {
        run_efficiency_check<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>();
        net = std::make_unique<Net>(args.lr, /*ft_weight_decay=*/1e-6f, /*hidden_weight_decay=*/1e-4f,
                                     args.lr_half_life);
        phase_start = std::chrono::steady_clock::now();
        TrainStats stats = run_only_train(args, *net);
        best_loss = stats.best_loss;
        final_loss = stats.final_loss;
    }

    // --- Shared tail (full, only_train): quantize, save, sanity-check,
    // validate against a fresh Stockfish-labeled set, record on the
    // leaderboard. gen_data already returned above, before reaching here.
    net->finalize_training();
    auto training_end = std::chrono::steady_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(training_end - phase_start).count();

    auto quantized = Inference::make();
    net->quantize_into(*quantized);
    const std::string net_path = args.name + ".nnue";
    quantized->save(net_path);
    std::printf("network written to %s\n", net_path.c_str());

    // Cheap regression check: does the trained network actually
    // discriminate between positions, or has it collapsed to predicting
    // (near-)the same thing everywhere? See eval_sanity_check.hpp - shared
    // with self_play.cpp, where this failure mode was found and chased.
    {
        constexpr int kSanityPositions = 200;
        std::vector<float> sanity_evals;
        sanity_evals.reserve(kSanityPositions);
        for (int i = 0; i < kSanityPositions; ++i) {
            chess_board board = random_position(rng);
            nnue::train::TrainingSample s = sample_from_board(board, 0);
            typename Net::Cache cache;
            net->forward(s, cache);
            sanity_evals.push_back(cache.out_pre);
        }
        print_eval_sanity_stats(compute_eval_sanity_stats(sanity_evals), sanity_evals.size());
    }

    // Held-out validation: for full_cycle/only_train this is mostly a
    // sanity check (train/held-out gap), since the training loss is already
    // Stockfish-anchored - but computing it the same way self_play.cpp does
    // keeps both methods' leaderboard entries on one directly comparable
    // scale. A fresh Stockfish process is spawned here regardless of mode
    // (gen_data never reaches this point, and only_train never spawned one
    // earlier) - it's a handful of positions, not worth threading.
    // If this run was trained on a narrowed --piece-min/--piece-max range
    // (a bucket-specialized net), the held-out set must be drawn from that
    // same range - otherwise most/all validation positions fall outside
    // what the net was ever trained to evaluate, and its val_loss/
    // correlation numbers are meaningless (both in isolation and compared
    // against a full-range net's numbers, which ARE measured on the range
    // they were trained on). Plain rejection sampling: random_position()
    // already covers the full board-development range, so drawing until
    // one lands in [piece_min, piece_max] is cheap for any range that
    // actually occurs during a game; the retry cap just guards against a
    // range nothing ever lands in (e.g. piece_min > piece_max entered by
    // mistake) turning into an infinite loop.
    const bool bucket_specialized = args.piece_min > kMinPieces || args.piece_max < kMaxPieces;
    if (bucket_specialized) {
        std::printf("== held-out set restricted to %d-%d pieces (bucket-specialized run) ==\n",
                    args.piece_min, args.piece_max);
    }
    std::printf("== validation against a fresh Stockfish-labeled held-out set (%d positions) ==\n",
                args.validation_positions);
    UciEngine engine(find_stockfish());
    auto gen_fen_and_cp = [&]() -> std::pair<std::string, int> {
        chess_board vboard = random_position(rng);
        if (bucket_specialized) {
            constexpr int kMaxAttempts = 2000;
            int attempts = 0;
            while (attempts < kMaxAttempts &&
                   (piece_count_of(vboard) < args.piece_min || piece_count_of(vboard) > args.piece_max)) {
                vboard = random_position(rng);
                ++attempts;
            }
            if (attempts >= kMaxAttempts)
                throw std::runtime_error("could not find a held-out position with " +
                                          std::to_string(args.piece_min) + "-" + std::to_string(args.piece_max) +
                                          " pieces after " + std::to_string(kMaxAttempts) +
                                          " attempts - check --piece-min/--piece-max");
        }
        std::string vfen = board_to_fen(vboard);
        int vcp = engine.evaluate(vfen, args.depth);
        return {vfen, vcp};
    };
    auto raw_eval_for_fen = [&](const std::string& fen) -> float {
        chess_board vboard{};
        setup_fen_position(vboard, fen);
        nnue::train::TrainingSample s = sample_from_board(vboard, 0);
        typename Net::Cache cache;
        net->forward(s, cache);
        return cache.out_pre;
    };
    ValidationResult validation =
        validate_against_stockfish(args.validation_positions, gen_fen_and_cp, raw_eval_for_fen);
    std::printf("  correlation=%.4f  calibrated_prob_mse=%.5f  (fit: scale=%.3f offset=%.3f)\n\n",
                validation.correlation, validation.calibrated_prob_mse, validation.scale, validation.offset);

    TrainingResult result;
    result.name = args.name;
    result.method = args.mode == Mode::OnlyTrain ? "stockfish (only_train)" : "stockfish";
    result.timestamp = current_timestamp();
    result.acc = NNUE_ACC_SIZE;
    result.h1 = NNUE_H1;
    result.h2 = NNUE_H2;
    result.h3 = NNUE_H3;
    result.depth = args.depth;
    result.samples = args.samples;
    result.best_loss = best_loss;
    result.final_loss = final_loss;
    result.total_time_ms = total_time_ms;
    result.validation_loss = validation.calibrated_prob_mse;
    result.validation_correlation = validation.correlation;
    append_result(result);
    std::printf("result recorded in %s\n", results_csv_path().c_str());

    print_leaderboard();

    return 0;
}
