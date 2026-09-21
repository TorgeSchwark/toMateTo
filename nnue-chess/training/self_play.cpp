// Self-play reinforcement learning: no Stockfish, no external labels at
// all. Plays games against itself using a small alpha-beta search whose
// leaf evaluator is the very network being trained; at every position
// actually visited during self-play, the search's own backed-up value
// (what alpha-beta says the position is worth after looking a few plies
// ahead) becomes that position's training target, the move the search
// preferred is then actually played, and the game continues.
//
// This is the "after the best move, the position should back up to the
// same evaluation" idea: by definition of negamax, value(P) already equals
// -value(child after the best move) - the point of training on it is
// bootstrapping, i.e. teaching the fast *static* (0-ply) evaluator to
// approximate what the slower *search* (D-ply) finds, so that eventually
// evaluating the position alone gets you most of what looking D plies
// ahead used to. This is the classic TD-Leaf(lambda) idea (Baxter, Beal &
// Smith - used e.g. in KnightCap); here it's the simplest version, TD(0)
// with lambda=1 truncated to one step, no eligibility trace.
//
// Uses its OWN COPY of the toMateTo move generator under
// engine/toMateTo_engine/ (not the original under ../toMateTo_engine/, the
// way full_cycle.cpp does it) - self-contained inside nnue-chess, no other
// files anywhere outside this directory are read or written.
//
// Runtime flags:
//   --search-depth N  alpha-beta depth per move, in plies; default 2
//   --games N         number of self-play games to generate + train on
//   --max-plies N     abort a game (as a draw, value 0) after this many
//                      plies; default 150 - bounds worst-case game length,
//                      no repetition/50-move-rule detection is implemented
//   --name NAME       names <NAME>.nnue, <NAME>.dataset and the run's
//                      leaderboard entry
//   --report          print the leaderboard and exit (no training)
//
// Self-play's training loss is measured against its OWN search, which has
// nothing anchoring its scale to real centipawns (unlike full_cycle.cpp,
// where every target is literally sigmoid(stockfish_cp/CP_SCALE)) - so
// after training, this tool ALSO spawns Stockfish purely to score a small
// held-out validation set and report a scale-corrected error against it
// (see validation.hpp) - the only place Stockfish is used here; it never
// sees or influences training itself.
//
// Architecture is fixed at compile time (see full_cycle.cpp's header
// comment for why); use ../self_play.sh, which reconfigures/rebuilds for
// you from a plain "--arch ACC,H1,H2,H3" flag.
//
// POSIX/Linux only (Stockfish is spawned via fork+exec for validation
// only - see uci_engine.hpp).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "train_network.hpp"
#include "leaderboard.hpp"
#include "efficiency_check.hpp"
#include "validation.hpp"
#include "uci_engine.hpp"

#include "engine/toMateTo_engine/move_generation/chess_board.h"
#include "engine/toMateTo_engine/table_generation/knight_tables.h"
#include "engine/toMateTo_engine/table_generation/magic_gen.h"
#include "engine/toMateTo_engine/table_generation/magic_king_tables.h"

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
    int search_depth = 2;
    long games = 200;
    int max_plies = 150;
    std::string name = "selfplay";
    bool report_only = false;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
                  "Usage: %s --games N --search-depth N --max-plies N --name NAME [--arch ACC,H1,H2,H3]\n"
                  "       %s --report\n"
                  "  --games         number of self-play games to generate AND train on\n"
                  "  --search-depth  alpha-beta plies per move (the network's own eval is the leaf\n"
                  "                  evaluator) - higher = better move choices, slower; default 2\n"
                  "  --max-plies     abort a game as a draw after this many plies; default 150\n"
                  "  --name          names <NAME>.nnue, <NAME>.dataset and the leaderboard entry\n"
                  "  --arch          sanity-checked, not applied: this binary was compiled for\n"
                  "                  (%d,%d,%d,%d); use ../self_play.sh to actually change it\n"
                  "  --report        print the results leaderboard and exit - no training\n",
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
        if (arg == "--search-depth") {
            a.search_depth = std::stoi(next());
        } else if (arg == "--games") {
            a.games = std::stol(next());
        } else if (arg == "--max-plies") {
            a.max_plies = std::stoi(next());
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
                                 "network regardless - use ../self_play.sh to actually change it.\n",
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

// Only used for validation at the end - searches upward from the current
// working directory for "testing/stockfish_bin", same convention as
// full_cycle.cpp.
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
        "could not find a Stockfish binary under any 'testing/stockfish_bin' directory above "
        "the current working directory (needed for the final validation step only) - run this "
        "from somewhere inside the toMateTo repo");
}

// ---- board -> NNUE feature extraction ----------------------------------
// Same adapter as full_cycle.cpp, duplicated rather than shared: this file
// links its OWN copy of chess_board (engine/toMateTo_engine/...), a
// different type from full_cycle.cpp's, despite the identical name - a
// shared header both files included would silently depend on whichever
// copy's chess_board.h happened to be found first via the include path.
// Keeping this ~30 lines separate avoids that entirely.

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

// target_wdl left at its default (0) - forward() never reads it, only
// train_step()'s loss computation does, and callers here set it explicitly
// (to the search value) before training on a sample.
nnue::train::TrainingSample sample_from_board(const chess_board& board) {
    nnue::train::TrainingSample s;
    int white_king = __builtin_ctzll(board.white.king);
    int black_king = __builtin_ctzll(board.black.king);
    collect_features(board, nnue::WHITE, white_king, s.white_features);
    collect_features(board, nnue::BLACK, black_king, s.black_features);
    s.side_to_move = board.whites_turn ? nnue::WHITE : nnue::BLACK;
    return s;
}

// Raw (pre-sigmoid) evaluation from the side-to-move's perspective, in the
// same "cp/CP_SCALE" units the rest of this codebase uses (see
// train_network.hpp's header comment) - this is what makes the search
// below negamax-able (symmetric domain, negate on ply change) and what
// validation.hpp calibrates against Stockfish's cp scale.
float raw_eval(const chess_board& board, Net& net) {
    nnue::train::TrainingSample s = sample_from_board(board);
    typename Net::Cache cache;
    net.forward(s, cache);
    return cache.out_pre;
}

// full_cycle.cpp's TrainNetwork starts its feature-transformer weights at
// exactly 0 (LazySparseAdamW's usual, sensible default for a huge sparse
// embedding table - see sparse_optimizer.hpp) because Stockfish's labels
// give every position a distinct, externally-anchored target from move
// one, which is plenty of signal to pull individual rows away from 0.
//
// Self-play has no such anchor. With every row at exactly 0, the
// accumulator for ANY position is just the (position-independent) bias, so
// raw_eval() returns the identical value everywhere - which means the
// search below collapses to returning that same constant at the root
// too (every leaf ties, so every internal negamax also returns the same
// constant, all the way up). The TD target then EXACTLY equals the
// network's own static prediction, the loss is exactly 0, and - this is
// the real problem, not just a cosmetic one - the *gradient* is exactly 0
// too (d/dw of (pred-target)^2 is 0 when pred==target bit-for-bit), so no
// weight ever moves and the network is stuck in this state permanently.
// It is a genuine fixed point of the update rule, not just slow learning.
//
// The fix: give every row a small random nudge before self-play starts, so
// different positions get different raw evaluations from move one and the
// search - and therefore the TD targets it produces - actually depend on
// what is on the board. +-0.1 (not +-0.01): loud enough that the 2-ply
// search actually finds moves whose resulting position looks meaningfully
// different from the root - too quiet (tried +-0.01 first) and the search
// barely disagrees with the static eval either, so the TD signal, while
// technically nonzero, stays too small to teach much in any reasonable
// number of games (see kExplorationRate's comment for the other, separate
// contributor to that same symptom).
void randomize_ft_weights(Net& net, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (std::size_t f = 0; f < nnue::NUM_FEATURES; ++f) {
        float* row = net.ft_weight_opt.row(f);
        for (std::size_t d = 0; d < NNUE_ACC_SIZE; ++d) row[d] = dist(rng);
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

chess_board fresh_start_position() {
    // {} + explicit assignment, not chess_board board; + setup_chess_board()
    // alone: setup_chess_board() only fills in white/black/whites_turn -
    // castling_rights, ep_square and the move counters are plain
    // uninitialized members otherwise (a real bug in the toMateTo engine
    // itself, found while building full_cycle.cpp - see this repo's
    // README). {} zero-initializes everything first; the assignments below
    // then set what a fresh start position actually needs.
    chess_board board{};
    board.setup_chess_board();
    board.castling_rights = ANY_CASTLING;
    board.ep_square = SQ_NONE;
    board.halve_move_counter = 0;
    board.full_move_counter = 1;
    return board;
}

// ---- alpha-beta search, network as leaf evaluator -----------------------

constexpr float kMateValue = 1000.0f; // in raw "cp/400"-ish units - dwarfs any positional score

float negamax(chess_board& board, int depth, float alpha, float beta, Net& net) {
    if (depth == 0) return raw_eval(board, net);
    MoveStacks ms;
    find_all_moves(&ms, &board);
    int total = ms.normal_size() + ms.capture_size();
    if (total == 0) return is_in_check(&board) ? -kMateValue : 0.0f;

    float best = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < total; ++i) {
        Move mv = i < ms.normal_size() ? ms.normal_moves[i] : ms.capture_moves[i - ms.normal_size()];
        StateInfo st;
        make_move(&board, mv, st);
        float score = -negamax(board, depth - 1, -beta, -alpha, net);
        undo_move(&board, mv, st);
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; // beta cutoff
    }
    return best;
}

struct SearchResult {
    Move best_move{};
    float value = 0.0f;
    bool has_move = false;
};

// Root search: examines every legal move (no cutoff at the root - we need
// the true best, not just "good enough to prune"), recursing into negamax
// (which does prune) for the rest of the tree.
SearchResult search_root(chess_board& board, int depth, Net& net) {
    MoveStacks ms;
    find_all_moves(&ms, &board);
    int total = ms.normal_size() + ms.capture_size();
    SearchResult result;
    if (total == 0) {
        result.value = is_in_check(&board) ? -kMateValue : 0.0f;
        return result;
    }
    float alpha = -std::numeric_limits<float>::infinity();
    const float beta = std::numeric_limits<float>::infinity();
    for (int i = 0; i < total; ++i) {
        Move mv = i < ms.normal_size() ? ms.normal_moves[i] : ms.capture_moves[i - ms.normal_size()];
        StateInfo st;
        make_move(&board, mv, st);
        float score = -negamax(board, depth - 1, -beta, -alpha, net);
        undo_move(&board, mv, st);
        if (!result.has_move || score > result.value) {
            result.value = score;
            result.best_move = mv;
            result.has_move = true;
        }
        if (score > alpha) alpha = score;
    }
    return result;
}

// ---- self-play game -------------------------------------------------------
// Plays one game to (bounded) completion, training on every position
// actually visited using that position's own search value as the target,
// then advancing the game with the move the search preferred. Returns the
// number of positions trained on.
// Fraction of moves played randomly instead of the search's top choice (the
// TD *target* still always uses the search's real value - only which move
// actually gets played to continue the game is randomized). Without this,
// move selection is 100% deterministic given the network + search, so as
// soon as preferences stabilize even slightly, every game collapses onto
// nearly the same handful of lines and most of the 40960-feature space
// never gets visited at all - which looks exactly like "the network always
// predicts the same thing", because for any position outside that narrow
// rut it effectively does. This is the standard RL fix (epsilon-greedy
// exploration; AlphaZero's Dirichlet-noise-at-the-root serves the same
// purpose, just fancier) and is a second, independent contributor to
// "always the same eval" beyond the exact fixed point randomize_ft_weights
// fixes - that one is about the gradient being zero, this one is about the
// *data* being too narrow for a nonzero gradient to teach much.
constexpr float kExplorationRate = 0.1f;

long play_self_play_game(std::mt19937& rng, Net& net, int search_depth, int max_plies, std::ofstream& dataset_file,
                          float& running_loss) {
    chess_board board = fresh_start_position();
    std::uniform_real_distribution<float> explore_roll(0.0f, 1.0f);

    // A couple of random opening plies for game diversity - otherwise
    // every self-play game starts identically and explores the same lines
    // as soon as the network's move preferences stabilize.
    std::uniform_int_distribution<int> opening_dist(0, 4);
    int opening_plies = opening_dist(rng);
    for (int i = 0; i < opening_plies; ++i) {
        MoveStacks ms;
        find_all_moves(&ms, &board);
        int total = ms.normal_size() + ms.capture_size();
        if (total == 0) return 0; // essentially impossible this early, just bail
        std::uniform_int_distribution<int> move_dist(0, total - 1);
        int idx = move_dist(rng);
        Move mv = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];
        StateInfo st;
        make_move(&board, mv, st);
    }

    long positions = 0;
    for (int ply = 0; ply < max_plies; ++ply) {
        SearchResult sr = search_root(board, search_depth, net);
        if (!sr.has_move) break; // checkmate or stalemate reached

        nnue::train::TrainingSample sample = sample_from_board(board);
        sample.target_wdl = nnue::train::sigmoid(sr.value); // always the search's real value, even on exploration plies
        running_loss += net.train_step(sample);
        ++positions;

        int cp = int(std::lround(double(sr.value) * nnue::train::CP_SCALE));
        dataset_file << board_to_fen(board) << ';' << cp << ';' << (board.whites_turn ? 'w' : 'b') << '\n';

        Move move_to_play = sr.best_move;
        if (explore_roll(rng) < kExplorationRate) {
            MoveStacks ms;
            find_all_moves(&ms, &board);
            int total = ms.normal_size() + ms.capture_size();
            std::uniform_int_distribution<int> move_dist(0, total - 1);
            int idx = move_dist(rng);
            move_to_play = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];
        }
        StateInfo st;
        make_move(&board, move_to_play, st);
    }
    return positions;
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
    std::printf("games=%ld search_depth=%d max_plies=%d name=\"%s\"\n\n", args.games, args.search_depth,
                args.max_plies, args.name.c_str());

    run_efficiency_check<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>();

    init_engine_tables();

    const std::string dataset_path = args.name + ".dataset";
    std::ofstream dataset_file(dataset_path);
    if (!dataset_file) throw std::runtime_error("could not open " + dataset_path + " for writing");

    std::mt19937 rng(std::random_device{}());
    // Higher lr than full_cycle.cpp's 1e-4: self-play positions are much
    // more expensive to generate (a real search per ply, vs. one Stockfish
    // call), so there are far fewer of them per unit of wall-clock time -
    // each one needs to move the needle more. train_network.hpp's
    // safeguards against runaway saturation (leaky ClippedReLU gradient,
    // bias weight decay - see its header comments) are what make this safe
    // to raise; without those this would risk the same "every unit
    // saturates and dies" failure mode found earlier while tuning
    // train_demo.cpp.
    Net net(/*lr=*/5e-4f, /*ft_weight_decay=*/1e-6f, /*hidden_weight_decay=*/1e-4f);
    randomize_ft_weights(net, rng); // see this function's comment: essential for self-play, not cosmetic

    std::printf(
        "== self-play: %ld games, search depth %d (network's own eval as leaf evaluator, no Stockfish) ==\n",
        args.games, args.search_depth);
    auto gen_start = std::chrono::steady_clock::now();
    float running_loss = 0.0f;
    long positions_since_report = 0;
    long total_positions = 0;
    float best_window_loss = std::numeric_limits<float>::infinity();
    float last_window_loss = 0.0f;
    const long report_every = std::max<long>(1, args.games / 20);

    for (long g = 0; g < args.games; ++g) {
        long before = total_positions;
        float loss_before = running_loss;
        long n = play_self_play_game(rng, net, args.search_depth, args.max_plies, dataset_file, running_loss);
        total_positions += n;
        positions_since_report += n;
        (void)loss_before;
        (void)before;

        if ((g + 1) % report_every == 0 || g + 1 == args.games) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - gen_start).count();
            double per_game = elapsed / double(g + 1);
            double eta = per_game * double(args.games - (g + 1));
            float window_loss = positions_since_report > 0 ? running_loss / float(positions_since_report) : 0.0f;
            best_window_loss = std::min(best_window_loss, window_loss);
            last_window_loss = window_loss;
            std::printf("[games %6ld/%6ld, positions %8ld] avg loss=%.7f  %.1f ms/game  ETA %.0fs\n", g + 1,
                        args.games, total_positions, window_loss, per_game * 1000.0, eta);
            running_loss = 0.0f;
            positions_since_report = 0;
        }
    }
    dataset_file.close();
    std::printf("dataset written to %s (%ld positions from %ld games)\n", dataset_path.c_str(), total_positions,
                args.games);

    net.finalize_training();
    auto training_end = std::chrono::steady_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(training_end - gen_start).count();

    auto quantized = Inference::make();
    net.quantize_into(*quantized);
    const std::string net_path = args.name + ".nnue";
    quantized->save(net_path);
    std::printf("network written to %s\n", net_path.c_str());

    // The only place this tool talks to Stockfish: purely to score a
    // held-out set never trained on, so the self-play network's internally
    // bootstrapped evaluation scale can be compared fairly against
    // full_cycle's Stockfish-anchored one (see validation.hpp's header
    // comment for why that comparison isn't meaningful without this step).
    std::printf("== validation against a fresh Stockfish-labeled held-out set ==\n");
    std::string sf_path = find_stockfish();
    UciEngine engine(sf_path);
    constexpr int kValidationPositions = 200;
    constexpr int kValidationDepth = 8;
    auto gen_fen_and_cp = [&]() -> std::pair<std::string, int> {
        chess_board vboard = fresh_start_position();
        std::uniform_int_distribution<int> ply_dist(0, 40);
        int plies = ply_dist(rng);
        for (int i = 0; i < plies; ++i) {
            MoveStacks ms;
            find_all_moves(&ms, &vboard);
            int total = ms.normal_size() + ms.capture_size();
            if (total == 0) break;
            std::uniform_int_distribution<int> move_dist(0, total - 1);
            int idx = move_dist(rng);
            Move mv = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];
            StateInfo st;
            make_move(&vboard, mv, st);
        }
        std::string vfen = board_to_fen(vboard);
        int vcp = engine.evaluate(vfen, kValidationDepth);
        return {vfen, vcp};
    };
    auto raw_eval_for_fen = [&](const std::string& fen) -> float {
        chess_board vboard{};
        setup_fen_position(vboard, fen);
        return raw_eval(vboard, net);
    };
    ValidationResult validation = validate_against_stockfish(kValidationPositions, gen_fen_and_cp, raw_eval_for_fen);
    std::printf("  correlation=%.4f  calibrated_prob_mse=%.5f  (fit: scale=%.3f offset=%.3f)\n\n",
                validation.correlation, validation.calibrated_prob_mse, validation.scale, validation.offset);

    TrainingResult result;
    result.name = args.name;
    result.method = "selfplay";
    result.timestamp = current_timestamp();
    result.acc = NNUE_ACC_SIZE;
    result.h1 = NNUE_H1;
    result.h2 = NNUE_H2;
    result.h3 = NNUE_H3;
    result.depth = args.search_depth;
    result.samples = total_positions;
    result.games = args.games;
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
