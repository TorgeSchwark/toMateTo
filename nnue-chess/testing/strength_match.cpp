// Plays --games games between the TestEngine (real search - TT, null-move
// pruning, PVS, delta-pruned quiescence, see
// engine/toMateTo_engine/toMateTo/toMateTo_nnue.{h,cpp} - evaluated by a
// trained NNUE net kept incrementally updated across the search tree) and
// Stockfish capped at --elo, parallelized across --threads worker threads
// (each an independent game: its own TestEngine search state/TT, its own
// Stockfish subprocess). Prints a final "RESULT wins=.. draws=.. losses=.."
// line and exits - this binary answers "how does the net do against
// Stockfish AT THIS ONE Elo", nothing more. The actual "what Elo is this
// net" bisection search that calls this repeatedly at different --elo
// values lives in gui/strength_search.py, which is what the GUI's
// "Spielstärke ermitteln" panel actually launches - see that file for why
// the bisection logic is there and not here.
//
// Colors alternate by game index so results aren't biased by White's first-
// move advantage. Draw adjudication: checkmate/stalemate (real), the
// 50-move rule (real), or --max-plies reached (heuristic cutoff, not real
// threefold repetition - see this file's header comment in the README
// section this gets documented under for the tradeoff).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/table_generation/TT.h"
#include "toMateTo_engine/toMateTo/toMateTo_nnue.h"

#include "../training/uci_engine.hpp"
#include "../training/progress_bar.hpp"

namespace fs = std::filesystem;

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
    chess_board board{};
    board.setup_chess_board();
    board.castling_rights = ANY_CASTLING;
    board.ep_square = SQ_NONE;
    board.halve_move_counter = 0;
    board.full_move_counter = 1;
    return board;
}

// Searches upward from cwd for "testing/stockfish_bin" - same convention as
// training/full_cycle.cpp's find_stockfish(), duplicated rather than shared
// for the same reason self_play.cpp duplicates its board helpers (this file
// links a *different* chess_board type than full_cycle.cpp's).
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
    throw std::runtime_error("could not find a Stockfish binary under any 'testing/stockfish_bin' directory "
                              "above the current working directory");
}

// Finds the legal move matching a UCI move string (e.g. "e2e4", "e7e8q") -
// needed because the string alone doesn't carry the internal EN_PASSANT/
// CASTLING flag bits Move needs (see move_stack.h).
Move parse_uci_move(chess_board* board, const std::string& uci) {
    MoveStacks ms;
    find_all_moves(&ms, board);
    auto try_list = [&](Move* begin, Move* end) -> Move* {
        for (Move* m = begin; m != end; ++m) {
            if (m->move_to_string(board->whites_turn) == uci) return m;
        }
        return nullptr;
    };
    if (Move* m = try_list(ms.capture_moves, ms.capture_end)) return *m;
    if (Move* m = try_list(ms.normal_moves, ms.normal_end)) return *m;
    throw std::runtime_error("Stockfish returned a move that doesn't match any legal move: " + uci);
}

enum class Outcome { Win, Draw, Loss }; // from the TestEngine's point of view

struct GameResult {
    Outcome outcome;
    long te_moves = 0;     // how many moves the TestEngine itself made
    long te_depth_sum = 0; // sum of nnue_search_last_depth() over those moves - avg = depth_sum/moves
};

// Optional: collects a training dataset from the positions the TestEngine
// actually faced in real games against a comparably-strong opponent - see
// --save-dataset. Shared across all worker threads, so writes are
// mutex-protected (this is once per TestEngine move, not a hot loop -
// contention here is a non-issue next to a whole search/Stockfish call).
struct DatasetSink {
    std::ofstream file;
    std::mutex mutex;
    int depth = 12;
};

// Called (if `sink` is non-null) once per TestEngine move, BEFORE that move
// is applied - `fen`/`board` is the position it just had to choose a move
// in. Gets a proper-depth Stockfish evaluation of that same position (the
// game's own move for the OTHER side used --movetime-ms, deliberately
// shallow/fast for playable game speed - this is a separate, slower,
// higher-quality analysis call purely for the training label) and appends
// it to the dataset file in the same "fen;cp;stm" format full_cycle.cpp's
// own datasets use, so it drops straight into --mode only_train.
void record_position_for_dataset(DatasetSink* sink, UciEngine& sf, const chess_board& board, const std::string& fen) {
    if (!sink) return;
    const int cp = sf.evaluate(fen, sink->depth);
    std::lock_guard<std::mutex> lock(sink->mutex);
    sink->file << fen << ';' << cp << ';' << (board.whites_turn ? 'w' : 'b') << '\n';
    sink->file.flush();
}

GameResult play_one_game(const std::vector<NnueBucket>* buckets, UciEngine& sf, bool test_engine_is_white,
                          int movetime_ms, int max_plies, DatasetSink* dataset_sink) {
    nnue_search_init_thread_buckets(buckets);
    sf.new_game();

    chess_board board = fresh_start_position();
    const double te_time_limit = movetime_ms / 1000.0;
    GameResult result;

    for (int ply = 0; ply < max_plies; ++ply) {
        MoveStacks ms;
        find_all_moves(&ms, &board);
        if (ms.empty()) {
            const bool in_check = is_in_check(&board);
            if (!in_check) { result.outcome = Outcome::Draw; return result; } // stalemate
            // side to move is checkmated
            const bool white_to_move = board.whites_turn;
            const bool test_engine_won = (white_to_move != test_engine_is_white);
            result.outcome = test_engine_won ? Outcome::Win : Outcome::Loss;
            return result;
        }
        if (board.halve_move_counter >= 100) { result.outcome = Outcome::Draw; return result; } // 50-move rule

        const bool test_engine_to_move = (board.whites_turn == test_engine_is_white);
        const std::string fen = board_to_fen(board);

        Move move;
        if (test_engine_to_move) {
            record_position_for_dataset(dataset_sink, sf, board, fen);
            const std::string uci = alpha_beta_tt_toMateTo(fen, te_time_limit);
            if (uci.empty()) { result.outcome = Outcome::Draw; return result; } // shouldn't happen, defensive
            move = parse_uci_move(&board, uci);
            ++result.te_moves;
            result.te_depth_sum += nnue_search_last_depth();
        } else {
            const std::string uci = sf.best_move(fen, movetime_ms);
            move = parse_uci_move(&board, uci);
        }

        StateInfo st;
        make_move(&board, move, st);
    }
    result.outcome = Outcome::Draw; // --max-plies reached
    return result;
}

struct Args {
    std::string net;         // single-net mode
    std::string net_buckets; // bucket mode: "MIN-MAX:PATH,MIN-MAX:PATH,..." - see print_usage(). Overrides --net.
    std::string stockfish_path;
    int elo = 1500;
    long games = 40;
    int threads = 0;
    int movetime_ms = 100;
    int max_plies = 200;
    std::string save_dataset;   // empty = don't collect training data from these games
    int dataset_depth = 12;     // Stockfish analysis depth for the saved positions' labels
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
                  "Usage: %s (--net PATH.nnue | --net-buckets SPEC) --elo N [--games N] [--threads N]\n"
                  "       %s [--movetime-ms N] [--max-plies N] [--stockfish PATH]\n"
                  "       %s [--save-dataset PATH] [--dataset-depth N]\n"
                  "  --net           one net used for every position (the usual case)\n"
                  "  --net-buckets   several nets, each specialized on one piece-count range -\n"
                  "                  e.g. a net trained with full_cycle's --piece-min/--piece-max\n"
                  "                  on just endgames plus another on just openings/midgames.\n"
                  "                  Format: \"MIN-MAX:PATH,MIN-MAX:PATH,...\" (whole-board piece\n"
                  "                  count, both kings included, so MIN/MAX are in 2..32), e.g.\n"
                  "                  \"2-15:endgame.nnue,16-32:opening.nnue\". The ranges must\n"
                  "                  exactly partition 2..32 - no gaps, no overlaps - checked at\n"
                  "                  startup before any game is played, not discovered mid-search.\n"
                  "                  Overrides --net if both are given.\n"
                  "  --save-dataset  also append every position the TestEngine had to move in\n"
                  "                  (fen;stockfish_cp;stm, same format full_cycle.cpp writes) to\n"
                  "                  this file - positions from real games against a comparably\n"
                  "                  strong opponent, for --mode only_train afterwards\n"
                  "  --dataset-depth Stockfish analysis depth used for --save-dataset's labels\n"
                  "                  (separate from --movetime-ms, which is deliberately shallow/\n"
                  "                  fast for playable game speed); default 12\n",
                  prog, prog, prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { print_usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if (arg == "--net") a.net = next();
        else if (arg == "--net-buckets") a.net_buckets = next();
        else if (arg == "--elo") a.elo = std::stoi(next());
        else if (arg == "--games") a.games = std::stol(next());
        else if (arg == "--threads") a.threads = std::stoi(next());
        else if (arg == "--movetime-ms") a.movetime_ms = std::stoi(next());
        else if (arg == "--max-plies") a.max_plies = std::stoi(next());
        else if (arg == "--stockfish") a.stockfish_path = next();
        else if (arg == "--save-dataset") a.save_dataset = next();
        else if (arg == "--dataset-depth") a.dataset_depth = std::stoi(next());
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); std::exit(1); }
    }
    if (a.net.empty() && a.net_buckets.empty()) {
        std::fprintf(stderr, "--net or --net-buckets is required\n");
        print_usage(argv[0]);
        std::exit(1);
    }
    return a;
}

// Parses "MIN-MAX:PATH,MIN-MAX:PATH,..." into (range, path) pairs - loading
// the nets themselves and validating coverage happens in main(), where the
// loaded SearchNets can be kept alive for the whole run.
struct BucketSpec {
    int min_pieces, max_pieces;
    std::string path;
};

std::vector<BucketSpec> parse_net_buckets(const std::string& spec) {
    std::vector<BucketSpec> out;
    std::size_t pos = 0;
    while (pos < spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string entry = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const std::size_t dash = entry.find('-');
        const std::size_t colon = entry.find(':');
        if (dash == std::string::npos || colon == std::string::npos || colon < dash)
            throw std::runtime_error("--net-buckets entry not in MIN-MAX:PATH form: " + entry);
        BucketSpec b;
        b.min_pieces = std::stoi(entry.substr(0, dash));
        b.max_pieces = std::stoi(entry.substr(dash + 1, colon - dash - 1));
        b.path = entry.substr(colon + 1);
        out.push_back(std::move(b));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    init_engine_tables();
    init_zobrist();

    // Every loaded net (one for single-net mode, several for --net-buckets)
    // lives in net_storage for main()'s whole lifetime - `buckets`'
    // NnueBucket::net pointers (and, for single-net mode, the one
    // play_one_game() call site below) borrow from it, never own.
    std::vector<std::unique_ptr<SearchNet>> net_storage;
    std::vector<NnueBucket> buckets;

    auto load_one = [&](const std::string& path) -> const SearchNet* {
        auto n = SearchNet::make();
        if (!n->load(path)) {
            std::fprintf(stderr, "could not load %s (wrong path, or it was exported for a different architecture "
                                  "than this binary was compiled for: NNUE<%d,%d,%d,%d>)\n",
                          path.c_str(), NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3);
            std::exit(1);
        }
        net_storage.push_back(std::move(n));
        return net_storage.back().get();
    };

    if (!args.net_buckets.empty()) {
        for (const BucketSpec& b : parse_net_buckets(args.net_buckets))
            buckets.push_back(NnueBucket{b.min_pieces, b.max_pieces, load_one(b.path)});
    } else {
        buckets.push_back(NnueBucket{2, 32, load_one(args.net)});
    }
    const std::string coverage_error = validate_bucket_coverage(buckets);
    if (!coverage_error.empty()) {
        std::fprintf(stderr, "--net-buckets: %s\n", coverage_error.c_str());
        return 1;
    }
    if (buckets.size() > 1) {
        std::printf("== %zu NNUE buckets loaded ==\n", buckets.size());
        for (const NnueBucket& b : buckets) std::printf("  %d-%d pieces\n", b.min_pieces, b.max_pieces);
    }

    const std::string sf_path = args.stockfish_path.empty() ? find_stockfish() : args.stockfish_path;
    const int num_threads = args.threads > 0 ? args.threads : std::max(1u, std::thread::hardware_concurrency());

    std::printf("== strength_match: %ld games vs Stockfish (Elo %d), %d threads, %dms/move ==\n", args.games,
                args.elo, num_threads, args.movetime_ms);

    std::unique_ptr<DatasetSink> dataset_sink;
    if (!args.save_dataset.empty()) {
        dataset_sink = std::make_unique<DatasetSink>();
        dataset_sink->depth = args.dataset_depth;
        dataset_sink->file.open(args.save_dataset, std::ios::app); // append: multiple runs can build up one dataset
        if (!dataset_sink->file) throw std::runtime_error("could not open --save-dataset file: " + args.save_dataset);
        std::printf("saving TestEngine's positions (Stockfish depth %d labels) to %s\n", args.dataset_depth,
                    args.save_dataset.c_str());
    }

    std::atomic<long> next_game{0};
    std::atomic<long> wins{0}, draws{0}, losses{0}, done{0};
    // Depth/length tracking, split by outcome - lets the caller tell apart
    // "the search is starved (low depth) specifically in the games it
    // loses" (a time-management/contention bug) from "it just plays worse
    // in longer, more strategically demanding games" (a training/eval
    // weakness, not a bug) - see strength_search.py's reporting.
    std::atomic<long> win_moves{0}, win_depth_sum{0};
    std::atomic<long> draw_moves{0}, draw_depth_sum{0};
    std::atomic<long> loss_moves{0}, loss_depth_sum{0};
    std::mutex progress_mutex;
    auto start_time = std::chrono::steady_clock::now();

    std::atomic<long> failed_games{0};

    // Spawns (or re-spawns, after a broken instance) a Stockfish subprocess.
    // Retries with a short backoff - transient spawn failures (fork/exec
    // hiccups under WSL, briefly observed while testing this) shouldn't be
    // fatal to the whole match, only make one worker briefly slower.
    auto spawn_stockfish = [&]() -> std::unique_ptr<UciEngine> {
        for (int attempt = 0; attempt < 5; ++attempt) {
            try {
                auto engine = std::make_unique<UciEngine>(sf_path);
                engine->set_limit_strength(args.elo);
                return engine;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "\nstrength_match: Stockfish spawn failed (attempt %d/5): %s\n", attempt + 1,
                             e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(200 * (attempt + 1)));
            }
        }
        throw std::runtime_error("could not spawn Stockfish after 5 attempts");
    };

    // A single game throwing (almost always: the Stockfish subprocess's
    // pipe closed unexpectedly mid-game - see spawn_stockfish's comment)
    // must NOT take down every other worker's in-progress games with it -
    // std::thread calls std::terminate() on an uncaught exception, killing
    // the WHOLE process, which is what earlier test runs of this hit. Catch
    // per-game, count it as neither win/draw/loss, respawn Stockfish, and
    // keep going - a few skipped games barely moves a 30+-game score, an
    // aborted process loses everything played so far.
    auto worker = [&]() {
        std::unique_ptr<UciEngine> sf;
        try {
            sf = spawn_stockfish();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "\nstrength_match: worker giving up, could not spawn Stockfish at all: %s\n",
                         e.what());
            return; // other workers still carry on
        }

        for (;;) {
            const long g = next_game.fetch_add(1);
            if (g >= args.games) break;
            const bool test_engine_is_white = (g % 2 == 0);

            GameResult r;
            try {
                r = play_one_game(&buckets, *sf, test_engine_is_white, args.movetime_ms, args.max_plies,
                                   dataset_sink.get());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "\nstrength_match: game %ld failed, skipping: %s\n", g, e.what());
                failed_games.fetch_add(1);
                done.fetch_add(1); // still counts toward progress/ETA, just not toward W/D/L
                try {
                    sf = spawn_stockfish();
                } catch (const std::exception&) {
                    break; // Stockfish is unusable on this worker - stop, other workers carry on
                }
                continue;
            }
            if (r.outcome == Outcome::Win) {
                wins.fetch_add(1);
                win_moves.fetch_add(r.te_moves);
                win_depth_sum.fetch_add(r.te_depth_sum);
            } else if (r.outcome == Outcome::Draw) {
                draws.fetch_add(1);
                draw_moves.fetch_add(r.te_moves);
                draw_depth_sum.fetch_add(r.te_depth_sum);
            } else {
                losses.fetch_add(1);
                loss_moves.fetch_add(r.te_moves);
                loss_depth_sum.fetch_add(r.te_depth_sum);
            }

            const long d = done.fetch_add(1) + 1;
            std::lock_guard<std::mutex> lock(progress_mutex);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            const double rate_ms = elapsed * 1000.0 / double(d);
            const double eta = rate_ms / 1000.0 * double(args.games - d);
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), "W%ld D%ld L%ld", wins.load(), draws.load(), losses.load());
            print_progress_bar(d, args.games, rate_ms, eta, suffix);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    const long total = wins.load() + draws.load() + losses.load();
    const double score = total > 0 ? (double(wins.load()) + 0.5 * double(draws.load())) / double(total) : 0.5;
    auto avg = [](long sum, long n) { return n > 0 ? double(sum) / double(n) : 0.0; };
    std::printf("RESULT wins=%ld draws=%ld losses=%ld games=%ld score=%.4f elo=%d failed=%ld\n", wins.load(),
                draws.load(), losses.load(), total, score, args.elo, failed_games.load());
    std::printf(
        "DEPTH win_avg_depth=%.2f win_avg_plies=%.1f draw_avg_depth=%.2f draw_avg_plies=%.1f "
        "loss_avg_depth=%.2f loss_avg_plies=%.1f\n",
        avg(win_depth_sum.load(), win_moves.load()), avg(win_moves.load(), wins.load()),
        avg(draw_depth_sum.load(), draw_moves.load()), avg(draw_moves.load(), draws.load()),
        avg(loss_depth_sum.load(), loss_moves.load()), avg(loss_moves.load(), losses.load()));
    return 0;
}
