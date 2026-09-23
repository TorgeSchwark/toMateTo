#pragma once
// NNUE-evaluated variant of ../../../../toMateTo_engine/toMateTo/toMateTo.cpp
// (the real engine's search) - copied and adapted here, inside nnue-chess/,
// per this project's standing rule of never modifying files outside
// nnue-chess/ (see self_play.cpp's own engine/ copy for the precedent this
// follows).
//
// What's different from the original:
//   - quiescence()'s stand-pat score comes from an NNUE net instead of
//     pesto_eval(). Delta pruning still uses pesto_piece_value()/
//     pesto_game_phase() (a plain material/PST estimate - it's a pruning
//     margin heuristic, not the actual score, so there is no reason to tie
//     it to whichever eval is scoring the position) - see toMateTo.cpp's
//     comment at its call site for why this split is safe.
//   - The NNUE accumulator is maintained INCREMENTALLY across the search
//     tree (add/remove features per move, full refresh only on the rare
//     "this side's own king moved" case - see update_accumulator() in the
//     .cpp) instead of recomputed from scratch per node, exactly like a
//     real NNUE engine and exactly what the user asked for ("mit update
//     nicht mit full recalculate"). It lives in a thread_local stack
//     indexed by search ply (SearchState::acc_stack), the same pattern
//     already used here for killer moves - no function signatures needed
//     to change to thread an accumulator parameter through.
//   - thread_local transposition_table (table_generation/TT.h) already
//     gives every search thread its own TT for free - no extra work needed
//     to run many of these searches concurrently (see strength_match.cpp).

#include <string>
#include <chrono>
#include <vector>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/move_generation/find_capture_moves.h"
#include "toMateTo_engine/move_generation/types.h"
#include "toMateTo_engine/table_generation/TT.h"
#include "toMateTo_engine/toMateTo/eval.h"
#include "toMateTo_engine/toMateTo/profiler.h"

#include "nnue/network.hpp"

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

using SearchNet = nnue::NNUE<NNUE_ACC_SIZE, NNUE_H1, NNUE_H2, NNUE_H3>;
using SearchAcc = nnue::Accumulator<NNUE_ACC_SIZE>;

// Bucket-NNUE: instead of one net for every position, several nets can each
// specialize on one game-phase slice (e.g. a net trained only on 2-15-piece
// endgame positions - see full_cycle.cpp's --piece-min/--piece-max). Whole-
// board piece count (both kings included, so 2..32) selects which net is
// active at any given search node - see bucket_for() in the .cpp. `net` is
// non-owning, same convention as the single-net API below: the caller (see
// strength_match.cpp) keeps every bucket's SearchNet alive for as long as
// any search thread might use it.
struct NnueBucket {
    int min_pieces;
    int max_pieces;
    const SearchNet* net;
};

// Checks that `buckets` partitions [2,32] exactly - no gaps, no overlaps
// (order in the vector doesn't matter, this sorts a copy). Returns "" if
// valid, otherwise a human-readable description of the first problem found
// - call this before trusting a --net-buckets argument, not after: a gap
// would otherwise silently fall back to bucket 0 mid-search (see
// bucket_for()'s comment), which is exactly the kind of "seems fine, plays
// worse for reasons that aren't obvious" bug this project has already hit
// more than once.
std::string validate_bucket_coverage(std::vector<NnueBucket> buckets);

// Must be called once before any search on this thread (each search thread
// gets its own bucket list + a fully-refreshed root accumulator per bucket
// slot - see strength_match.cpp's worker setup). The SearchNet(s) pointed
// to by `net`/`buckets` must outlive the search calls; ownership stays with
// the caller (typically shared, read-only NNUEs loaded once and pointed to
// by every worker thread).
void nnue_search_init_thread(const SearchNet* net);
// Bucket-NNUE variant of the above - `buckets` must already be validated
// (validate_bucket_coverage() returned "") and must outlive the search
// calls on this thread.
void nnue_search_init_thread_buckets(const std::vector<NnueBucket>* buckets);

// Sets ply-0 of this thread's accumulator stack to `board`'s position (full
// refresh, both perspectives) - call once per new game/position before
// alpha_beta_tt_toMateTo(), which otherwise assumes ply 0 is already valid.
void nnue_search_set_root(const chess_board& board);

int alpha_beta(chess_board* board, int depth, int alpha, int beta, int root_dist);

int quiescence(chess_board* board, int alpha, int beta, int root_dist);

// Same iterative-deepening root driver as the original, NNUE-scored. Time
// limit in seconds. Call nnue_search_set_root() with the same position
// first.
std::string alpha_beta_tt_toMateTo(std::string fen_position, double time_limit_seconds);

// Iterative-deepening depth the most recent alpha_beta_tt_toMateTo() call on
// this thread actually completed (see its definition's comment).
int nnue_search_last_depth();

// Testing-only (see tests/test_nnue_accumulator.cpp): applies `m` to
// `board` via the exact same incremental-accumulator code path real search
// uses (updating this thread's ply-0 slot, S.acc_stack[0], so repeated
// calls walk a whole game forward exactly like search descending the tree
// would), then independently recomputes a from-scratch full refresh of the
// resulting position and returns whether the two agree bit-for-bit. This is
// the automated correctness check for update_accumulator() - the highest-
// risk part of this file (a silent bug here wouldn't crash anything, it
// would just make the net see a wrong position and quietly play worse).
// nnue_search_init_thread()/nnue_search_set_root() must be called first.
bool nnue_search_debug_check_move(chess_board* board, Move m);
