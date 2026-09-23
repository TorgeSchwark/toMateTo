// Correctness test for toMateTo_nnue.cpp's incremental NNUE accumulator
// update (update_accumulator(), the highest-risk part of the TestEngine
// search integration - see that file's header comment). Plays many random
// legal games from the start position and, after every single move,
// verifies the incrementally-updated accumulator matches a from-scratch
// full refresh, bit-for-bit, for both perspectives. A silent bug here
// wouldn't crash anything - it would just feed the net a wrong position and
// quietly make the engine play worse, so this needs to be checked by an
// actual test, not just "it compiles and doesn't crash".
//
// Random weights, not a trained network: only feature_transformer.weights/
// biases matter for this test (everything downstream of the accumulator is
// untouched), and random nonzero weights make a wrong accumulator produce a
// numerically different (and hence detectable) result just as reliably as
// real trained weights would.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/table_generation/TT.h"
#include "toMateTo_engine/toMateTo/toMateTo_nnue.h"

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

int main() {
    init_engine_tables();
    init_zobrist();

    // Heap-allocate, not a local: SearchNet (NNUE<ACC_SIZE,...>) is tens of
    // MB (feature_transformer.weights alone is NUM_FEATURES * ACC_SIZE
    // bytes - see feature_transformer.hpp's warning), far too big for the
    // stack.
    auto net = SearchNet::make();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> w8(-30, 30);
    std::uniform_int_distribution<int> w16(-200, 200);
    for (auto& row : net->feature_transformer.weights)
        for (auto& w : row) w = static_cast<int8_t>(w8(rng));
    for (auto& b : net->feature_transformer.biases) b = static_cast<int16_t>(w16(rng));

    nnue_search_init_thread(net.get());

    constexpr int kGames = 300;
    constexpr int kMaxPlies = 100;
    long total_moves = 0;
    int mismatches = 0;

    std::uniform_int_distribution<int> pick_dist(0, 255);

    for (int g = 0; g < kGames; ++g) {
        chess_board board = fresh_start_position();
        nnue_search_set_root(board);

        for (int ply = 0; ply < kMaxPlies; ++ply) {
            MoveStacks ms;
            find_all_moves(&ms, &board);
            const int total = ms.normal_size() + ms.capture_size();
            if (total == 0) break; // checkmate/stalemate - start a new game

            const int idx = pick_dist(rng) % total;
            const Move move = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];

            const bool ok = nnue_search_debug_check_move(&board, move);
            ++total_moves;
            if (!ok) {
                ++mismatches;
                std::fprintf(stderr, "MISMATCH at game %d, ply %d, move %s\n", g, ply,
                             move.move_to_string(board.whites_turn).c_str());
                if (mismatches > 5) {
                    std::fprintf(stderr, "(too many mismatches, stopping early)\n");
                    return 1;
                }
            }
        }
    }

    std::printf("checked %ld moves across %d games: %d mismatches\n", total_moves, kGames, mismatches);
    return mismatches == 0 ? 0 : 1;
}
