// Hunts for the bug the user found by hand: after some moves, a king's
// bitboard was empty where "find king save squares" expected it - which can
// only happen if some earlier move was allowed to capture a king, which can
// only happen if that king's own side was left in check by an EARLIER move
// the legal-move generator should have rejected. Plays many random legal
// games and, after every move, independently re-checks (via simple ray-
// casting, not the codebase's own magic-bitboard attack code - so this
// can't share whatever bug is in that code) whether the side that just
// moved is still in check, and whether both kings still have exactly one
// bit set. Prints the offending FEN + move and aborts on the first
// violation - a reproducible failing case beats speculation.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"

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

// Independent (no magic bitboards, no shared helpers with the engine's own
// attack code) check: is `target` attacked by `by_white`'s pieces?
bool square_attacked_bruteforce(const chess_board& b, int target, bool by_white) {
    const one_side& atk = by_white ? b.white : b.black;
    const int tf = target % 8, tr = target / 8;

    auto has = [&](Bitboard bb, int f, int r) {
        if (f < 0 || f > 7 || r < 0 || r > 7) return false;
        return ((bb >> (r * 8 + f)) & 1ULL) != 0;
    };

    // pawns: a white pawn attacks diagonally "forward" (increasing rank)
    if (by_white) {
        if (has(atk.pawns, tf - 1, tr - 1) || has(atk.pawns, tf + 1, tr - 1)) return true;
    } else {
        if (has(atk.pawns, tf - 1, tr + 1) || has(atk.pawns, tf + 1, tr + 1)) return true;
    }

    static const int kd[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    for (auto& d : kd)
        if (has(atk.knights, tf + d[0], tr + d[1])) return true;

    for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr)
            if ((df || dr) && has(atk.king, tf + df, tr + dr)) return true;

    const Bitboard occ = b.complete_board;
    auto ray = [&](int df, int dr, Bitboard mask) -> bool {
        int f = tf + df, r = tr + dr;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            const int sq = r * 8 + f;
            const Bitboard bit = 1ULL << sq;
            if (occ & bit) return (mask & bit) != 0;
            f += df;
            r += dr;
        }
        return false;
    };
    const Bitboard diag = atk.bishop | atk.queen;
    if (ray(1, 1, diag) || ray(1, -1, diag) || ray(-1, 1, diag) || ray(-1, -1, diag)) return true;
    const Bitboard orth = atk.rooks | atk.queen;
    if (ray(1, 0, orth) || ray(-1, 0, orth) || ray(0, 1, orth) || ray(0, -1, orth)) return true;
    return false;
}

void report_and_abort(const chess_board& before, Move m, const char* why) {
    std::fprintf(stderr, "\n=== ILLEGAL MOVE FOUND: %s ===\n", why);
    std::fprintf(stderr, "FEN before move: %s\n", board_to_fen(before).c_str());
    std::fprintf(stderr, "move: %s\n", m.move_to_string(before.whites_turn).c_str());
    std::exit(1);
}

int main() {
    init_engine_tables();

    std::mt19937 rng(1234567);
    constexpr int kGames = 30000;
    constexpr int kMaxPlies = 150;
    long total_moves = 0;

    // Pure-uniform random play rarely sets up pins/en-passant-pins/discovered
    // checks (the geometrically narrow situations this codebase's trickiest
    // king-safety code handles) - biasing toward captures makes the walk
    // look more like an actual (search- or human-driven) game, which visits
    // those tactical situations far more often, at the cost of still being
    // cheap/fast to generate.
    std::uniform_real_distribution<float> bias(0.0f, 1.0f);

    for (int g = 0; g < kGames; ++g) {
        chess_board board = fresh_start_position();

        for (int ply = 0; ply < kMaxPlies; ++ply) {
            MoveStacks ms;
            find_all_moves(&ms, &board);
            const int total = ms.normal_size() + ms.capture_size();
            if (total == 0) break;

            int idx;
            if (ms.capture_size() > 0 && bias(rng) < 0.6f) {
                std::uniform_int_distribution<int> pick_cap(0, ms.capture_size() - 1);
                idx = ms.normal_size() + pick_cap(rng);
            } else {
                std::uniform_int_distribution<int> pick(0, total - 1);
                idx = pick(rng);
            }
            const Move move = idx < ms.normal_size() ? ms.normal_moves[idx] : ms.capture_moves[idx - ms.normal_size()];

            const chess_board before = board;
            const bool white_moving = board.whites_turn;

            StateInfo st;
            make_move(&board, move, st);
            ++total_moves;

            if (__builtin_popcountll(board.white.king) != 1)
                report_and_abort(before, move, "white king bitboard doesn't have exactly one bit after this move");
            if (__builtin_popcountll(board.black.king) != 1)
                report_and_abort(before, move, "black king bitboard doesn't have exactly one bit after this move");

            const int mover_king_sq =
                white_moving ? __builtin_ctzll(board.white.king) : __builtin_ctzll(board.black.king);
            if (square_attacked_bruteforce(board, mover_king_sq, !white_moving))
                report_and_abort(before, move, "this move left the mover's own king in check (should be illegal)");
        }
    }

    std::printf("checked %ld moves across %d games: no illegal moves found\n", total_moves, kGames);
    return 0;
}
