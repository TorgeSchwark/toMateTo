#include "toMateTo_nnue.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

const int MATE_SCORE = 99998;
const int DELTA_MARGIN = 50;
const int ALPHA_START = -99999;
const int BETA_START = 99999;

namespace {

constexpr int MAX_PLY = 128;
constexpr int ACC_STACK_SIZE = MAX_PLY + 64; // headroom over MAX_PLY - see .h comment

constexpr int SCORE_TT       = 2000000;
constexpr int SCORE_CAPTURE  = 1000000;
constexpr int SCORE_PROMO_Q  = 900000;
constexpr int SCORE_KILLER1  = 800000;
constexpr int SCORE_KILLER2  = 700000;

// Whole-board piece count range a bucket-NNUE setup must exactly partition
// - duplicated from full_cycle.cpp's identical constants rather than
// shared, same reasoning as this file's other small duplicated helpers
// (this links a different chess_board type than full_cycle.cpp's own).
constexpr int kMinPieces = 2;
constexpr int kMaxPieces = 32;

inline int piece_count_of(const chess_board& board) {
    return __builtin_popcountll(board.white.side_all | board.black.side_all);
}

struct SearchState {
    Move killers[MAX_PLY][2];
    uint64_t nodes;
    bool stop;
    std::chrono::steady_clock::time_point deadline;

    const std::vector<NnueBucket>* buckets = nullptr;
    std::vector<SearchAcc> acc_stack;  // acc_stack[root_dist] == accumulator for the position at that ply
    std::vector<int> acc_bucket;       // acc_bucket[root_dist] == index into *buckets that acc_stack[root_dist] was built with
    int last_completed_depth = 0;
};

thread_local SearchState S;

// Single-net callers (nnue_search_init_thread(), still the common case) get
// a one-entry bucket list covering the whole range, so the rest of this
// file only has to know about the bucket-aware path, not two parallel
// ones. Lives per-thread (nnue_search_init_thread() is called once per
// search thread) so its address stays valid for that thread's lifetime.
thread_local std::vector<NnueBucket> S_single_bucket_holder;

// Which bucket covers `piece_count` - linear scan over (at most a few
// dozen) buckets, called once per move, nowhere near hot enough to matter.
// Returns 0 (not -1) if none match, so a gap that slipped past
// validate_bucket_coverage() degrades to "wrong net for some positions"
// rather than an out-of-bounds access - validate_bucket_coverage() is what
// actually prevents that in practice.
int bucket_for(int piece_count) {
    for (std::size_t i = 0; i < S.buckets->size(); ++i) {
        const NnueBucket& b = (*S.buckets)[i];
        if (piece_count >= b.min_pieces && piece_count <= b.max_pieces) return static_cast<int>(i);
    }
    return 0;
}

// ---- incremental NNUE accumulator update ---------------------------------
// See toMateTo_nnue.h's header comment. Must run AFTER make_move(board, m,
// st) has already mutated `board` - it reads the moved/captured piece types
// out of `st` (make_move already filled those in) and the board's new king
// squares, rather than needing a separate "before" snapshot.

inline nnue::PieceType to_nnue_piece(PieceType pt) {
    switch (pt) {
        case PAWN:   return nnue::PAWN;
        case KNIGHT: return nnue::KNIGHT;
        case BISHOP: return nnue::BISHOP;
        case ROOK:   return nnue::ROOK;
        case QUEEN:  return nnue::QUEEN;
        default:     return nnue::PAWN; // unreachable: KING is never a tracked feature (caller excludes it)
    }
}

// Fixed-capacity, matching efficiency_check.hpp's "32 active features" bound
// (max 15 non-king pieces per side even with promotions in any realistic
// game; 32 leaves generous headroom without heap-allocating in a hot path).
struct FeatureList {
    std::array<int, 32> idx;
    int n = 0;
    void push(int f) { idx[static_cast<std::size_t>(n++)] = f; }
    const int* begin() const { return idx.data(); }
    const int* end() const { return idx.data() + n; }
};

void collect_features_for(const chess_board* b, nnue::Color perspective, int king_sq, FeatureList& out) {
    out.n = 0;
    auto scan_side = [&](const one_side& side, nnue::Color color) {
        auto scan_bb = [&](Bitboard bb, nnue::PieceType pt) {
            while (bb) {
                int sq = __builtin_ctzll(bb);
                bb &= bb - 1;
                out.push(nnue::feature_index(perspective, sq, pt, color, king_sq));
            }
        };
        scan_bb(side.pawns, nnue::PAWN);
        scan_bb(side.knights, nnue::KNIGHT);
        scan_bb(side.bishop, nnue::BISHOP);
        scan_bb(side.rooks, nnue::ROOK);
        scan_bb(side.queen, nnue::QUEEN);
    };
    scan_side(b->white, nnue::WHITE);
    scan_side(b->black, nnue::BLACK);
}

// `child` starts as a copy of `parent` (the position before this move) and
// is updated in place to represent the position after it. `board` must
// already reflect the move (make_move already called). `white_moved`: which
// side made the move (board->whites_turn has already been flipped by
// make_move by this point, so it can't be read back off the board itself).
void update_accumulator(SearchAcc& child, const SearchAcc& parent, const chess_board* board, Move m,
                         const StateInfo& st, bool white_moved, const nnue::FeatureTransformer<NNUE_ACC_SIZE>& ft) {
    child = parent;
    const square from = m.from_sq();
    const square to = m.to_sq();
    const int flag = m.move_flag();
    const nnue::Color mover_color = white_moved ? nnue::WHITE : nnue::BLACK;
    const nnue::Color other_color = white_moved ? nnue::BLACK : nnue::WHITE;
    const int king_sq_white = __builtin_ctzll(board->white.king);
    const int king_sq_black = __builtin_ctzll(board->black.king);

    if (st.moving == KING) {
        // Every feature this side's perspective sees is king-square-relative
        // (see halfkp.hpp) - a king move invalidates all of them, so refresh
        // from scratch rather than trying to patch individual features. This
        // one refresh also correctly accounts for a captured piece (if any)
        // and, for castling, the rook's new square too - all just read back
        // off the post-move board.
        const nnue::Color mover_persp = mover_color;
        const int mover_king_sq = mover_persp == nnue::WHITE ? king_sq_white : king_sq_black;
        FeatureList feats;
        collect_features_for(board, mover_persp, mover_king_sq, feats);
        child.refresh(ft, mover_persp, feats);

        // The OTHER perspective's own king didn't move, so it needs only an
        // ordinary incremental patch: the king itself isn't a tracked
        // feature for anyone, but castling's rook move and a king-capture's
        // removed piece both are.
        const nnue::Color other_persp = other_color;
        const int other_king_sq = other_persp == nnue::WHITE ? king_sq_white : king_sq_black;
        if (flag == 3) { // castling - rook squares exactly as make_move derives them
            square rook_from, rook_to;
            if (to > from) { rook_from = to + 1; rook_to = to - 1; } // kingside: h->f
            else { rook_from = to - 2; rook_to = to + 1; }           // queenside: a->d
            child.apply_remove(ft, other_persp,
                                nnue::feature_index(other_persp, rook_from, nnue::ROOK, mover_color, other_king_sq));
            child.apply_add(ft, other_persp,
                             nnue::feature_index(other_persp, rook_to, nnue::ROOK, mover_color, other_king_sq));
        } else if (st.captured != NO_PIECE_TYPE) { // plain king move that captures (never en passant/promotion)
            child.apply_remove(ft, other_persp,
                                nnue::feature_index(other_persp, to, to_nnue_piece(st.captured), other_color,
                                                     other_king_sq));
        }
        return;
    }

    // Non-king mover: both perspectives see the same physical change, just
    // indexed relative to their own (here: unaffected) king square.
    for (nnue::Color persp : {nnue::WHITE, nnue::BLACK}) {
        const int ksq = persp == nnue::WHITE ? king_sq_white : king_sq_black;
        if (flag == 1) { // promotion: the pawn disappears, the promoted piece appears
            child.apply_remove(ft, persp, nnue::feature_index(persp, from, nnue::PAWN, mover_color, ksq));
            child.apply_add(ft, persp,
                             nnue::feature_index(persp, to, to_nnue_piece(static_cast<PieceType>(m.promo_piece())),
                                                  mover_color, ksq));
        } else {
            child.apply_remove(ft, persp, nnue::feature_index(persp, from, to_nnue_piece(st.moving), mover_color, ksq));
            child.apply_add(ft, persp, nnue::feature_index(persp, to, to_nnue_piece(st.moving), mover_color, ksq));
        }
        if (st.captured != NO_PIECE_TYPE) {
            if (flag == 2) { // en passant - captured pawn is NOT on `to`
                const square cap_sq = static_cast<square>(to + (white_moved ? -8 : 8));
                child.apply_remove(ft, persp, nnue::feature_index(persp, cap_sq, nnue::PAWN, other_color, ksq));
            } else {
                child.apply_remove(ft, persp,
                                    nnue::feature_index(persp, to, to_nnue_piece(st.captured), other_color, ksq));
            }
        }
    }
}

inline int nnue_eval(const chess_board* board, int root_dist) {
    const std::size_t slot = static_cast<std::size_t>(root_dist);
    const SearchNet* net = (*S.buckets)[static_cast<std::size_t>(S.acc_bucket[slot])].net;
    return static_cast<int>(net->evaluate_cp(S.acc_stack[slot], board->whites_turn ? nnue::WHITE : nnue::BLACK));
}

inline bool out_of_time() {
    if ((++S.nodes & 2047) == 0 && std::chrono::steady_clock::now() >= S.deadline) S.stop = true;
    return S.stop;
}

inline void store_killer(int ply, Move m) {
    if (ply >= MAX_PLY) return;
    if (S.killers[ply][0].move != m.move) {
        S.killers[ply][1] = S.killers[ply][0];
        S.killers[ply][0] = m;
    }
}

struct OrderedMove {
    Move move;
    PieceType victim;
    int score;
};

inline void pick_best(OrderedMove* list, int i, int n) {
    int best = i;
    for (int j = i + 1; j < n; ++j)
        if (list[j].score > list[best].score) best = j;
    if (best != i) std::swap(list[i], list[best]);
}

inline int score_captures(OrderedMove* list, int n, const Move* begin, const Move* end, const one_side& us,
                           const one_side& them, int tt_move) {
    for (const Move* m = begin; m != end; ++m) {
        const int flag = m->move_flag();
        const PieceType attacker = piece_on(us, m->from_sq());
        const PieceType victim = piece_on(them, m->to_sq());
        const PieceType v = (flag == 2) ? PAWN : victim;

        int s = SCORE_CAPTURE + PIECE_VALUE[v] * 256 - PIECE_VALUE[attacker];
        if (flag == 1 && m->promo_piece() == QUEEN) s += 100000;
        if (m->move == tt_move) s = SCORE_TT;

        list[n++] = {*m, victim, s};
    }
    return n;
}

inline int build_move_list(OrderedMove* list, const MoveStacks& ms, const chess_board* b, int ply, int tt_move) {
    const one_side& us = b->whites_turn ? b->white : b->black;
    const one_side& them = b->whites_turn ? b->black : b->white;

    int n = score_captures(list, 0, ms.capture_moves, ms.capture_end, us, them, tt_move);

    const int k1 = (ply < MAX_PLY) ? S.killers[ply][0].move : 0;
    const int k2 = (ply < MAX_PLY) ? S.killers[ply][1].move : 0;

    for (const Move* m = ms.normal_moves; m != ms.normal_end; ++m) {
        const int flag = m->move_flag();
        int s = 0;
        if (flag == 1) s = (m->promo_piece() == QUEEN) ? SCORE_PROMO_Q : -1;
        else if (m->move == k1) s = SCORE_KILLER1;
        else if (m->move == k2) s = SCORE_KILLER2;
        if (m->move == tt_move) s = SCORE_TT;
        list[n++] = {*m, NO_PIECE_TYPE, s};
    }
    return n;
}

} // namespace

std::string validate_bucket_coverage(std::vector<NnueBucket> buckets) {
    if (buckets.empty()) return "keine Buckets angegeben";
    for (const NnueBucket& b : buckets)
        if (!b.net) return "Bucket " + std::to_string(b.min_pieces) + "-" + std::to_string(b.max_pieces) +
                            " hat kein geladenes Netz";
    std::sort(buckets.begin(), buckets.end(), [](const NnueBucket& a, const NnueBucket& b) {
        return a.min_pieces < b.min_pieces;
    });
    if (buckets.front().min_pieces != kMinPieces)
        return "Abdeckung beginnt bei " + std::to_string(buckets.front().min_pieces) + " Figuren, muss aber bei " +
               std::to_string(kMinPieces) + " beginnen";
    for (std::size_t i = 0; i + 1 < buckets.size(); ++i) {
        if (buckets[i].max_pieces + 1 < buckets[i + 1].min_pieces)
            return "Luecke zwischen " + std::to_string(buckets[i].max_pieces) + " und " +
                   std::to_string(buckets[i + 1].min_pieces) + " Figuren - nicht abgedeckt";
        if (buckets[i].max_pieces >= buckets[i + 1].min_pieces)
            return "Ueberlappung: Bucket " + std::to_string(buckets[i].min_pieces) + "-" +
                   std::to_string(buckets[i].max_pieces) + " und " + std::to_string(buckets[i + 1].min_pieces) + "-" +
                   std::to_string(buckets[i + 1].max_pieces);
    }
    if (buckets.back().max_pieces != kMaxPieces)
        return "Abdeckung endet bei " + std::to_string(buckets.back().max_pieces) + " Figuren, muss aber bei " +
               std::to_string(kMaxPieces) + " enden";
    return "";
}

void nnue_search_init_thread(const SearchNet* net) {
    S_single_bucket_holder.assign(1, NnueBucket{kMinPieces, kMaxPieces, net});
    S.buckets = &S_single_bucket_holder;
    S.acc_stack.assign(ACC_STACK_SIZE, SearchAcc{});
    S.acc_bucket.assign(ACC_STACK_SIZE, 0);
}

void nnue_search_init_thread_buckets(const std::vector<NnueBucket>* buckets) {
    S.buckets = buckets;
    S.acc_stack.assign(ACC_STACK_SIZE, SearchAcc{});
    S.acc_bucket.assign(ACC_STACK_SIZE, 0);
}

void nnue_search_set_root(const chess_board& board) {
    const int bucket = bucket_for(piece_count_of(board));
    const SearchNet* net = (*S.buckets)[static_cast<std::size_t>(bucket)].net;
    FeatureList feats;
    const int king_sq_white = __builtin_ctzll(board.white.king);
    const int king_sq_black = __builtin_ctzll(board.black.king);
    collect_features_for(&board, nnue::WHITE, king_sq_white, feats);
    S.acc_stack[0].refresh(net->feature_transformer, nnue::WHITE, feats);
    collect_features_for(&board, nnue::BLACK, king_sq_black, feats);
    S.acc_stack[0].refresh(net->feature_transformer, nnue::BLACK, feats);
    S.acc_bucket[0] = bucket;
}

#ifdef NNUE_DEBUG_LEGALITY
// Independent (no magic bitboards) re-check, same as tests/test_move_
// legality.cpp - duplicated here rather than shared so this stays a single
// self-contained #ifdef block that costs nothing when NNUE_DEBUG_LEGALITY
// isn't defined. Only used to hunt the "king sometimes missing" report -
// not meant to stay compiled into normal builds (it's not free: one full
// attack re-scan per move).
bool debug_square_attacked_bruteforce(const chess_board& b, int target, bool by_white) {
    const one_side& atk = by_white ? b.white : b.black;
    const int tf = target % 8, tr = target / 8;
    auto has = [&](Bitboard bb, int f, int r) {
        if (f < 0 || f > 7 || r < 0 || r > 7) return false;
        return ((bb >> (r * 8 + f)) & 1ULL) != 0;
    };
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

void debug_check_legality(const chess_board& before, const chess_board& after, Move m, bool white_moved) {
    if (__builtin_popcountll(after.white.king) != 1 || __builtin_popcountll(after.black.king) != 1) {
        std::fprintf(stderr, "\n=== NNUE_DEBUG_LEGALITY: a king disappeared ===\nFEN before: %s\nmove: %s\n",
                      board_to_fen(before).c_str(), m.move_to_string(before.whites_turn).c_str());
        std::abort();
    }
    const int mover_king_sq = white_moved ? __builtin_ctzll(after.white.king) : __builtin_ctzll(after.black.king);
    if (debug_square_attacked_bruteforce(after, mover_king_sq, !white_moved)) {
        std::fprintf(stderr,
                      "\n=== NNUE_DEBUG_LEGALITY: move left the mover's own king in check ===\n"
                      "FEN before: %s\nmove: %s\n",
                      board_to_fen(before).c_str(), m.move_to_string(before.whites_turn).c_str());
        std::abort();
    }
}
#endif

// Applies `move` to `board` (make_move) AND keeps S.acc_stack[root_dist+1]
// consistent with the resulting position, derived incrementally from
// S.acc_stack[root_dist]. Every make_move call site in this file goes
// through this wrapper instead of calling make_move directly, specifically
// so the accumulator is never recomputed from scratch - see toMateTo_nnue.h.
inline void search_make_move(chess_board* board, Move m, StateInfo& st, int root_dist) {
    const bool white_moved = board->whites_turn;
#ifdef NNUE_DEBUG_LEGALITY
    const chess_board before = *board;
#endif
    make_move(board, m, st);
#ifdef NNUE_DEBUG_LEGALITY
    debug_check_legality(before, *board, m, white_moved);
#endif
    const std::size_t child_slot = static_cast<std::size_t>(root_dist + 1);
    if (child_slot < S.acc_stack.size()) {
        const std::size_t parent_slot = static_cast<std::size_t>(root_dist);
        const int new_bucket = bucket_for(piece_count_of(*board));
        const SearchNet* net = (*S.buckets)[static_cast<std::size_t>(new_bucket)].net;
        if (new_bucket == S.acc_bucket[parent_slot]) {
            update_accumulator(S.acc_stack[child_slot], S.acc_stack[parent_slot], board, m, st, white_moved,
                                net->feature_transformer);
        } else {
            // Crossed into a different bucket's net (multi-bucket setups
            // only - single-net callers always have exactly one bucket, so
            // this branch is unreachable for them). Accumulators from
            // different nets' weight spaces aren't compatible with each
            // other, so this can never be incremental regardless of what
            // the move itself was - full refresh with the new net, for
            // both perspectives.
            FeatureList wf, bf;
            const int ksq_w = __builtin_ctzll(board->white.king);
            const int ksq_b = __builtin_ctzll(board->black.king);
            collect_features_for(board, nnue::WHITE, ksq_w, wf);
            collect_features_for(board, nnue::BLACK, ksq_b, bf);
            S.acc_stack[child_slot].refresh(net->feature_transformer, nnue::WHITE, wf);
            S.acc_stack[child_slot].refresh(net->feature_transformer, nnue::BLACK, bf);
        }
        S.acc_bucket[child_slot] = new_bucket;
    }
    // If the (pathological, practically unreachable within any real time
    // budget - see the .h's ACC_STACK_SIZE comment) search line exceeds the
    // stack's headroom, the eval at that ply would read stale/wrong data;
    // out_of_time()'s node-count-based deadline check makes actually
    // reaching that depth within a normal search essentially impossible.
}

inline void search_undo_move(chess_board* board, Move m, const StateInfo& st) { undo_move(board, m, st); }

bool nnue_search_debug_check_move(chess_board* board, Move m) {
    const bool white_moved = board->whites_turn;
    const SearchNet* net = (*S.buckets)[static_cast<std::size_t>(S.acc_bucket[0])].net;
    StateInfo st;
    const SearchAcc parent = S.acc_stack[0];
    make_move(board, m, st);

    SearchAcc incremental;
    update_accumulator(incremental, parent, board, m, st, white_moved, net->feature_transformer);

    SearchAcc refreshed;
    FeatureList feats;
    const int ksq_w = __builtin_ctzll(board->white.king);
    const int ksq_b = __builtin_ctzll(board->black.king);
    collect_features_for(board, nnue::WHITE, ksq_w, feats);
    refreshed.refresh(net->feature_transformer, nnue::WHITE, feats);
    collect_features_for(board, nnue::BLACK, ksq_b, feats);
    refreshed.refresh(net->feature_transformer, nnue::BLACK, feats);

    S.acc_stack[0] = incremental; // carry forward so the next call continues the same game

    return incremental.values == refreshed.values;
}

// ============================================================
//  Root
// ============================================================
std::string alpha_beta_tt_toMateTo(std::string fen, double time_limit) {
    chess_board board;
    setup_fen_position(board, fen);
    nnue_search_set_root(board);

    MoveStacks moves;
    find_all_moves(&moves, &board);
    if (moves.empty()) return "";

    Move all_moves[256];
    int count = 0;
    for (Move* m = moves.capture_moves; m != moves.capture_end; ++m) all_moves[count++] = *m;
    for (Move* m = moves.normal_moves; m != moves.normal_end; ++m) all_moves[count++] = *m;

    if (count == 1) return all_moves[0].move_to_string(board.whites_turn);

    for (int p = 0; p < MAX_PLY; ++p) {
        S.killers[p][0] = Move{};
        S.killers[p][1] = Move{};
    }
    S.nodes = 0;
    S.stop = false;

    const auto start = std::chrono::steady_clock::now();
    S.deadline =
        start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(time_limit));

    int scores[256]{};
    int order[256];

    Move best_move = all_moves[0];

    for (int depth = 1; depth <= 64; ++depth) {
        for (int i = 0; i < count; ++i) order[i] = i;

        if (depth > 1) {
            std::stable_sort(order, order + count, [&](int a, int b) {
                const bool a_best = all_moves[a].move == best_move.move;
                const bool b_best = all_moves[b].move == best_move.move;
                if (a_best != b_best) return a_best;
                return scores[a] > scores[b];
            });
        }

        int alpha = ALPHA_START;
        const int beta = BETA_START;
        int best_score = ALPHA_START;
        Move iter_best = best_move;

        for (int i = 0; i < count; ++i) {
            const int index = order[i];
            const Move move = all_moves[index];

            StateInfo st;
            search_make_move(&board, move, st, 0);

            int score;
            if (i == 0) {
                score = -alpha_beta(&board, depth - 1, -beta, -alpha, 1);
            } else {
                score = -alpha_beta(&board, depth - 1, -alpha - 1, -alpha, 1);
                if (score > alpha && score < beta) score = -alpha_beta(&board, depth - 1, -beta, -alpha, 1);
            }

            search_undo_move(&board, move, st);

            if (S.stop) break;

            scores[index] = score;
            if (score > best_score) {
                best_score = score;
                iter_best = move;
            }
            if (score > alpha) alpha = score;
        }

        if (S.stop) break;

        best_move = iter_best;
        S.last_completed_depth = depth;

        if (best_score > MATE_SCORE - 200) break;

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= time_limit) break;
    }

    return best_move.move_to_string(board.whites_turn);
}

// Iterative-deepening depth actually completed by the most recent
// alpha_beta_tt_toMateTo() call on this thread (0 if none yet) - see
// strength_match.cpp, which averages this across all TestEngine moves to
// answer "how deep is it actually searching under this --movetime-ms" (a
// slow/degenerate average here, e.g. always stuck at depth 1-2, is the kind
// of thing that would explain "wins fast games, loses long ones": a time-
// management or contention problem starving the search, not necessarily an
// eval/training problem).
int nnue_search_last_depth() { return S.last_completed_depth; }

// ============================================================
//  Alpha-Beta
// ============================================================
int alpha_beta(chess_board* board, int depth, int alpha, int beta, int root_dist) {
    if (depth <= 0) return quiescence(board, alpha, beta, root_dist);
    if (out_of_time()) return 0;

    const uint64_t hash = calculate_hash(board);
    TTEntry& entry = transposition_table[hash & (TT_SIZE - 1)];
    const TTEntry tt = entry;

    int tt_move = 0;
    if (tt.key == hash) {
        tt_move = tt.best_move.move;
        if (tt.depth >= depth) {
            if (tt.flag == LOWERBOUND) {
                if (tt.score >= beta) return tt.score;
            } else if (tt.flag == UPPERBOUND) {
                if (tt.score <= alpha) return tt.score;
            } else {
                return tt.score;
            }
        }
    }

    const int original_alpha = alpha;

    MoveStacks moves;
    find_all_moves(&moves, board);

    if (moves.empty()) {
        if (board->attack_count) return -MATE_SCORE + root_dist;
        return 0;
    }
    const bool in_check = board->attack_count != 0;

    OrderedMove list[256];
    const one_side& me = board->whites_turn ? board->white : board->black;
    const bool has_pieces = (me.knights | me.bishop | me.rooks | me.queen) != 0;

    if (depth >= 3 && !in_check && has_pieces && beta < MATE_SCORE - 200 && beta > -MATE_SCORE + 200) {
        const square old_ep = board->ep_square;
        board->ep_square = SQ_NONE;
        board->whites_turn = !board->whites_turn;

        // Null move doesn't touch any piece, so the accumulator is
        // unaffected - just reuse this ply's slot for the child ply too
        // (both represent "the same pieces, other side to move").
        const std::size_t slot = static_cast<std::size_t>(root_dist);
        const std::size_t child_slot = static_cast<std::size_t>(root_dist + 1);
        if (child_slot < S.acc_stack.size()) S.acc_stack[child_slot] = S.acc_stack[slot];

        const int R = 2 + (depth >= 6);
        const int score = -alpha_beta(board, depth - 1 - R, -beta, -beta + 1, root_dist + 1);

        board->whites_turn = !board->whites_turn;
        board->ep_square = old_ep;

        if (S.stop) return 0;
        if (score >= beta) return beta;
    }

    const int n = build_move_list(list, moves, board, root_dist, tt_move);
    const Bitboard enemy_occ = board->whites_turn ? board->black.side_all : board->white.side_all;

    int best_score = ALPHA_START;
    Move best_move{};
    bool cutoff = false;

    for (int i = 0; i < n; ++i) {
        pick_best(list, i, n);
        const Move move = list[i].move;

        const int flag = move.move_flag();
        const bool quiet = !((enemy_occ >> move.to_sq()) & 1ULL) && flag != 2 && flag != 1;

        StateInfo st;
        search_make_move(board, move, st, root_dist);

        int score;
        if (i == 0) {
            score = -alpha_beta(board, depth - 1, -beta, -alpha, root_dist + 1);
        } else {
            const int r = (quiet && !in_check && depth >= 3 && i >= 3 && list[i].score == 0) ? (i >= 8 ? 2 : 1) : 0;
            score = -alpha_beta(board, depth - 1 - r, -alpha - 1, -alpha, root_dist + 1);
            if (score > alpha && r > 0) score = -alpha_beta(board, depth - 1, -alpha - 1, -alpha, root_dist + 1);
            if (score > alpha && score < beta) score = -alpha_beta(board, depth - 1, -beta, -alpha, root_dist + 1);
        }

        search_undo_move(board, move, st);

        if (S.stop) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        if (score > alpha) alpha = score;

        if (alpha >= beta) {
            cutoff = true;
            if (quiet) store_killer(root_dist, move);
            break;
        }
    }

    TTFlag flag;
    if (cutoff) flag = LOWERBOUND;
    else if (best_score <= original_alpha) flag = UPPERBOUND;
    else flag = EXACT;

    if (entry.key != hash || entry.depth <= depth) {
        entry.key = hash;
        entry.best_move = best_move;
        entry.score = best_score;
        entry.depth = depth;
        entry.flag = flag;
    }

    return cutoff ? beta : best_score;
}

// ============================================================
//  Quiescence
// ============================================================
int quiescence(chess_board* board, int alpha, int beta, int root_dist) {
    if (out_of_time()) return 0;

    if (is_in_check(board)) {
        MoveStacks moves;
        find_all_moves(&moves, board);
        if (moves.empty()) return -MATE_SCORE + root_dist;

        OrderedMove list[256];
        const int n = build_move_list(list, moves, board, MAX_PLY, 0);

        for (int i = 0; i < n; ++i) {
            pick_best(list, i, n);
            const Move move = list[i].move;

            StateInfo st;
            search_make_move(board, move, st, root_dist);
            const int score = -quiescence(board, -beta, -alpha, root_dist + 1);
            search_undo_move(board, move, st);

            if (S.stop) return 0;
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    }

    // ---------- Stand-Pat (NNUE, not pesto_eval) ----------
    const int stand_pat = nnue_eval(board, root_dist);
    // game_phase is still PESTO's - only used below for delta_pruning's
    // rough material-value estimate, not for the actual score (see this
    // file's header comment for why that split is fine).
    const int game_phase = pesto_game_phase(board);

    if (stand_pat >= beta) return beta;
    int local_alpha = alpha;
    if (stand_pat > local_alpha) local_alpha = stand_pat;

    MoveStacks moves;
    find_all_capture_moves(&moves, board);
    if (moves.capture_end == moves.capture_moves) return local_alpha;

    const one_side& us = board->whites_turn ? board->white : board->black;
    const one_side& them = board->whites_turn ? board->black : board->white;

    OrderedMove list[256];
    const int n = score_captures(list, 0, moves.capture_moves, moves.capture_end, us, them, 0);

    for (int i = 0; i < n; ++i) {
        pick_best(list, i, n);
        const Move move = list[i].move;

        if (move.move_flag() != 1) {
            const int victim_value =
                pesto_piece_value(list[i].victim, move.to_sq(), !board->whites_turn, game_phase, board->ep_square);
            if (stand_pat + victim_value + DELTA_MARGIN <= local_alpha) continue;
        }

        StateInfo st;
        search_make_move(board, move, st, root_dist);
        const int score = -quiescence(board, -beta, -local_alpha, root_dist + 1);
        search_undo_move(board, move, st);

        if (S.stop) return 0;
        if (score >= beta) return beta;
        if (score > local_alpha) local_alpha = score;
    }

    return local_alpha;
}
