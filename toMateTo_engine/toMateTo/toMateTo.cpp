#include "toMateTo.h"

#include <algorithm>
#include <chrono>
#include <iostream>

const int MATE_SCORE = 99998;
const int DELTA_MARGIN = 50;
const int ALPHA_START = -99999;
const int BETA_START = 99999;

// ============================================================
//  Profiler nur bei Bedarf (-DENABLE_PROFILER)
// ============================================================
#ifdef ENABLE_PROFILER
    #define PROF(x)        (x)
    #define PROF_SCOPE(n)  Profiler::Scope profile(n)
    #define PROF_STOP()    profile.stop()
    #define PROF_START()   profile.start()
#else
    #define PROF(x)        ((void)0)
    #define PROF_SCOPE(n)
    #define PROF_STOP()    ((void)0)
    #define PROF_START()   ((void)0)
#endif

// ============================================================
//  Suchzustand (pro Thread) + Killer Moves
// ============================================================
namespace {

constexpr int MAX_PLY = 128;

// Sortierwerte
constexpr int SCORE_TT       = 2000000;
constexpr int SCORE_CAPTURE  = 1000000;   // + MVV-LVA
constexpr int SCORE_PROMO_Q  = 900000;    // ruhige Damen-Umwandlung
constexpr int SCORE_KILLER1  = 800000;
constexpr int SCORE_KILLER2  = 700000;

struct SearchState {
    Move killers[MAX_PLY][2];
    uint64_t nodes;
    bool stop;
    std::chrono::steady_clock::time_point deadline;
};

thread_local SearchState S;

struct OrderedMove {
    Move move;
    PieceType victim;   // rohes piece_on-Ergebnis (bei En Passant NO_PIECE_TYPE)
    int score;
};

inline bool out_of_time()
{
    if ((++S.nodes & 2047) == 0 &&
        std::chrono::steady_clock::now() >= S.deadline)
        S.stop = true;
    return S.stop;
}

inline void store_killer(int ply, Move m)
{
    if (ply >= MAX_PLY) return;
    if (S.killers[ply][0].move != m.move) {
        S.killers[ply][1] = S.killers[ply][0];
        S.killers[ply][0] = m;
    }
}

inline void pick_best(OrderedMove* list, int i, int n)
{
    int best = i;
    for (int j = i + 1; j < n; ++j)
        if (list[j].score > list[best].score)
            best = j;
    if (best != i)
        std::swap(list[i], list[best]);
}

// Schlagzüge bewerten (MVV-LVA)
inline int score_captures(OrderedMove* list, int n, const Move* begin, const Move* end,
                          const one_side& us, const one_side& them, int tt_move)
{
    for (const Move* m = begin; m != end; ++m) {
        const int flag = m->move_flag();
        const PieceType attacker = piece_on(us, m->from_sq());
        const PieceType victim   = piece_on(them, m->to_sq());
        const PieceType v        = (flag == 2) ? PAWN : victim;

        int s = SCORE_CAPTURE + PIECE_VALUE[v] * 256 - PIECE_VALUE[attacker];
        if (flag == 1 && m->promo_piece() == QUEEN)
            s += 100000;
        if (m->move == tt_move)
            s = SCORE_TT;

        list[n++] = { *m, victim, s };
    }
    return n;
}

// Alle Züge in eine bewertete Liste. ply >= MAX_PLY => keine Killer (z. B. Quiescence)
inline int build_move_list(OrderedMove* list, const MoveStacks& ms, const chess_board* b,
                           int ply, int tt_move)
{
    const one_side& us   = b->whites_turn ? b->white : b->black;
    const one_side& them = b->whites_turn ? b->black : b->white;

    int n = score_captures(list, 0, ms.capture_moves, ms.capture_end, us, them, tt_move);

    const int k1 = (ply < MAX_PLY) ? S.killers[ply][0].move : 0;
    const int k2 = (ply < MAX_PLY) ? S.killers[ply][1].move : 0;

    for (const Move* m = ms.normal_moves; m != ms.normal_end; ++m) {
        const int flag = m->move_flag();
        int s = 0;

        if (flag == 1)
            s = (m->promo_piece() == QUEEN) ? SCORE_PROMO_Q : -1;
        else if (m->move == k1)
            s = SCORE_KILLER1;
        else if (m->move == k2)
            s = SCORE_KILLER2;

        if (m->move == tt_move)
            s = SCORE_TT;

        list[n++] = { *m, NO_PIECE_TYPE, s };
    }
    return n;
}

} // namespace

// ============================================================
//  Root
// ============================================================
std::string alpha_beta_tt_toMateTo(std::string fen, double time_limit)
{
    chess_board board;
    setup_fen_position(board, fen);

    MoveStacks moves;
    find_all_moves(&moves, &board);

    if (moves.empty())
        return "";

    // Schlagzüge zuerst
    Move all_moves[256];
    int count = 0;
    for (Move* m = moves.capture_moves; m != moves.capture_end; ++m) all_moves[count++] = *m;
    for (Move* m = moves.normal_moves;  m != moves.normal_end;  ++m) all_moves[count++] = *m;

    if (count == 1)
        return all_moves[0].move_to_string(board.whites_turn);

    // Suchzustand zurücksetzen
    for (int p = 0; p < MAX_PLY; ++p) {
        S.killers[p][0] = Move{};
        S.killers[p][1] = Move{};
    }
    S.nodes = 0;
    S.stop = false;

    const auto start = std::chrono::steady_clock::now();
    S.deadline = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(time_limit));

    int scores[256]{};
    int order[256];

    Move best_move = all_moves[0];
    int completed_depth = 0;

    for (int depth = 1; depth <= 64; ++depth)
    {
        for (int i = 0; i < count; ++i)
            order[i] = i;

        if (depth > 1)
        {
            std::stable_sort(order, order + count,
                [&](int a, int b)
                {
                    const bool a_best = all_moves[a].move == best_move.move;
                    const bool b_best = all_moves[b].move == best_move.move;
                    if (a_best != b_best)
                        return a_best;
                    return scores[a] > scores[b];
                });
        }

        int alpha = ALPHA_START;
        const int beta = BETA_START;

        int best_score = ALPHA_START;
        Move iter_best = best_move;

        for (int i = 0; i < count; ++i)
        {
            const int index = order[i];
            const Move move = all_moves[index];

            StateInfo st;
            make_move(&board, move, st);

            int score;
            if (i == 0) {
                score = -alpha_beta(&board, depth - 1, -beta, -alpha, 1);
            } else {
                score = -alpha_beta(&board, depth - 1, -alpha - 1, -alpha, 1);
                if (score > alpha && score < beta)
                    score = -alpha_beta(&board, depth - 1, -beta, -alpha, 1);
            }

            undo_move(&board, move, st);

            if (S.stop)
                break;

            scores[index] = score;

            if (score > best_score) {
                best_score = score;
                iter_best = move;
            }
            if (score > alpha)
                alpha = score;
        }

        if (S.stop)
            break;                       // unvollständige Iteration verwerfen

        best_move = iter_best;
        completed_depth = depth;

        if (best_score > MATE_SCORE - 200)   // Matt gefunden
            break;

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= time_limit)
            break;
    }

    return best_move.move_to_string(board.whites_turn);
}

// ============================================================
//  Alpha-Beta
// ============================================================
int alpha_beta(chess_board* board, int depth, int alpha, int beta, int root_dist)
{
    if (depth <= 0)
        return quiescence(board, alpha, beta, root_dist);

    if (out_of_time())
        return 0;

    PROF(Profiler::nodes++);
    PROF_SCOPE("alpha_beta");

    const uint64_t hash = calculate_hash(board);
    TTEntry& entry = transposition_table[hash & (TT_SIZE - 1)];
    const TTEntry tt = entry;                       // Kopie: Referenz kann in der Rekursion überschrieben werden

    PROF(Profiler::tt_lookups++);

    int tt_move = 0;
    if (tt.key == hash)
    {
        tt_move = tt.best_move.move;

        if (tt.depth >= depth)
        {
            PROF(Profiler::tt_hits++);

            if (tt.flag == LOWERBOUND)
            {
                if (tt.score >= beta)
                {
                    PROF(Profiler::tt_cutoffs++);
                    return tt.score;
                }
            }
            else if (tt.flag == UPPERBOUND)
            {
                if (tt.score <= alpha)
                {
                    PROF(Profiler::tt_cutoffs++);
                    return tt.score;
                }
            }
            else
            {
                PROF(Profiler::tt_cutoffs++);
                return tt.score;
            }
        }
    }

    const int original_alpha = alpha;

    MoveStacks moves;
    find_all_moves(&moves, board);

    if (moves.empty())
    {
        // attack_count wurde von find_all_moves für die Seite am Zug gesetzt
        if (board->attack_count)
            return -MATE_SCORE + root_dist;
        return 0;
    }
    const bool in_check = board->attack_count != 0;

    OrderedMove list[256];

    const one_side& me = board->whites_turn ? board->white : board->black;
    const bool has_pieces = (me.knights | me.bishop | me.rooks | me.queen) != 0;

    if (depth >= 3 && !in_check && has_pieces &&
        beta < MATE_SCORE - 200 && beta > -MATE_SCORE + 200)
    {
        const square old_ep = board->ep_square;
        board->ep_square = SQ_NONE;
        board->whites_turn = !board->whites_turn;

        const int R = 2 + (depth >= 6);
        const int score = -alpha_beta(board, depth - 1 - R, -beta, -beta + 1, root_dist + 1);

        board->whites_turn = !board->whites_turn;
        board->ep_square = old_ep;

        if (S.stop) return 0;
        if (score >= beta) return beta;
    }
    const int n = build_move_list(list, moves, board, root_dist, tt_move);

    const Bitboard enemy_occ = board->whites_turn ? board->black.side_all
                                                  : board->white.side_all;

    int best_score = ALPHA_START;
    Move best_move{};
    bool cutoff = false;

    for (int i = 0; i < n; ++i)
    {
        pick_best(list, i, n);
        const Move move = list[i].move;

        // ruhiger Zug = kein Schlagzug, kein En Passant, keine Umwandlung
        const int flag = move.move_flag();
        const bool quiet = !((enemy_occ >> move.to_sq()) & 1ULL) && flag != 2 && flag != 1;

        StateInfo st;
        make_move(board, move, st);

        PROF_STOP();

        int score;
        if (i == 0) {
            score = -alpha_beta(board, depth - 1, -beta, -alpha, root_dist + 1);
        } else {
            // nur "gewöhnliche" ruhige Züge (keine Killer, kein TT-Zug, keine Umwandlung)
            const int r = (quiet && !in_check && depth >= 3 && i >= 3 && list[i].score == 0)
                            ? (i >= 8 ? 2 : 1) : 0;

            score = -alpha_beta(board, depth - 1 - r, -alpha - 1, -alpha, root_dist + 1);
            if (score > alpha && r > 0)                                   // Reduktion war zu optimistisch
                score = -alpha_beta(board, depth - 1, -alpha - 1, -alpha, root_dist + 1);
            if (score > alpha && score < beta)
                score = -alpha_beta(board, depth - 1, -beta, -alpha, root_dist + 1);
        }

        PROF_START();

        undo_move(board, move, st);

        if (S.stop)
            return 0;                                // nichts in die TT schreiben

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }

        if (score > alpha)
            alpha = score;

        if (alpha >= beta)
        {
            cutoff = true;
            if (quiet)
                store_killer(root_dist, move);
            break;
        }
    }

    TTFlag flag;
    if (cutoff)
        flag = LOWERBOUND;
    else if (best_score <= original_alpha)
        flag = UPPERBOUND;
    else
        flag = EXACT;

    if (entry.key != hash || entry.depth <= depth)
    {
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
int quiescence(chess_board* board, int alpha, int beta, int root_dist)
{
    if (out_of_time())
        return 0;

    PROF(Profiler::q_nodes++);
    PROF_SCOPE("quiescence");

    // ---------- im Schach: alle Ausweichzüge ----------
    if (is_in_check(board))
    {
        MoveStacks moves;
        find_all_moves(&moves, board);

        if (moves.empty())
            return -MATE_SCORE + root_dist;

        OrderedMove list[256];
        const int n = build_move_list(list, moves, board, MAX_PLY, 0);

        for (int i = 0; i < n; ++i)
        {
            pick_best(list, i, n);
            const Move move = list[i].move;

            StateInfo st;
            make_move(board, move, st);

            PROF_STOP();
            const int score = -quiescence(board, -beta, -alpha, root_dist + 1);
            PROF_START();

            undo_move(board, move, st);

            if (S.stop)
                return 0;

            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }
        return alpha;
    }

    // ---------- Stand-Pat ----------
    PROF_STOP();
    const PestoEvalResult eval = pesto_eval(board, &board->white, &board->black);
    PROF_START();

    const int stand_pat = eval.score;
    const int game_phase = eval.game_phase;

    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    // ---------- Schlagzüge ----------
    MoveStacks moves;
    find_all_capture_moves(&moves, board);

    if (moves.capture_end == moves.capture_moves)
        return alpha;

    const one_side& us   = board->whites_turn ? board->white : board->black;
    const one_side& them = board->whites_turn ? board->black : board->white;

    OrderedMove list[256];
    const int n = score_captures(list, 0, moves.capture_moves, moves.capture_end, us, them, 0);

    for (int i = 0; i < n; ++i)
    {
        pick_best(list, i, n);
        const Move move = list[i].move;

        // Delta-Pruning (Pesto-Wert nur hier berechnen)
        if (move.move_flag() != 1)
        {
            const int victim_value = pesto_piece_value(
                list[i].victim, move.to_sq(), !board->whites_turn, game_phase, board->ep_square);

            if (stand_pat + victim_value + DELTA_MARGIN <= alpha)
            {
                PROF(Profiler::q_cutoffs++);
                continue;
            }
        }

        StateInfo st;
        make_move(board, move, st);

        PROF_STOP();
        const int score = -quiescence(board, -beta, -alpha, root_dist + 1);
        PROF_START();

        undo_move(board, move, st);

        if (S.stop)
            return 0;

        if (score >= beta)
            return beta;
        if (score > alpha)
            alpha = score;
    }

    return alpha;
}

// ============================================================
//  Alte Hilfsfunktionen (werden von der Suche nicht mehr benutzt,
//  bleiben für den Fall, dass andere Dateien sie aufrufen)
// ============================================================
int mvv_lva_score(chess_board* board, Move m)
{
    PieceType victim;
    PieceType attacker;
    if (board->whites_turn) {
        attacker = piece_on(board->white, m.from_sq());
        victim   = piece_on(board->black, m.to_sq());
    } else {
        attacker = piece_on(board->black, m.from_sq());
        victim   = piece_on(board->white, m.to_sq());
    }
    return PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker];
}

int mvv_lva_score_pesto(chess_board* board, Move m, int gamePhase)
{
    const bool is_white = board->whites_turn;
    PieceType attacker;
    PieceType victim;

    if (is_white) {
        attacker = piece_on(board->white, m.from_sq());
        victim   = piece_on(board->black, m.to_sq());
    } else {
        attacker = piece_on(board->black, m.from_sq());
        victim   = piece_on(board->white, m.to_sq());
    }

    const int attacker_value = pesto_piece_value(attacker, m.from_sq(), is_white, gamePhase, board->ep_square);
    const int victim_value   = pesto_piece_value(victim, m.to_sq(), !is_white, gamePhase, board->ep_square);

    return victim_value * 1000 - attacker_value;
}

struct ScoredMove { Move move; int score; int victim_value; };

void sort_capture_moves(Move* moves, Move* end, int* victim_values, chess_board* board, int game_phase)
{
    const int count = static_cast<int>(end - moves);
    ScoredMove scored[256];
    const bool is_white = board->whites_turn;

    for (int i = 0; i < count; ++i) {
        const Move move = moves[i];
        PieceType attacker;
        PieceType victim;

        if (is_white) {
            attacker = piece_on(board->white, move.from_sq());
            victim   = piece_on(board->black, move.to_sq());
        } else {
            attacker = piece_on(board->black, move.from_sq());
            victim   = piece_on(board->white, move.to_sq());
        }

        const int attacker_value = pesto_piece_value(attacker, move.from_sq(), is_white, game_phase, board->ep_square);
        const int victim_value   = pesto_piece_value(victim, move.to_sq(), !is_white, game_phase, board->ep_square);

        scored[i] = { move, victim_value * 1000 - attacker_value, victim_value };
    }

    std::sort(scored, scored + count, [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

    for (int i = 0; i < count; ++i) {
        moves[i] = scored[i].move;
        victim_values[i] = scored[i].victim_value;
    }
}

bool delta_pruning(chess_board* board, int alpha, Move move, int game_phase, int eval)
{
    PieceType victim = board->whites_turn ? piece_on(board->black, move.to_sq())
                                          : piece_on(board->white, move.to_sq());

    const int victim_value = pesto_piece_value(victim, move.to_sq(), !board->whites_turn, game_phase, board->ep_square);
    return eval + victim_value + DELTA_MARGIN <= alpha;
}