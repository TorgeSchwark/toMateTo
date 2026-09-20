#include "toMateTo.h"

const int MATE_SCORE = 99998;
const int DELTA_MARGIN = 50;
const int ALPHA_START = -99999;
const int BETA_START = 99999;

std::string alpha_beta_tt_toMateTo(std::string fen, double time_limit)
{
    chess_board board;
    setup_fen_position(board, fen);

    Move moves[256];
    Move* end = find_all_moves(moves, &board);
    if (moves == end) return "";

    const int count = static_cast<int>(end - moves);
    Move best_move = moves[0];
    int scores[256];

    const auto start = std::chrono::steady_clock::now();
    int depth = 1;

    for (; depth <= 64; ++depth)
    {
        int alpha = ALPHA_START, beta = BETA_START;
        int best_score = ALPHA_START;
        Move iter_best = best_move;

        for (int i = 0; i < count; ++i)
        {
            StateInfo st;
            make_move(&board, moves[i], st);
            int score = -alpha_beta(&board, depth - 1, -beta, -alpha, 1);
            undo_move(&board, moves[i], st);

            scores[i] = score;
            if (score > best_score) { best_score = score; iter_best = moves[i]; }
            alpha = std::max(alpha, score);
        }
        best_move = iter_best;

        // Sortieren: bester Zug zuerst, Rest nach Score
        int order[256];
        for (int i = 0; i < count; ++i) order[i] = i;
        std::stable_sort(order, order + count, [&](int a, int b) {
            bool ab = moves[a].move == best_move.move;
            bool bb = moves[b].move == best_move.move;
            if (ab != bb) return ab;
            return scores[a] > scores[b];
        });
        Move sorted[256];
        for (int i = 0; i < count; ++i) sorted[i] = moves[order[i]];
        std::copy(sorted, sorted + count, moves);

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= time_limit) break;
    }

    std::cerr << "depth " << depth << "\n";
    return best_move.move_to_string(board.whites_turn);
}


int alpha_beta(chess_board* board, int depth, int alpha, int beta, int root_dist){

    if (depth == 0){
        return quiescence(board, alpha, beta, root_dist);
    }

    uint64_t hash = calculate_hash(board);

    TTEntry& entry =
        transposition_table[hash & (TT_SIZE - 1)];

    if (entry.key == hash && entry.depth >= depth){
        if (entry.flag == LOWERBOUND){
            if (entry.score >= beta){
                return entry.score;
            }
        }else if (entry.flag == UPPERBOUND){
            if (entry.score <= alpha){
                return entry.score;
            }
        }else{
            return entry.score;
        }
    }

    int original_alpha = alpha;

    Move moves[256];
    Move* end = find_all_moves(moves, board);

    if (moves == end){
        if (is_in_check(board))
            return -MATE_SCORE + root_dist;

        return 0; // stalemate
    }

    int best_score = ALPHA_START;
    Move best_move{};

    if (entry.key == hash){
        for (Move* m = moves; m != end; ++m){
            if (m->move == entry.best_move.move){
                std::swap(moves[0], *m);
                break;
            }
        }
    }

    for (Move* m = moves; m != end; ++m){
        StateInfo st;

        make_move(board, *m, st);

        int score = -alpha_beta(board, depth -1, -beta, -alpha, root_dist+1);

        undo_move(board, *m, st);

        if (score > best_score){
            best_score = score;
            best_move = *m;
        }

        alpha = std::max(alpha, score);

        if (alpha >= beta){
            break;
        }
    }

    TTFlag flag;

    if (best_score <= original_alpha){
        flag = UPPERBOUND;
    }else if (best_score >= beta){
        flag = LOWERBOUND;
    }else{
        flag = EXACT;
    }

    entry.key = hash;
    entry.best_move = best_move;
    entry.score = best_score;
    entry.depth = depth;
    entry.flag = flag;

    return best_score;
}


int mvv_lva_score(chess_board* board, Move m)
{      
    PieceType victim;
    PieceType attacker;
    if(board->whites_turn){
        attacker = piece_on(board->white, m.from_sq());
        victim   = piece_on(board->black, m.to_sq());
    }else{
        attacker = piece_on(board->black, m.from_sq());
        victim   = piece_on(board->white, m.to_sq());
    }
    
    return PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker];
}

int mvv_lva_score_pesto(chess_board* board, Move m, int gamePhase)
{
    bool is_white = board->whites_turn;

    PieceType attacker;
    PieceType victim;

    if (is_white){
        attacker = piece_on(board->white, m.from_sq());
        victim   = piece_on(board->black, m.to_sq());
    }else{
        attacker = piece_on(board->black, m.from_sq());
        victim   = piece_on(board->white, m.to_sq());
    }

    // Game State EINMAL bestimmen

    int attacker_value = pesto_piece_value(attacker, m.from_sq(), is_white, gamePhase, board->ep_square);

    int victim_value = pesto_piece_value(victim, m.to_sq(), !is_white, gamePhase, board->ep_square);

    return victim_value * 1000 - attacker_value;
}


struct ScoredMove{Move move; int score; int victim_value;};

void sort_capture_moves(Move* moves, Move* end, int* victim_values, chess_board* board, int game_phase){
    const int count = static_cast<int>(end - moves);
    ScoredMove scored[256];

    const bool is_white = board->whites_turn;

    for (int i = 0; i < count; ++i){
        Move move = moves[i];

        PieceType attacker;
        PieceType victim;

        if (is_white){
            attacker = piece_on(board->white, move.from_sq());
            victim   = piece_on(board->black, move.to_sq());
        }else{
            attacker = piece_on(board->black, move.from_sq());
            victim   = piece_on(board->white, move.to_sq());
        }

        int attacker_value = pesto_piece_value(attacker, move.from_sq(), is_white, game_phase, board->ep_square);

        int victim_value = pesto_piece_value(victim, move.to_sq(), !is_white, game_phase, board->ep_square);

        scored[i] = {move, victim_value * 1000 - attacker_value, victim_value};
    }

    std::sort(scored, scored + count, [](const ScoredMove& a, const ScoredMove& b){ return a.score > b.score;});

    for (int i = 0; i < count; ++i){
        moves[i] = scored[i].move;
        victim_values[i] = scored[i].victim_value;
    }
}

bool delta_pruning(chess_board* board, int alpha, Move move, int game_phase, int eval){
    PieceType victim;

    if(board->whites_turn){
        victim = piece_on(board->black, move.to_sq());
    }else{
        victim = piece_on(board->white, move.to_sq());
    }

    int victim_value = pesto_piece_value(victim, move.to_sq(), !board->whites_turn, game_phase, board->ep_square);

    if (eval + victim_value + DELTA_MARGIN <= alpha){
        return true;
    }

    return false;

}

int quiescence(chess_board* board, int alpha, int beta, int root_dist)
{

    if (is_in_check(board)){

        Move moves[256];
        Move* end = find_all_moves(moves, board);

        if (moves == end)
            return -MATE_SCORE + root_dist; // Schachmatt

        for (Move* m = moves; m != end; ++m){
    
            StateInfo st;

            make_move(board, *m, st);
            int score = -quiescence(board, -beta, -alpha, root_dist);

            undo_move(board, *m, st);

            if (score >= beta)
                return beta;

            if (score > alpha)
                alpha = score;
        }

        return alpha;
    }

    int game_phase = pesto_game_phase(board);

    int stand_pat = pesto_eval(board, &board->white, &board->black);

    if (stand_pat >= beta)
        return beta;

    if (stand_pat > alpha)
        alpha = stand_pat;

   // nur Captures
    Move moves[256];
    int victim_values[256];

    Move* end = find_all_capture_moves(moves, board);

    sort_capture_moves(moves, end, victim_values, board, game_phase);

    const int count = static_cast<int>(end - moves);

    for (int i = 0; i < count; ++i)
    {
        Move move = moves[i];

        if (move.move_flag() != 1 && stand_pat + victim_values[i] + DELTA_MARGIN <= alpha)
        {
            continue;
        }

        StateInfo st;

        make_move(board, move, st);

        int score = -quiescence(board, -beta, -alpha, root_dist);

        undo_move(board, move, st);

        if (score >= beta)
            return beta;

        if (score > alpha)
            alpha = score;
    }

    return alpha;
}
