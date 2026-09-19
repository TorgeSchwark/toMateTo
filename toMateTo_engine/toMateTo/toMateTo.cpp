#include "toMateTo.h"
#include "profiler.h"

int MATE_SCORE = 99998;
int DELTA_MARGIN = 20;


std::map<std::string, int> alpha_beta_toMaTo(std::string fen_position, int depth){

    chess_board board;
    setup_fen_position(board, fen_position);

    std::map<std::string, int> result;

    Move moves[256];
    Move* end = find_all_moves(moves, &board);

    int alpha = -99999;
    int beta  =  99999;

    for (Move* m = moves; m != end; ++m)
    {
        StateInfo st;

        std::string move_string =
            m->move_to_string(board.whites_turn);

        make_move(&board, *m, st);
        // get best move by opponent reverse it for how good the pos is for us    
        
        
        int eval = -alpha_beta(&board, depth - 1, -beta, -alpha);

        undo_move(&board, *m, st);

        result[move_string] = eval;
        // alpha is the current best move under consid of opp
        alpha = std::max(alpha, eval);
    }

    return result;
}

#include <chrono>

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
        int alpha = -100000, beta = 100000;
        int best_score = -100000;
        Move iter_best = best_move;

        for (int i = 0; i < count; ++i)
        {
            StateInfo st;
            make_move(&board, moves[i], st);
            int score = -alpha_beta(&board, depth - 1, -beta, -alpha);
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

std::string alpha_beta_toMateTo(std::string fen_position, int depth)
{
    chess_board board;
    setup_fen_position(board, fen_position);

    Move moves[256];
    Move* end = find_all_moves(moves, &board);

    int best_eval = -99999;
    std::string best_move;

    for (Move* m = moves; m != end; ++m)
    {
        StateInfo st;

        std::string move_string =
            m->move_to_string(board.whites_turn);

        make_move(&board, *m, st);

        // Für jeden Root-Zug ein vollständiges Alpha-Beta-Fenster!
        int eval = -alpha_beta_old(
            &board,
            depth - 1,
            -99999,
            99999
        );

        undo_move(&board, *m, st);

        if (eval > best_eval)
        {
            best_eval = eval;
            best_move = move_string;
        }
    }

    return best_move;
}

int alpha_beta_old(chess_board* board, int depth, int alpha, int beta)
{
    if (depth == 0)
    {
        return quiescence(board, alpha, beta);
    }

    Move moves[256];
    Move* end = find_all_moves(moves, board);

    int best_score = -99999;

    for (Move* m = moves; m != end; ++m)
    {
        StateInfo st;

        make_move(board, *m, st);

        int score = -alpha_beta_old(
            board,
            depth - 1,
            -beta,
            -alpha
        );

        undo_move(board, *m, st);

        best_score = std::max(best_score, score);
        alpha = std::max(alpha, score);

        if (alpha >= beta)
        {
            break;
        }
    }

    return best_score;
}

int alpha_beta(
    chess_board* board,
    int depth,
    int alpha,
    int beta)
{
    Profiler::Scope profile("alpha_beta");

    Profiler::nodes++;

    if (depth == 0)
    {
        profile.stop();

        return quiescence(board, alpha, beta);
        
    }

    uint64_t hash = calculate_hash(board);

    TTEntry& entry =
        transposition_table[hash & (TT_SIZE - 1)];

    Profiler::tt_lookups++;

    
    if (entry.key == hash &&
        entry.depth >= depth)
    {
        Profiler::tt_hits++;

        if (entry.flag == LOWERBOUND)
        {
            if (entry.score >= beta)
            {
                Profiler::tt_lowerbound++;
                Profiler::tt_cutoffs++;
                return entry.score;
            }
        }
        else if (entry.flag == UPPERBOUND)
        {
            if (entry.score <= alpha)
            {
                Profiler::tt_upperbound++;
                Profiler::tt_cutoffs++;
                return entry.score;
            }
        }
        else
        {
            Profiler::tt_exact++;
            Profiler::tt_cutoffs++;
            return entry.score;
        }
    }

    int original_alpha = alpha;

    Move moves[256];
    Move* end = find_all_moves(moves, board);

    int best_score = -99999;
    Move best_move{};

    if (entry.key == hash)
    {
        for (Move* m = moves; m != end; ++m)
        {
            if (m->move == entry.best_move.move)
            {
                std::swap(moves[0], *m);
                break;
            }
        }
    }

 
    for (Move* m = moves; m != end; ++m)
    {
        StateInfo st;

        make_move(board, *m, st);

        profile.stop();

        int score = -alpha_beta(
            board,
            depth - 1,
            -beta,
            -alpha
        );

        profile.start();

        undo_move(board, *m, st);

        if (score > best_score)
        {
            best_score = score;
            best_move = *m;
        }

        alpha = std::max(alpha, score);

        if (alpha >= beta)
        {
            break;
        }
    }

    TTFlag flag;

    if (best_score <= original_alpha)
    {
        flag = UPPERBOUND;
    }
    else if (best_score >= beta)
    {
        flag = LOWERBOUND;
    }
    else
    {
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

    if (is_white)
    {
        attacker = piece_on(board->white, m.from_sq());
        victim   = piece_on(board->black, m.to_sq());
    }
    else
    {
        attacker = piece_on(board->black, m.from_sq());
        victim   = piece_on(board->white, m.to_sq());
    }

    // Game State EINMAL bestimmen

    int attacker_value = pesto_piece_value(
        attacker,
        m.from_sq(),
        is_white,
        gamePhase
    );

    int victim_value = pesto_piece_value(
        victim,
        m.to_sq(),
        !is_white,
        gamePhase
    );

    return victim_value * 1000 - attacker_value;
}


struct ScoredMove
{
    Move move;
    int score;
    int victim_value;
};

void sort_capture_moves(
    Move* moves,
    Move* end,
    int* victim_values,
    chess_board* board,
    int game_phase)
{
    const int count = static_cast<int>(end - moves);
    ScoredMove scored[256];

    const bool is_white = board->whites_turn;

    for (int i = 0; i < count; ++i)
    {
        Move move = moves[i];

        PieceType attacker;
        PieceType victim;

        if (is_white)
        {
            attacker = piece_on(board->white, move.from_sq());
            victim   = piece_on(board->black, move.to_sq());
        }
        else
        {
            attacker = piece_on(board->black, move.from_sq());
            victim   = piece_on(board->white, move.to_sq());
        }

        int attacker_value = pesto_piece_value(
            attacker, move.from_sq(), is_white, game_phase);

        int victim_value = pesto_piece_value(
            victim, move.to_sq(), !is_white, game_phase);

        scored[i] = {
            move,
            victim_value * 1000 - attacker_value,
            victim_value
        };
    }

    std::sort(scored, scored + count,
        [](const ScoredMove& a, const ScoredMove& b)
        {
            return a.score > b.score;
        });

    for (int i = 0; i < count; ++i)
    {
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

    int victim_value = pesto_piece_value(
        victim,
        move.to_sq(),
        !board->whites_turn,
        game_phase
    );

    if (eval + victim_value + DELTA_MARGIN <= alpha){
        return true;
    }
    return false;

}

int quiescence(chess_board* board, int alpha, int beta)
{


    Profiler::Scope profile("quiescence");
    ++Profiler::q_nodes;

    if (is_in_check(board))
    {
        Move moves[256];
        Move* end = find_all_moves(moves, board);

        if (moves == end)
            return -MATE_SCORE; // Schachmatt

        for (Move* m = moves; m != end; ++m)
        {
            
            StateInfo st;

            make_move(board, *m, st);
            profile.stop();
            int score = -quiescence(board, -beta, -alpha);
            profile.start();

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

    sort_capture_moves(
        moves,
        end,
        victim_values,
        board,
        game_phase
    );

    const int count = static_cast<int>(end - moves);

    for (int i = 0; i < count; ++i)
    {
        Move move = moves[i];

        // if (move.move_flag() != 1 && stand_pat + victim_values[i] + DELTA_MARGIN <= alpha)
        // {
        //     // ++Profiler::q_cutoffs;
        //     continue;
        // }

        StateInfo st;

        make_move(board, move, st);

        profile.stop();
        int score = -quiescence(board, -beta, -alpha);
        profile.start();

        undo_move(board, move, st);

        if (score >= beta)
            return beta;

        if (score > alpha)
            alpha = score;
    }

    return alpha;
}
