#include "toMateTo.h"
#include "profiler.h"

int MATE_SCORE = 99999;


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
        std::cout << move_string << " -> "
          << pesto_eval(&board, &board.white, &board.black)
          << "\n";
        
        int eval = -alpha_beta(&board, depth - 1, -beta, -alpha);

        undo_move(&board, *m, st);

        result[move_string] = eval;
        // alpha is the current best move under consid of opp
        alpha = std::max(alpha, eval);
    }

    return result;
}

std::string alpha_beta_tt_toMateTo(
    std::string fen_position,
    int depth)
{
    chess_board board;
    setup_fen_position(board, fen_position);

    for (int itt_depth = 1; itt_depth <= depth; ++itt_depth)
    {
        std::cout
            << "Searching depth "
            << itt_depth
            << "...\n";

        alpha_beta(
            &board,
            itt_depth,
            -99999,
            99999
        );
    }

    uint64_t hash = calculate_hash(&board);

    TTEntry& entry =
        transposition_table[hash & (TT_SIZE - 1)];

    if (entry.key == hash)
    {
        return entry.best_move.move_to_string(
            board.whites_turn
        );
    }

    return "";
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
};

void sort_capture_moves(Move* moves, Move* end, chess_board* board)
{
    const int count = static_cast<int>(end - moves);

    ScoredMove scored[256];

    for (int i = 0; i < count; ++i)
    {
        scored[i].move = moves[i];
        scored[i].score = mvv_lva_score(board, moves[i]);
    }

    std::sort(scored, scored + count,
        [](const ScoredMove& a, const ScoredMove& b)
        {
            return a.score > b.score;
        });

    for (int i = 0; i < count; ++i)
        moves[i] = scored[i].move;
}



int quiescence(chess_board* board, int alpha, int beta)
{
    static thread_local int q_depth = 0;
    ++q_depth;

    struct QDepthGuard
    {
        int& depth;
        ~QDepthGuard()
        {
            --depth;
        }
    } guard{q_depth};

    if (q_depth > 100)
    {
        Move moves[256];
        Move* end = find_all_moves(moves, board);
        std::cerr << q_depth ;
        std::cerr << board_to_fen(*board);
        std::cerr << moves[0].move_to_string(board->whites_turn) << "\n";
        std::cerr << "QSEARCH DEPTH LIMIT!\n";
    }
    if (q_depth > 120){
        std::abort();
    }


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

            int score = -quiescence(board, -beta, -alpha);

            undo_move(board, *m, st);

            if (score >= beta)
                return beta;

            if (score > alpha)
                alpha = score;
        }

        return alpha;
    }

    int stand_pat = pesto_eval(board, &board->white, &board->black);

    if (stand_pat >= beta)
        return beta;

    if (stand_pat > alpha)
        alpha = stand_pat;

    // nur Captures
    Move moves[256];
    Move* end = find_all_capture_moves(moves, board);

    sort_capture_moves(moves, end, board);

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
