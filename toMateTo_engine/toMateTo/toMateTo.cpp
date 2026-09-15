#include "toMateTo.h"

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

int alpha_beta(chess_board* board, int depth, int alpha, int beta)
{
    if (depth == 0)
    {
        return quiescence(board, -beta, -alpha);
    }

    Move moves[256];
    Move* end = find_all_moves(moves, board);

    int best_score = -99999;

    for (Move* m = moves; m != end; ++m)
    {
        StateInfo st;

        make_move(board, *m, st);

        int score = -alpha_beta(
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


void sort_capture_moves(Move* moves, Move* end, chess_board* board)
{   
    int gamePhase = pesto_game_phase(board);
    for (Move* m = moves; m != end; ++m)
    {
        Move* best = m;
        int best_score = mvv_lva_score(board, *m);

        for (Move* n = m + 1; n != end; ++n)
        {
            int score = mvv_lva_score(board, *n);

            if (score > best_score)
            {
                best = n;
                best_score = score;
            }
        }

        if (best != m)
        {
            std::swap(*m, *best);
        }
    }
}

int quiescence(chess_board* board, int alpha, int beta)
{
     int stand_pat = pesto_eval(board, &board->white, &board->black);

    // if (stand_pat >= beta)
    //     return beta;

    if (stand_pat > alpha)
        alpha = stand_pat;

    Move moves[256];
    Move* end = find_all_capture_moves(moves, board);

    sort_capture_moves(moves, end, board);

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

