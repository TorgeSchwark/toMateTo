#include "toMateTo.h"


// should handle everything (find best move + time control)
std::map<std::string, uint64_t> toMateTo(std::string fen_position){

    std::map<std::string, uint64_t> result;

    chess_board board;
    setup_fen_position(board, fen_position);

    Move moves[256];

    float eval;
    
    Move* end = find_all_moves(moves, &board);

    for (Move* m = moves; m != end; ++m) {

        StateInfo st;

        make_move(&board, *m, st);

        eval = full_search_eval(&board, 5);

        undo_move(&board, *m, st);

        result[m->move_to_string(board.whites_turn)] = eval;

    }

    return result;

}

// evaluates a position, by searching in a certain depth 
// caller function must then get eval for every move
float full_search_eval(chess_board* chess_board, int depth){
    Move moves[256];
    Move* end = find_all_moves(moves, chess_board);

    float best_val = -99999.0;
    float current_val = -99999.0;

    if(depth == 1){
        return pesto_eval(chess_board, &chess_board->white, &chess_board->black);
    }
    for(Move* m = moves; m != end; ++m){
        StateInfo st;

        make_move(chess_board, *m, st);

        current_val = full_search_eval(chess_board, depth-1);
        
        undo_move(chess_board, *m, st);

        if(current_val > best_val){
            best_val = current_val;
        }
    }
    return best_val;
}