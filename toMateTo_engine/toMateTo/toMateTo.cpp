#include "toMateTo.h"


// should handle everything (find best move + time control)
std::string toMateTo(std::string fen_position){

    chess_board board;
    setup_fen_position(board, fen_position);

    

}

// evaluates a position, by searching in a certain depth 
// caller function must then get eval for every move
float full_search_eval(chess_board* chess_board, int depth){
    Move moves[256];
    Move* end = find_all_moves(moves, chess_board);

    float best_val = -99999.0;
    float current_val = -99999.0;

    if(depth == 1){
        return 0.0;
    }
    for(Move* m; m < end; m++){
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