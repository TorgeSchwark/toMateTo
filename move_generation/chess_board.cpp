#include "chess_board.h"


void find_all_moves(move_stack* move_stack, chess_board* chess_board){

    if(chess_board.whites_turn){
        if(chess_board->white_knights){ // at least one knight alive
            find_knight_moves(move_stack, chess_board)
        }
    }else{

    }
    
}

void find_knight_moves(move_stack* move_stack, chess_board* chess_board){
    int64_t nights = chess_board.white_knights;
    

}

#include <cstdint>

int msb_index(uint64_t bb) {
    return 63 - __builtin_clzll(bb);  // Returns position of MSB (0-based)
}
