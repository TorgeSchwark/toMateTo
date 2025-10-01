#include "chess_board.h"

inline int msb_index(int64_t bb) {
    return 63 - __builtin_clzll(bb);
}

inline void set_index_zero(int64_t* bitboard, int64_t index){
    *bitboard &= ~(1ULL << index);
}



void find_all_moves(move_stack* move_stack, chess_board* chess_board){

    if(chess_board->whites_turn){
        if(chess_board->white.knights){ // at least one knight alive
            find_knight_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black));
        }
    }else{

    }
    
}

void find_knight_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int64_t nights = chess_board->white.knights;
}



