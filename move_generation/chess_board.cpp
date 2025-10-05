#include "chess_board.h"

// moves from one knight move to the next
// maybe split to capture and standard



int msb_index(int64_t bb) {
    return 63 - __builtin_clzll(bb);
}

void set_index_zero(int64_t* bitboard, int64_t index){
    *bitboard &= ~(1ULL << index);
}

void set_index_one(int64_t* bitboard, int64_t index) {
    *bitboard |= (1ULL << index);
}

void find_all_moves(move_stack* move_stack, chess_board* chess_board){

    if(chess_board->whites_turn){
        if(chess_board->white.knights){ // at least one knight alive
            find_knight_moves(move_stack, &(chess_board->white), &(chess_board->black));
        }
    }else{

    }
    
}

void find_bishop_moves(move_stack* move_stack, one_side* player, one_side* enemy){
    int64_t bishops = player->bishop;
    while(bishops){
        int bishop_index = __builtin_ctzll(bishops);        // Get index of least significant bit
        bishops &= bishops - 1;

    }

}

void find_knight_moves(move_stack* move_stack, one_side* player, one_side* enemy){
    int64_t knights = player->knights;
    while(knights){
        int knight_index = __builtin_ctzll(knights);        // Get index of least significant bit
        knights &= knights - 1; 
        int64_t current_knight_to;
        for(int i = 1; i < KNIGHT_LOOCKUP_TABLE[knight_index][0]; i++){
            current_knight_to = KNIGHT_LOOCKUP_TABLE[knight_index][i];
            if (!(player->side_all & current_knight_to)){
                int64_t from = 1LL << knight_index;
                move_stack->add_move(from, current_knight_to, KNIGHT, NORMAL_MOVE);
            }
        }
    }
}



