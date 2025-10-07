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
    chess_board->pinned_pieces == 0LL;
    if(chess_board->whites_turn){
        if(chess_board->white.knights){ // at least one knight alive
            find_knight_moves(move_stack, &(chess_board->white), &(chess_board->black));
        }
        if(chess_board->white.knights){
            find_bishop_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black));
        }
        if(chess_board->white.rooks){
            find_rook_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black));
        }
        if(chess_board->white.queen){
            find_queen_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black));
        }
    }else{

    }
}

// Vorausgesetzt: 
// BISHOP_ATTACKS[index][blocker_occupation] gibt Bitboard mit allen erreichbaren Zielen zurück

void find_bishop_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int64_t bishops = player->bishop;
    while(bishops){
        int bishop_index = __builtin_ctzll(bishops);        // Get index of least significant bit
        bishops &= bishops - 1;
        int64_t current_bishop_to;
        int64_t occ = chess_board->complete_board & BISHOP_MAGIC[bishop_index].mask;
        int index = (int)((occ * BISHOP_MAGIC[bishop_index].magic_number) >> (64 - BISHOP_MAGIC[bishop_index].relevant_bits));
        int64_t atack_moves = BISHOP_MAGIC[bishop_index].attack_list[index] & ~player->side_all;
        while(atack_moves){
            int bishop_index_to = __builtin_ctzll(atack_moves); // Index des Ziel-Feldes
            atack_moves &= atack_moves - 1;                      // LSb entfernen
            current_bishop_to = 1ULL << bishop_index_to;
            int64_t from = 1ULL << bishop_index;
            move_stack->add_move(from, current_bishop_to, BISHOP, NORMAL_MOVE);
        }
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


void find_rook_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int64_t rooks = player->rooks;
    while(rooks){
        int rook_index = __builtin_ctzll(rooks);        // Get index of least significant bit
        rooks &= rooks - 1;
        int64_t current_rook_to;
        int64_t occ = chess_board->complete_board & ROOOK_MAGIC[rook_index].mask;
        int index = (int)((occ * ROOOK_MAGIC[rook_index].magic_number) >> (64 - ROOOK_MAGIC[rook_index].relevant_bits));
        int64_t atack_moves = ROOOK_MAGIC[rook_index].attack_list[index] & ~player->side_all;
        while(atack_moves){
            int rook_index_to = __builtin_ctzll(atack_moves); // Index des Ziel-Feldes
            atack_moves &= atack_moves - 1;                      // LSb entfernen
            current_rook_to = 1ULL << rook_index_to;
            int64_t from = 1ULL << rook_index;
            move_stack->add_move(from, current_rook_to, ROOK, NORMAL_MOVE);
        }
    }
}

void find_queen_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int64_t queens = player->queen;
    while(queens){
        int quenn_index = __builtin_ctzll(queens);        // Get index of least significant bit
        queens &= queens - 1;
        int64_t current_queen_to;
        int64_t from;
        int queen_index_to;

        int64_t occ = chess_board->complete_board & ROOOK_MAGIC[quenn_index].mask;
        int index = (int)((occ * ROOOK_MAGIC[quenn_index].magic_number) >> (64 - ROOOK_MAGIC[quenn_index].relevant_bits));
        int64_t atack_moves = ROOOK_MAGIC[quenn_index].attack_list[index];

        occ = chess_board->complete_board & BISHOP_MAGIC[quenn_index].mask;
        index = (int)((occ * BISHOP_MAGIC[quenn_index].magic_number) >> (64 - BISHOP_MAGIC[quenn_index].relevant_bits));
        atack_moves |= BISHOP_MAGIC[quenn_index].attack_list[index];
        atack_moves &= ~player->side_all;

        while(atack_moves){
            queen_index_to = __builtin_ctzll(atack_moves); // Index des Ziel-Feldes
            atack_moves &= atack_moves - 1;                      // LSb entfernen
            current_queen_to = 1ULL << queen_index_to;
            from = 1ULL << quenn_index;
            move_stack->add_move(from, current_queen_to, QUEEN, NORMAL_MOVE);
        }
    }
}


void find_all_king_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int king_index = __builtin_ctzll(player->king);        // Get index of least significant bit
    int64_t king_mask = KING_MOVES_MASK[king_index] & player->side_all;
    
    int64_t king_diagonals_occ =  enemy->side_all & BISHOP_MAGIC[king_index].mask; // all enemy pieces
    int index = (int)((king_diagonals_occ * ROOOK_MAGIC[king_index].magic_number) >> (64 - ROOOK_MAGIC[king_index].relevant_bits));
    int64_t diagonals_until_enemy = ROOOK_MAGIC[king_index].attack_list[index];

}


