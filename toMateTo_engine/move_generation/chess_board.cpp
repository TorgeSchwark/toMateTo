#include "chess_board.h"

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


void find_knight_moves(move_stack* move_stack, one_side* player, one_side* enemy) {
    int64_t knights = player->knights;

    while (knights) {
        int knight_index = __builtin_ctzll(knights); 
        knights &= knights - 1;                      

        int64_t attack_moves = KNIGHT_LOOKUP_TABLE[knight_index];

        attack_moves &= ~player->side_all;

        while (attack_moves) {
            int knight_index_to = __builtin_ctzll(attack_moves); 
            attack_moves &= attack_moves - 1;                    

            int64_t current_knight_to = 1ULL << knight_index_to;
            int64_t from = 1ULL << knight_index;

            move_stack->add_move(from, current_knight_to, KNIGHT, NORMAL_MOVE);
        }
    }
}



void find_rook_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int64_t rooks = player->rooks;
    while(rooks){
        int rook_index = __builtin_ctzll(rooks);        // Get index of least significant bit
        rooks &= rooks - 1;
        int64_t current_rook_to;
        int64_t occ = chess_board->complete_board & ROOK_MAGIC[rook_index].mask;
        int index = (int)((occ * ROOK_MAGIC[rook_index].magic_number) >> (64 - ROOK_MAGIC[rook_index].relevant_bits));
        int64_t atack_moves = ROOK_MAGIC[rook_index].attack_list[index] & ~player->side_all;
        int64_t from = 1ULL << rook_index;
        //  if ( chess_board->pinned_pieces & from){
        //    if(!one_direction[rock_from]&player->king){
        //          atack_moves &= (~one_direction) delete the direction
        //     }
        // }
        while(atack_moves){
            int rook_index_to = __builtin_ctzll(atack_moves); // Index des Ziel-Feldes
            atack_moves &= atack_moves - 1;                      // LSb entfernen
            current_rook_to = 1ULL << rook_index_to;
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

        int64_t occ = chess_board->complete_board & ROOK_MAGIC[quenn_index].mask;
        int index = (int)((occ * ROOK_MAGIC[quenn_index].magic_number) >> (64 - ROOK_MAGIC[quenn_index].relevant_bits));
        int64_t atack_moves = ROOK_MAGIC[quenn_index].attack_list[index];

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


bool not_attacked(chess_board* chess_board, one_side* player, one_side* enemy, int64_t pos_ind){

    int64_t straight_attackers = get_straight_attackers(enemy, pos_ind);
    // this will mark attackers and every blocking piece both enemy blocking and team
    int64_t relevant_attackers_and_defenders = straight_attackers&(chess_board->complete_board);
    if (get_diagonal_attacks(enemy, player, pos_ind, relevant_attackers_and_defenders)){
        return false;
    }
    // return directions where an attacker exists until attacker including attacker!
    int64_t diagonal_attackers = get_diagonal_attackers(enemy, pos_ind);
    // this will mark attackers and every blocking piece both enemy blocking and team
    relevant_attackers_and_defenders = diagonal_attackers&(chess_board->complete_board);
    if(int64_t attacking_pieces = get_diagonal_attacks(enemy, player, pos_ind, relevant_attackers_and_defenders)){
        return false;
    }

    int64_t knight_moves = KNIGHT_LOOKUP_TABLE[pos_ind];

    return true;
}

// get the blocker hash
// this can be used if all attackers and blockers in attack direction are marked it will only return the attackers if there attack is not blocked!
int64_t get_diagonal_attacks(one_side* enemy, one_side* player, int pos_ind, int64_t relevant_pieces){
    int magic_index_blocked = (relevant_pieces *PINNED_PIECES_BISHOP_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_BISHOP_MAGIC[pos_ind].relevant_bits);
    int64_t attacking_pieces= PINNED_PIECES_BISHOP_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return attacking_pieces;
}

int64_t get_straight_attacks(one_side* enemy, one_side* player, int pos_ind, int64_t relevant_pieces){
    int magic_index_blocked = (relevant_pieces *PINNED_PIECES_ROOK_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_ROOK_MAGIC[pos_ind].relevant_bits);
    int64_t pinned_pieces = PINNED_PIECES_ROOK_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return pinned_pieces;
}

int64_t get_diagonal_attackers(one_side* enemy, int pos_ind){
    int64_t king_diagonal_attackers = (enemy->bishop | enemy->queen) & BISHOP_MAGIC[pos_ind].mask;
    int magic_index = ((king_diagonal_attackers * BISHOP_MAGIC[pos_ind].magic_number) >> (64 - BISHOP_MAGIC[pos_ind].relevant_bits));
    int64_t attack_mask = ATTACK_PATTERN_BISHOP_MAGIC[pos_ind].attack_list[magic_index];
    return attack_mask;
}

int64_t get_straight_attackers(one_side* enemy, int pos_ind){
    int64_t king_straight_atackers = (enemy->rooks | enemy->queen) & ROOK_MAGIC[pos_ind].mask;
    int magic_index = ((king_straight_atackers * ROOK_MAGIC[pos_ind].magic_number) >> (64 - ROOK_MAGIC[pos_ind].relevant_bits));
    int64_t attack_mask = ATTACK_PATTERN_ROOK_MAGIC[pos_ind].attack_list[magic_index];
    return attack_mask;
}

int64_t get_straight_pins(one_side* enemy, one_side* player, int pos_ind, int64_t attack_mask){
    int64_t blocked_pieces = attack_mask&(enemy->knights|enemy->bishop|enemy->pawns|player->side_all);
    int magic_index_blocked = (blocked_pieces *PINNED_PIECES_ROOK_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_ROOK_MAGIC[pos_ind].relevant_bits);
    int64_t pinned_pieces = PINNED_PIECES_ROOK_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return pinned_pieces;
}

int64_t get_diagonal_pins(one_side* enemy, one_side* player, int pos_ind, int64_t attack_mask){
    int64_t blocked_pieces = attack_mask&(enemy->knights|enemy->rooks|enemy->pawns|player->side_all);
    int magic_index_blocked = (blocked_pieces *PINNED_PIECES_BISHOP_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_BISHOP_MAGIC[pos_ind].relevant_bits);
    int64_t pinned_pieces = PINNED_PIECES_BISHOP_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return pinned_pieces;
}


void find_all_king_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int king_pos = __builtin_ctzll(player->king);

    int64_t attack_mask = get_diagonal_attackers(enemy, king_pos);
    // pinned pieces 
    int64_t pinned_pieces = get_diagonal_pins(enemy, player, king_pos, attack_mask);
    
    chess_board->pinned_pieces = chess_board->pinned_pieces | pinned_pieces;

    // find all straight attacks to king
    attack_mask = get_straight_attackers(enemy, king_pos);

    // pinned pieces 
    pinned_pieces = get_straight_pins(enemy, player, king_pos, attack_mask);
    
    chess_board->pinned_pieces = chess_board->pinned_pieces | pinned_pieces;
    
    int64_t king_moves = KING_MOVES_MASK[king_pos];
    while(king_moves){
        int king_index_to = __builtin_ctzll(king_moves); 
        king_moves &= king_moves - 1;                      // delete LSB
        int64_t king_to = 1ULL << king_index_to;
        int64_t from = 1ULL << king_pos;

        if (not_attacked(chess_board, player, enemy, king_pos)){
            move_stack->add_move(from, king_to, KING, NORMAL_MOVE);
        }
    }
}




