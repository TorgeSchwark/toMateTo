#include "chess_board.h"

int msb_index(Bitboard bb) {
    return 63 - __builtin_clzll(bb);
}

void set_index_zero(Bitboard* bitboard, Bitboard index){
    *bitboard &= ~(1ULL << index);
}

void set_index_one(Bitboard* bitboard, Bitboard index) {
    *bitboard |= (1ULL << index);
}

void find_all_moves(move_stack* move_stack, chess_board* chess_board){
    chess_board->pinned_pieces = 0LL;
    if(chess_board->whites_turn){
        find_knight_moves(move_stack, &(chess_board->white), &(chess_board->black));
        find_bishop_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.bishop));
        find_rook_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.rooks));

        find_bishop_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
        find_rook_moves(move_stack, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));

    }else{

    }
}

void find_bishop_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* bishop){
    Bitboard bishops = *bishop;
    while(bishops){

        int bishop_index = pop_lsb(bishops);

        Bitboard from = 1ULL << bishop_index;

        Bitboard attack_moves = magic(BISHOP, bishop_index, chess_board, player);

        while(attack_moves){
            int8_t bishop_index_to = pop_lsb(attack_moves);    

            Bitboard current_bishop_to = 1ULL << bishop_index_to;
            
            move_stack->add_move(from, current_bishop_to, BISHOP, NORMAL_MOVE);
        }
    }
}

void find_rook_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* rook){
    Bitboard rooks = *rook;
    while(rooks){
        int rook_index = pop_lsb(rooks);        

        Bitboard attack_moves = magic(ROOK, rook_index, chess_board, player);

        Bitboard from = 1ULL << rook_index;
    
        while(attack_moves){
            int rook_index_to = pop_lsb(attack_moves);   

            Bitboard current_rook_to = 1ULL << rook_index_to;

            move_stack->add_move(from, current_rook_to, ROOK, NORMAL_MOVE);
        }
    }
}


void find_knight_moves(move_stack* move_stack, one_side* player, one_side* enemy) {
    Bitboard knights = player->knights;

    while (knights) {
        int knight_index = pop_lsb(knights);

        Bitboard from = 1ULL << knight_index;          

        Bitboard attack_moves = KNIGHT_LOOKUP_TABLE[knight_index] & (~player->side_all);

        while (attack_moves) {
            int knight_index_to = pop_lsb(attack_moves); 

            Bitboard current_knight_to = 1ULL << knight_index_to;

            move_stack->add_move(from, current_knight_to, KNIGHT, NORMAL_MOVE);
        }
    }
}

bool not_attacked(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind){

    Bitboard straight_attackers = get_straight_attackers(enemy, pos_ind);
    // this will mark attackers and every blocking piece both enemy blocking and team
    Bitboard relevant_attackers_and_defenders = straight_attackers&(chess_board->complete_board);
    if (get_diagonal_attacks(enemy, player, pos_ind, relevant_attackers_and_defenders)){
        return false;
    }
    // return directions where an attacker exists until attacker including attacker!
    Bitboard diagonal_attackers = get_diagonal_attackers(enemy, pos_ind);
    // this will mark attackers and every blocking piece both enemy blocking and team
    relevant_attackers_and_defenders = diagonal_attackers&(chess_board->complete_board);
    if(Bitboard attacking_pieces = get_diagonal_attacks(enemy, player, pos_ind, relevant_attackers_and_defenders)){
        return false;
    }

    Bitboard knight_moves = KNIGHT_LOOKUP_TABLE[pos_ind];

    return true;
}

// get the blocker hash
// this can be used if all attackers and blockers in attack direction are marked it will only return the attackers if there attack is not blocked!
Bitboard get_diagonal_attacks(one_side* enemy, one_side* player, int pos_ind, Bitboard relevant_pieces){
    int magic_index_blocked = (relevant_pieces *PINNED_PIECES_BISHOP_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_BISHOP_MAGIC[pos_ind].relevant_bits);
    Bitboard attacking_pieces= PINNED_PIECES_BISHOP_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return attacking_pieces;
}

Bitboard get_straight_attacks(one_side* enemy, one_side* player, int pos_ind, Bitboard relevant_pieces){
    int magic_index_blocked = (relevant_pieces *PINNED_PIECES_ROOK_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_ROOK_MAGIC[pos_ind].relevant_bits);
    Bitboard pinned_pieces = PINNED_PIECES_ROOK_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return pinned_pieces;
}

Bitboard get_diagonal_attackers(one_side* enemy, int pos_ind){
    Bitboard king_diagonal_attackers = (enemy->bishop | enemy->queen) & BISHOP_MAGIC[pos_ind].mask;
    int magic_index = ((king_diagonal_attackers * BISHOP_MAGIC[pos_ind].magic_number) >> (64 - BISHOP_MAGIC[pos_ind].relevant_bits));
    Bitboard attack_mask = ATTACK_PATTERN_BISHOP_MAGIC[pos_ind].attack_list[magic_index];
    return attack_mask;
}

Bitboard get_straight_attackers(one_side* enemy, int pos_ind){
    Bitboard king_straight_atackers = (enemy->rooks | enemy->queen) & ROOK_MAGIC[pos_ind].mask;
    int magic_index = ((king_straight_atackers * ROOK_MAGIC[pos_ind].magic_number) >> (64 - ROOK_MAGIC[pos_ind].relevant_bits));
    Bitboard attack_mask = ATTACK_PATTERN_ROOK_MAGIC[pos_ind].attack_list[magic_index];
    return attack_mask;
}

Bitboard get_straight_pins(one_side* enemy, one_side* player, int pos_ind, Bitboard attack_mask){
    Bitboard blocked_pieces = attack_mask&(enemy->knights|enemy->bishop|enemy->pawns|player->side_all);
    int magic_index_blocked = (blocked_pieces *PINNED_PIECES_ROOK_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_ROOK_MAGIC[pos_ind].relevant_bits);
    Bitboard pinned_pieces = PINNED_PIECES_ROOK_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return pinned_pieces;
}

Bitboard get_diagonal_pins(one_side* enemy, one_side* player, int pos_ind, Bitboard attack_mask){
    Bitboard blocked_pieces = attack_mask&(enemy->knights|enemy->rooks|enemy->pawns|player->side_all);
    int magic_index_blocked = (blocked_pieces *PINNED_PIECES_BISHOP_MAGIC[pos_ind].magic_number) >> (64 - PINNED_PIECES_BISHOP_MAGIC[pos_ind].relevant_bits);
    Bitboard pinned_pieces = PINNED_PIECES_BISHOP_MAGIC[pos_ind].attack_list[magic_index_blocked];
    return pinned_pieces;
}


void find_all_king_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    int king_pos = __builtin_ctzll(player->king);

    Bitboard attack_mask = get_diagonal_attackers(enemy, king_pos);
    // pinned pieces 
    Bitboard pinned_pieces = get_diagonal_pins(enemy, player, king_pos, attack_mask);
    
    chess_board->pinned_pieces = chess_board->pinned_pieces | pinned_pieces;

    // find all straight attacks to king
    attack_mask = get_straight_attackers(enemy, king_pos);

    // pinned pieces 
    pinned_pieces = get_straight_pins(enemy, player, king_pos, attack_mask);
    
    chess_board->pinned_pieces = chess_board->pinned_pieces | pinned_pieces;
    
    Bitboard king_moves = KING_MOVES_MASK[king_pos];
    while(king_moves){
        int king_index_to = __builtin_ctzll(king_moves); 
        king_moves &= king_moves - 1;                      // delete LSB
        Bitboard king_to = 1ULL << king_index_to;
        Bitboard from = 1ULL << king_pos;

        if (not_attacked(chess_board, player, enemy, king_pos)){
            move_stack->add_move(from, king_to, KING, NORMAL_MOVE);
        }
    }
}




