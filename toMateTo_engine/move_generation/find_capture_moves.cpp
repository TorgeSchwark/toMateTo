#include "find_capture_moves.h"


Move* find_all_capture_moves(Move* moves, chess_board* chess_board){

    if(chess_board->whites_turn){
        square white_king_square = __builtin_ctzll(chess_board->white.king);

        find_pin_information(chess_board,  &(chess_board->white), &(chess_board->black), white_king_square);

        moves = find_king_save_squares_captures(moves, chess_board,  &(chess_board->white), &(chess_board->black), white_king_square);
        // moves = add_castling(moves, chess_board, &(chess_board->white), &(chess_board->black), white_king_square, chess_board->whites_turn);

        if(chess_board->attack_count < 2){
            if(chess_board->attack_count == 1){
                chess_board->attack_defend_squares = SQUARES_IN_BETWEEN[white_king_square][__builtin_ctzll(chess_board->attacking_pieces)];
            }

            moves = find_knight_capture_moves(moves, chess_board, &(chess_board->white), &(chess_board->black));

            moves = find_pawn_capture_moves(moves, chess_board, &(chess_board->white), &(chess_board->black));

            moves = find_bishop_capture_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.bishop));
            moves = find_rook_capture_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.rooks));

            moves = find_bishop_capture_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
            moves = find_rook_capture_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
        }
    }else{
        square black_king_square = __builtin_ctzll(chess_board->black.king);

        find_pin_information(chess_board, &(chess_board->black), &(chess_board->white), black_king_square);

        moves = find_king_save_squares_captures(moves, chess_board,  &(chess_board->black), &(chess_board->white), black_king_square);
        // moves = add_castling(moves, chess_board,  &(chess_board->black), &(chess_board->white), black_king_square, chess_board->whites_turn);

        if(chess_board->attack_count < 2){
            if(chess_board->attack_count == 1){
                chess_board->attack_defend_squares = SQUARES_IN_BETWEEN[black_king_square][__builtin_ctzll(chess_board->attacking_pieces)];
            }

            moves = find_knight_capture_moves(moves, chess_board, &(chess_board->black), &(chess_board->white));

            moves = find_pawn_capture_moves(moves, chess_board, &(chess_board->black), &(chess_board->white));

            moves = find_bishop_capture_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.bishop));
            moves = find_rook_capture_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.rooks));

            moves = find_bishop_capture_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.queen));
            moves = find_rook_capture_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.queen));
        }
    }
    return moves;
}

Move* find_king_save_squares_captures(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, square king_position){
    Bitboard possible_king_moves = (KING_MOVES_MASK[king_position] & ~player->side_all) & enemy->side_all;
    player->save_king_squares = 0LL;
    while(possible_king_moves){
        square to = pop_lsb(possible_king_moves);
        if(is_save_square(chess_board, player, enemy, to, player->king)){ // there can be a piece as long as the square is not under attack
            // this whole function could be split in only parallel moves and the rest so this is not done for every free square:
            player->save_king_squares |= (1ULL << to);
            *moves++ = Move(king_position, to);
        }
    }return moves;
}


Move* find_bishop_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* bishop){
    Bitboard bishops = *bishop;
    while(bishops){

        square bishop_square = pop_lsb(bishops);
        bool is_pinned = (1LL<<bishop_square)&chess_board->pinned_pieces;
        if(is_pinned && (chess_board->attack_count)){
            continue;
        }    

        Bitboard bishop_destinations_captures = bishop_magic_captures(bishop_square, chess_board, enemy);

        // a pinned piece can only walk if the king is not in check
        if(chess_board->attack_count){
            // if we get here and king is attacked it means piece in not pinned!
            // Filter for defending moves4
            bishop_destinations_captures &= chess_board->attack_defend_squares;
        }else if(is_pinned){
            // if we get here and piece is pinned it means king is not attacked!
            // only walk on pinned line!
            bishop_destinations_captures &= SQUARES_ON_THE_LINE[bishop_square][__builtin_ctzll(player->king)];
        }// else king is not attacked and piece not pinned!

        moves = add_normal_moves(bishop_square, bishop_destinations_captures, moves);
    }
    return moves;
}

Move* find_knight_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy) {
    Bitboard knights = player->knights & (~chess_board->pinned_pieces); // pinned nights cant walk
    while (knights) {

        square knight_square = pop_lsb(knights);
         
        Bitboard knight_destinations_captures = KNIGHT_LOOKUP_TABLE[knight_square] & enemy->side_all;

        if(chess_board->attack_count){
            // if we get here and king is attacked it means piece in not pinned!
            // Filter for defending moves
            knight_destinations_captures &= chess_board->attack_defend_squares;
        }
        moves = add_normal_moves(knight_square, knight_destinations_captures, moves);
    }
    return moves;
}

Move* find_pawn_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy){
    Bitboard empty_squares = ~chess_board->complete_board;
    Bitboard pawns = player->pawns;

    Bitboard pinned_pawns = pawns & chess_board->pinned_pieces;
    pawns &= ~pinned_pawns;

    // if the ep pawn attacks the king we can capture it even though we dont move to its square
    Bitboard eq_pawns_pos = (1ULL << (chess_board->ep_square + color_dir(FORWARD, !chess_board->whites_turn)));
    Bitboard attack_defend_squares_pawns = chess_board->attack_defend_squares;
    if(chess_board->attack_defend_squares & eq_pawns_pos){
        attack_defend_squares_pawns |= 1ULL << (chess_board->ep_square);
    }
    
    Bitboard results[RESULT_COUNT] = {0LL};
    Bitboard results_pinned[RESULT_COUNT] = {0LL};
    find_different_pawn_capture_moves(pawns, empty_squares, player, enemy, chess_board, results);
    if(!chess_board->attack_count){
        if(pinned_pawns){
            while(pinned_pawns){
                square pinned_pawn_square = pop_lsb(pinned_pawns);
                Bitboard pinned_pawn = 1LL << pinned_pawn_square;
                Bitboard allowed_squares = SQUARES_ON_THE_LINE[pinned_pawn_square][__builtin_ctzll(player->king)];

                find_different_pawn_capture_moves(pinned_pawn, empty_squares, player, enemy, chess_board, results_pinned);
                
                for (int i = 0; i < RESULT_COUNT; ++i){
                    results[i] |= (allowed_squares & results_pinned[i]);
                }
            }            
        }
    }else{
        for (int i = 0; i < RESULT_COUNT; ++i){
            results[i] &= attack_defend_squares_pawns;
        }
    }
    
    moves = add_all_pawn_moves(results, moves, chess_board->whites_turn);

    return moves;

}

void find_different_pawn_capture_moves(Bitboard pawns, Bitboard empty, one_side* player, one_side* enemy, chess_board* chess_board, Bitboard* results){
    // single push
    bool is_white = chess_board->whites_turn;

    // captures
    Bitboard pawnCapL = BB_SHIFT_FORWARD_LEFT(pawns, is_white) & (~BOARD_FILE[FILE_H]);
    Bitboard pawnCapR = BB_SHIFT_FORWARD_RIGHT(pawns, is_white) & (~BOARD_FILE[FILE_A]);

    // normal captures
    results[CAPL] = pawnCapL & enemy->side_all;
    results[CAPR]= pawnCapR & enemy->side_all;

    // promotions
    results[PROMO_CAPL] = results[CAPL] & PROMOTION_ROW[is_white];
    results[CAPL] &= ~PROMOTION_ROW[is_white];
    results[PROMO_CAPR]  = results[CAPR] & PROMOTION_ROW[is_white];
    results[CAPR] &= ~PROMOTION_ROW[is_white];

    // This cant happen!?
    // if(!((1LL << (chess_board->ep_square-color_dir(FORWARD, is_white))) & chess_board->pinned_pieces)){
    if (chess_board->ep_square != SQ_NONE){
        square king_pos = __builtin_ctzll(player->king);
        // en passant
        Bitboard attackers_straight = attackers_straight_ray_pawns(chess_board, player, enemy, __builtin_ctzll(player->king), player->king);
        Bitboard attackers_on_king_line = attackers_straight & ROWS[king_pos >> 3];
        bool en_passant_capture_is_pinned = false;
        Bitboard eq_pawns_pos = (1ULL << (chess_board->ep_square + color_dir(FORWARD, !is_white)));
        while(attackers_on_king_line){
            square attack_piece = pop_lsb(attackers_on_king_line);
            Bitboard squares_in_between_attack = SQUARES_IN_BETWEEN[attack_piece][king_pos];

            if(eq_pawns_pos & squares_in_between_attack && __builtin_popcountll(squares_in_between_attack & (player->pawns | enemy->pawns)) == 2){
                en_passant_capture_is_pinned = true;
                break;
            }
        }
        if(!en_passant_capture_is_pinned){
            results[EPL] = pawnCapL & (1ULL << chess_board->ep_square);
            results[EPR] = pawnCapR & (1ULL << chess_board->ep_square);
        }else{
            results[EPL] = 0LL;
            results[EPR] = 0LL;
        }
    }else{
        results[EPL] = 0LL;
        results[EPR] = 0LL;
    }

}

Move* find_rook_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* rook){
    Bitboard rooks = *rook;
    while(rooks){
        square rook_square = pop_lsb(rooks);    
        bool is_pinned = (1LL<<rook_square)&chess_board->pinned_pieces;
        // piece cant move to save king if pinned.
        if(is_pinned && (chess_board->attack_count)){
            continue;
        }    

        Bitboard rook_destinations_captures = rook_magic_captures(rook_square, chess_board, enemy);

        // a pinned piece can only walk if the king is not in check
        if(chess_board->attack_count){
            // if we get here and king is attacked it means piece in not pinned!
            // Filter for defending moves
            rook_destinations_captures &= chess_board->attack_defend_squares;
        }else if(is_pinned){
            // if we get here and piece is pinned it means king is not attacked!
            // only walk on pinned line!
            rook_destinations_captures &= SQUARES_ON_THE_LINE[rook_square][__builtin_ctzll(player->king)];
        }// else king is not attacked and piece not pinned!

        moves = add_normal_moves(rook_square, rook_destinations_captures, moves);
    }
    return moves;
}