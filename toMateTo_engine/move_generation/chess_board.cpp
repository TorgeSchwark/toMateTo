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

Move* find_all_moves(Move* moves, chess_board* chess_board){
    chess_board->pinned_pieces = 0LL;
    if(chess_board->whites_turn){
        moves = find_knight_moves(moves, &(chess_board->white), &(chess_board->black));
        moves = find_bishop_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.bishop));
        moves = find_rook_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.rooks));

        moves = find_bishop_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
        moves = find_rook_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));

    }else{

    }
    return moves;
}

Move* add_normal_moves(square from, Bitboard destinations, Move* moves){
    while(destinations){
        *moves++ = Move(from, pop_lsb(destinations));
    }return moves;
}

Move* find_bishop_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* bishop){
    Bitboard bishops = *bishop;
    while(bishops){

        square bishop_square = pop_lsb(bishops);

        Bitboard bishop_destinations = bishop_magic(bishop_square, chess_board, player);

        moves = add_normal_moves(bishop_square, bishop_destinations, moves);
    }
    return moves;
}



Move* find_rook_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* rook){
    Bitboard rooks = *rook;
    while(rooks){
        square rook_square = pop_lsb(rooks);        

        Bitboard rook_destinations = rook_magic(rook_square, chess_board, player);

        moves = add_normal_moves(rook_square, rook_destinations, moves);
    }
    return moves;
}

Move* find_knight_moves(Move* moves, one_side* player, one_side* enemy) {
    Bitboard knights = player->knights;
    Bitboard negative_player = ~player->side_all;
    while (knights) {
        square knight_square= pop_lsb(knights);

        Bitboard knight_destinations = KNIGHT_LOOKUP_TABLE[knight_square] & negative_player;

        moves = add_normal_moves(knight_square, knight_destinations, moves);
    }
    return moves;
}

bool is_save_square(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind){
    // Checks if a pos is attacked by a piece

    // returns a Bitboard where every square between nearest attacker and the square is marked! 
    Bitboard relevant_squares = get_straight_attackers(enemy, pos_ind);
    // with this we get only the attacking piece itself and every piece in the line of attack
    if (is_straight_attacked(chess_board, pos_ind, relevant_squares)){
        return false;
    }
    // return directions where an attacker exists until attacker including attacker!
    relevant_squares = get_diagonal_attackers(enemy, pos_ind);
    // this will mark attackers and every blocking piece both enemy blocking and team
    if(is_diagonal_attacked(chess_board, pos_ind, relevant_squares)){
        return false;
    }

    if(KNIGHT_LOOKUP_TABLE[pos_ind] & enemy->knights){
        return false;
    }

    if(PAWN_ATTACK_LOOKUP_TABLE[chess_board->whites_turn][pos_ind]&enemy->pawns){
        return false;
    }

    return true;
}

void find_all_king_moves(move_stack* move_stack, chess_board* chess_board, one_side* player, one_side* enemy){
    // check first if King is in check! 


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

        if (is_save_square(chess_board, player, enemy, king_pos)){
            move_stack->add_move(from, king_to, KING, NORMAL_MOVE);
        }
    }
}

void setup_fen_position(chess_board& board, const std::string& fen)
{
    // Clear board
    board.white = {};
    board.black = {};
    board.complete_board = 0ULL;
    board.castling_rights = NO_CASTLING;
    board.ep_square = -1;
    board.halve_move_counter = 0;
    board.full_move_counter = 1;

    std::istringstream ss(fen);

    std::string placement, active, castling, enpassant;
    int halfmove = 0, fullmove = 1;

    ss >> placement >> active >> castling >> enpassant >> halfmove >> fullmove;

    // --- Piece placement ---
    int square = 56;   // a8

    for (char c : placement)
    {
        if (c == '/')
        {
            square -= 16;   // next rank down
            continue;
        }

        if (std::isdigit(c))
        {
            square += (c - '0');
            continue;
        }

        Bitboard bit = 1ULL << square;

        switch (c)
        {
            case 'P': board.white.pawns   |= bit; break;
            case 'N': board.white.knights |= bit; break;
            case 'B': board.white.bishop  |= bit; break;
            case 'R': board.white.rooks   |= bit; break;
            case 'Q': board.white.queen   |= bit; break;
            case 'K': board.white.king    |= bit; break;

            case 'p': board.black.pawns   |= bit; break;
            case 'n': board.black.knights |= bit; break;
            case 'b': board.black.bishop  |= bit; break;
            case 'r': board.black.rooks   |= bit; break;
            case 'q': board.black.queen   |= bit; break;
            case 'k': board.black.king    |= bit; break;
        }

        square++;
    }

    // --- Active color ---
    board.whites_turn = (active == "w");

    // --- Castling rights ---
    board.castling_rights = NO_CASTLING;

    if (castling != "-")
    {
        for (char c : castling)
        {
            switch (c)
            {
                case 'K':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | WHITE_KING_SIDE);
                    break;

                case 'Q':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | WHITE_QUEEN_SIDE);
                    break;

                case 'k':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | BLACK_KING_SIDE);
                    break;

                case 'q':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | BLACK_QUEEN_SIDE);
                    break;
            }
        }
    }

    // --- En-passant square ---
    if (enpassant != "-" && enpassant.size() == 2)
    {
        int file = enpassant[0] - 'a';
        int rank = enpassant[1] - '1';
        board.ep_square = rank * 8 + file;
    }
    else
    {
        board.ep_square = -1;
    }

    // --- Clocks ---
    board.halve_move_counter  = halfmove;
    board.full_move_counter = fullmove;

    // --- Derived bitboards ---
    board.white.update_side();
    board.black.update_side();
    board.complete_board = board.white.side_all | board.black.side_all;
}



