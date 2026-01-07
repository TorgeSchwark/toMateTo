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
        square white_king_square = __builtin_ctzll(chess_board->white.king);
        attack_amounts_to_square(chess_board,  &(chess_board->white), &(chess_board->black), white_king_square);
        if(chess_board->attack_count){
            moves = find_king_save_squares(moves, chess_board,  &(chess_board->white), &(chess_board->black), white_king_square);
        }
        if(chess_board->attack_count < 2){
            chess_board->attack_defend_squares = SQUARES_IN_BETWEEN[white_king_square][chess_board->attacking_pieces];
            moves = find_knight_moves(moves, chess_board, &(chess_board->white), &(chess_board->black));
            moves = find_bishop_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.bishop));
            moves = find_rook_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.rooks));

            moves = find_bishop_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
            moves = find_rook_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
        }
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
        bool is_pinned = bishop_square&chess_board->pinned_pieces;
        if(is_pinned && (chess_board->attack_count)){
            return moves;
        }    

        Bitboard bishop_destinations = bishop_magic(bishop_square, chess_board, player);

        // a pinned piece can only walk if the king is not in check
        if(chess_board->attack_count){
            // if we get here and king is attacked it means piece in not pinned!
            // Filter for defending moves
            bishop_destinations &= chess_board->attack_defend_squares;
        }else if(is_pinned){
            // if we get here and piece is pinned it means king is not attacked!
            // only walk on pinned line!
            bishop_destinations |= SQUARES_ON_THE_LINE[bishop_square][player->king];
        }// else king is not attacked and piece not pinned!

        moves = add_normal_moves(bishop_square, bishop_destinations, moves);
    }
    return moves;
}

Move* find_rook_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* rook){
    Bitboard rooks = *rook;
    while(rooks){
        square rook_square = pop_lsb(rooks);    
        bool is_pinned = rook_square&chess_board->pinned_pieces;
        // piece cant move to save king if pinned.
        if(is_pinned && (chess_board->attack_count)){
            return moves;
        }    

        Bitboard rook_destinations = rook_magic(rook_square, chess_board, player);

        // a pinned piece can only walk if the king is not in check
        if(chess_board->attack_count){
            // if we get here and king is attacked it means piece in not pinned!
            // Filter for defending moves
            rook_destinations &= chess_board->attack_defend_squares;
        }else if(is_pinned){
            // if we get here and piece is pinned it means king is not attacked!
            // only walk on pinned line!
            rook_destinations |= SQUARES_ON_THE_LINE[rook_square][player->king];
        }// else king is not attacked and piece not pinned!

        moves = add_normal_moves(rook_square, rook_destinations, moves);
    }
    return moves;
}

Move* find_knight_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy) {
    Bitboard knights = player->knights;
    Bitboard negative_player = ~player->side_all;
    while (knights) {

        square knight_square= pop_lsb(knights);
        bool is_pinned = knight_square&chess_board->pinned_pieces;

        // pinned knight cant move!
        if(is_pinned){
            return moves;
        }  

        Bitboard knight_destinations = KNIGHT_LOOKUP_TABLE[knight_square] & negative_player;

        if(chess_board->attack_count){
            // if we get here and king is attacked it means piece in not pinned!
            // Filter for defending moves
            knight_destinations &= chess_board->attack_defend_squares;
        }
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

int8_t attack_amounts_to_square(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind){
    // this is the same as is_save_square but it gives the extra information about how many attacks face ibe square

    int8_t count_attacks;
    // returns a Bitboard where every square between nearest attacker and the square is marked! 
    Bitboard relevant_squares = get_straight_attackers(enemy, pos_ind);
    // with this we get only the attacking piece itself and every piece in the line of attack
    count_attacks += __builtin_popcountll(is_straight_attacked(chess_board, pos_ind, relevant_squares));
    
    // return directions where an attacker exists until attacker including attacker!
    relevant_squares = get_diagonal_attackers(enemy, pos_ind);
    // this will mark attackers and every blocking piece both enemy blocking and team
    count_attacks += __builtin_popcountll(is_diagonal_attacked(chess_board, pos_ind, relevant_squares));

    count_attacks += __builtin_popcountll(KNIGHT_LOOKUP_TABLE[pos_ind] & enemy->knights);
    
    count_attacks += __builtin_popcountll(PAWN_ATTACK_LOOKUP_TABLE[chess_board->whites_turn][pos_ind]&enemy->pawns);

    return count_attacks;
}

int8_t find_pin_information(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind){
    // Finds Number of attacks on the king
    // Stores all pinned Pieces 
    // Stores all pieces attacking the King

    int8_t count_attacks;

    // Straight block
    Bitboard relevant_squares = get_straight_attackers(enemy, pos_ind);
    Bitboard pinned_pieces_straight = get_straight_pins(enemy, player, pos_ind, relevant_squares);
    Bitboard straight_attackers = is_straight_attacked(chess_board, pos_ind, relevant_squares);
    count_attacks += __builtin_popcountll(straight_attackers);
    
    // Diagonal block
    relevant_squares = get_diagonal_attackers(enemy, pos_ind);
    Bitboard pinned_pieces_diagonal = get_diagonal_pins(enemy, player, pos_ind, relevant_squares);
    Bitboard diagonal_attackers = is_straight_attacked(chess_board, pos_ind, relevant_squares);
    count_attacks += __builtin_popcountll(diagonal_attackers);

    // Knights and Pawns
    Bitboard knight_attackers = KNIGHT_LOOKUP_TABLE[pos_ind] & enemy->knights;
    count_attacks += __builtin_popcountll(knight_attackers);
    Bitboard pawn_attackers = PAWN_ATTACK_LOOKUP_TABLE[chess_board->whites_turn][pos_ind]&enemy->pawns;
    count_attacks += __builtin_popcountll(pawn_attackers);

    // store all pinned pieces: These can also be enemy pieces Pinned by there Team
    chess_board->pinned_pieces |= pinned_pieces_diagonal | pinned_pieces_straight;
    //store all attacking pieces
    chess_board->attacking_pieces |= straight_attackers | diagonal_attackers;

    chess_board->attack_count = count_attacks;
}

Move* find_king_save_squares(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, square king_position){
    Bitboard possible_king_moves = KING_MOVES_MASK[king_position] & ~player->side_all; 
    while(possible_king_moves){
        square to = pop_lsb(possible_king_moves);
        if(is_save_square(chess_board, player, enemy, to)){ // there can be a piece as long as the square is not under attack
            *moves++ = Move(king_position, pop_lsb(possible_king_moves));
        }
    }return moves;
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

std::string board_to_fen(const chess_board& board)
{
    std::string fen;

    // --- Piece placement ---
    for (int rank = 7; rank >= 0; --rank)
    {
        int empty = 0;

        for (int file = 0; file < 8; ++file)
        {
            int square = rank * 8 + file;
            Bitboard bit = 1ULL << square;
            char piece = 0;

            // White pieces
            if (board.white.pawns   & bit) piece = 'P';
            else if (board.white.knights & bit) piece = 'N';
            else if (board.white.bishop  & bit) piece = 'B';
            else if (board.white.rooks   & bit) piece = 'R';
            else if (board.white.queen   & bit) piece = 'Q';
            else if (board.white.king    & bit) piece = 'K';

            // Black pieces
            else if (board.black.pawns   & bit) piece = 'p';
            else if (board.black.knights & bit) piece = 'n';
            else if (board.black.bishop  & bit) piece = 'b';
            else if (board.black.rooks   & bit) piece = 'r';
            else if (board.black.queen   & bit) piece = 'q';
            else if (board.black.king    & bit) piece = 'k';

            if (piece)
            {
                if (empty > 0)
                {
                    fen += std::to_string(empty);
                    empty = 0;
                }
                fen += piece;
            }
            else
            {
                ++empty;
            }
        }

        if (empty > 0)
            fen += std::to_string(empty);

        if (rank > 0)
            fen += '/';
    }

    // --- Active color ---
    fen += ' ';
    fen += (board.whites_turn ? 'w' : 'b');

    // --- Castling rights ---
    fen += ' ';
    std::string castling;

    if (board.castling_rights & WHITE_KING_SIDE)  castling += 'K';
    if (board.castling_rights & WHITE_QUEEN_SIDE) castling += 'Q';
    if (board.castling_rights & BLACK_KING_SIDE)  castling += 'k';
    if (board.castling_rights & BLACK_QUEEN_SIDE) castling += 'q';

    fen += (castling.empty() ? "-" : castling);

    // --- En-passant square ---
    fen += ' ';
    if (board.ep_square >= 0)
    {
        int file = board.ep_square % 8;
        int rank = board.ep_square / 8;
        fen += static_cast<char>('a' + file);
        fen += static_cast<char>('1' + rank);
    }
    else
    {
        fen += "-";
    }

    // --- Clocks ---
    fen += ' ';
    fen += std::to_string(board.halve_move_counter);
    fen += ' ';
    fen += std::to_string(board.full_move_counter);

    return fen;
}
