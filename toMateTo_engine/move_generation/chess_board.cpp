#include "chess_board.h"

int RESULT_COUNT = 9;

#define BB_SHIFT(bb, s, color) \
    ((color) ?  ((bb) << (s)) : ((bb) >> (s)) )

int try_all_moves(chess_board* cb, int depth) {
    

    int count = 0;
    Move moves[256];
    Move* end = find_all_moves(moves, cb);
    int num_moves = end - moves;
    if (depth == 1 || (!num_moves)){
        // int count = (end - moves);
        // printf("all moves in that fuck shit");
        // for (int i = 0; i < count; ++i) {
        //         std::cout << moves[i].move_to_string() << "\n";
        //     }
        return num_moves;
    }

    for (Move* m = moves; m != end; ++m) {

        // 🔒 Nur h2 -> h4 zulassen, wenn depth == 3
        // if (depth == 3) {
        //     if (!(m->from_sq() == H2 && m->to_sq() == H4)) {
        //         continue; // alle anderen Züge überspringen
        //     }
        // }else if(depth == 2){
        //    if (!(m->from_sq() == B8 && m->to_sq() == A6)) {
        //         continue; // alle anderen Züge überspringen
        //     } 
        // }

        StateInfo st;
        make_move(cb, *m, st);

        int check_perft = try_all_moves(cb, depth - 1);

        // if (depth == 1) {
        //     printf("%s\n", m->move_to_string().c_str());
        //     printf("number: %i\n\n", check_perft);
        // }

        count += check_perft;
        undo_move(cb, *m, st);
    }
    return count;
}



void make_move(chess_board* cb, Move m, StateInfo& st) {
    one_side& us   = cb->whites_turn ? cb->white : cb->black;
    one_side& them = cb->whites_turn ? cb->black : cb->white;

    square from = m.from_sq();
    square to   = m.to_sq();
    int flag    = m.move_flag();

    // ---------- save undo info ----------
    st.epSquare = cb->ep_square;
    st.castling = cb->castling_rights;
    st.rule50   = cb->halve_move_counter;
    st.captured = NO_PIECE_TYPE;

    PieceType moving = piece_on(us, from);

    // ---------- handle captures ----------
    if (flag == 2) { // en passant
        square cap = to + (cb->whites_turn ? -8 : 8);
        st.captured = PAWN;
        remove_piece(them, PAWN, cap);
    } else {
        PieceType cap = piece_on(them, to);
        if (cap != NO_PIECE_TYPE) {
            st.captured = cap;
            remove_piece(them, cap, to);
        }
    }

    // ---------- move piece ----------
    remove_piece(us, moving, from);
    add_piece(us, moving, to);

    // ---------- promotion ----------
    if (flag == 1) {
        remove_piece(us, PAWN, to);
        add_piece(us, (PieceType)m.promo_piece(), to);
    }

    // ---------- castling ----------
    if (flag == 3) {
        if (to == G1) { // white king side
            remove_piece(us, ROOK, H1);
            add_piece(us, ROOK, F1);
        } else if (to == C1) {
            remove_piece(us, ROOK, A1);
            add_piece(us, ROOK, D1);
        } else if (to == G8) {
            remove_piece(us, ROOK, H8);
            add_piece(us, ROOK, F8);
        } else if (to == C8) {
            remove_piece(us, ROOK, A8);
            add_piece(us, ROOK, D8);
        }
    }

    if (moving == KING) {
        if (cb->whites_turn)
            cb->castling_rights &= ~(WHITE_KING_SIDE | WHITE_QUEEN_SIDE);
        else
            cb->castling_rights &= ~(BLACK_KING_SIDE | BLACK_QUEEN_SIDE);
    }


    if (moving == ROOK) {
        if (from == A1) cb->castling_rights &= ~WHITE_QUEEN_SIDE;
        if (from == H1) cb->castling_rights &= ~WHITE_KING_SIDE;
        if (from == A8) cb->castling_rights &= ~BLACK_QUEEN_SIDE;
        if (from == H8) cb->castling_rights &= ~BLACK_KING_SIDE;
    }

    if (st.captured == ROOK) {
        if (to == A1) cb->castling_rights &= ~WHITE_QUEEN_SIDE;
        if (to == H1) cb->castling_rights &= ~WHITE_KING_SIDE;
        if (to == A8) cb->castling_rights &= ~BLACK_QUEEN_SIDE;
        if (to == H8) cb->castling_rights &= ~BLACK_KING_SIDE;
    }


    // ---------- update EP square ----------
    cb->ep_square = SQ_NONE;
    if (moving == PAWN && abs(to - from) == 16) {
        cb->ep_square = (from + to) / 2;
    }

    // ---------- update counters ----------
    cb->halve_move_counter =
        (moving == PAWN || st.captured != NO_PIECE_TYPE) ? 0 : cb->halve_move_counter + 1;

    if (!cb->whites_turn)
        cb->full_move_counter++;

    // ---------- update bitboards ----------
    (*cb).update_board();

    // ---------- switch side ----------
    cb->whites_turn ^= 1;
}

void undo_move(chess_board* cb, Move m, const StateInfo& st) {

    // ---------- restore side to move ----------
    cb->whites_turn ^= 1;

    one_side& us   = cb->whites_turn ? cb->white : cb->black;
    one_side& them = cb->whites_turn ? cb->black : cb->white;

    square from = m.from_sq();
    square to   = m.to_sq();
    int flag    = m.move_flag();

    // ---------- restore state ----------
    cb->ep_square          = st.epSquare;
    cb->castling_rights    = st.castling;
    cb->halve_move_counter = st.rule50;

    if (!cb->whites_turn)
        cb->full_move_counter--;

    // ---------- undo castling rook move ----------
    if (flag == 3) {
        if (to == G1) {
            remove_piece(us, ROOK, F1);
            add_piece(us, ROOK, H1);
        } else if (to == C1) {
            remove_piece(us, ROOK, D1);
            add_piece(us, ROOK, A1);
        } else if (to == G8) {
            remove_piece(us, ROOK, F8);
            add_piece(us, ROOK, H8);
        } else if (to == C8) {
            remove_piece(us, ROOK, D8);
            add_piece(us, ROOK, A8);
        }
    }

    // ---------- undo move (promotion or normal) ----------
    if (flag == 1) { // promotion
        remove_piece(us, (PieceType)m.promo_piece(), to);
        add_piece(us, PAWN, from);
    } else {
        PieceType moving = piece_on(us, to);
        remove_piece(us, moving, to);
        add_piece(us, moving, from);
    }

    // ---------- restore captured piece ----------
    if (st.captured != NO_PIECE_TYPE) {
        if (flag == 2) { // en passant
            square cap = to + (cb->whites_turn ? -8 : 8);
            add_piece(them, PAWN, cap);
        } else {
            add_piece(them, st.captured, to);
        }
    }

    // ---------- rebuild derived bitboards ----------
    cb->update_board();
}

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

    if(chess_board->whites_turn){
        square white_king_square = __builtin_ctzll(chess_board->white.king);

        find_pin_information(chess_board,  &(chess_board->white), &(chess_board->black), white_king_square);

        moves = find_king_save_squares(moves, chess_board,  &(chess_board->white), &(chess_board->black), white_king_square);
        moves = add_castling(moves, chess_board, &(chess_board->white), &(chess_board->black), white_king_square, chess_board->whites_turn);

        if(chess_board->attack_count < 2){
            if(chess_board->attack_count == 1){
                chess_board->attack_defend_squares = SQUARES_IN_BETWEEN[white_king_square][__builtin_ctzll(chess_board->attacking_pieces)];
            }

            moves = find_knight_moves(moves, chess_board, &(chess_board->white), &(chess_board->black));

            moves = find_pawn_moves(moves, chess_board, &(chess_board->white), &(chess_board->black));

            moves = find_bishop_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.bishop));
            moves = find_rook_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.rooks));

            moves = find_bishop_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
            moves = find_rook_moves(moves, chess_board, &(chess_board->white), &(chess_board->black), &(chess_board->white.queen));
        }
    }else{
        square black_king_square = __builtin_ctzll(chess_board->black.king);

        find_pin_information(chess_board, &(chess_board->black), &(chess_board->white), black_king_square);

        moves = find_king_save_squares(moves, chess_board,  &(chess_board->black), &(chess_board->white), black_king_square);
        moves = add_castling(moves, chess_board,  &(chess_board->black), &(chess_board->white), black_king_square, chess_board->whites_turn);

        if(chess_board->attack_count < 2){
            if(chess_board->attack_count == 1){
                chess_board->attack_defend_squares = SQUARES_IN_BETWEEN[black_king_square][__builtin_ctzll(chess_board->attacking_pieces)];
            }

            moves = find_knight_moves(moves, chess_board, &(chess_board->black), &(chess_board->white));

            moves = find_pawn_moves(moves, chess_board, &(chess_board->black), &(chess_board->white));

            moves = find_bishop_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.bishop));
            moves = find_rook_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.rooks));

            moves = find_bishop_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.queen));
            moves = find_rook_moves(moves, chess_board, &(chess_board->black), &(chess_board->white), &(chess_board->black.queen));
        }
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

void find_different_pawn_moves(Bitboard pawns, bool is_white, Bitboard empty, Bitboard enemy, chess_board* chess_board, Bitboard* results){
    // single push

    results[PUSH1] = BB_SHIFT(pawns, FORWARD, is_white) & empty;

    // double push
    results[PUSH2] =
        BB_SHIFT((pawns & PAWN_ROW[is_white]), FORWARD, is_white) & empty;
    results[PUSH2] = BB_SHIFT(results[PUSH2], FORWARD, is_white) & empty;

    // captures
    Bitboard pawnCapL = BB_SHIFT(pawns, FORWARD_LEFT, is_white) & (~BOARD_FILE[FILE_H]);
    Bitboard pawnCapR = BB_SHIFT(pawns, FORWARD_RIGHT, is_white) & (~BOARD_FILE[FILE_A]);

    // normal captures
    results[CAPL] = pawnCapL & enemy;
    results[CAPR]= pawnCapR & enemy;

    // promotions
    results[PROMO_PUSH] = results[PUSH1] & PROMOTION_ROW[is_white];
    results[PUSH1] &= ~PROMOTION_ROW[is_white];
    results[PROMO_CAPL] = results[CAPL] & PROMOTION_ROW[is_white];
    results[PROMO_CAPR]  = results[CAPR] & PROMOTION_ROW[is_white];

    // en passant
    if(!((1LL << (chess_board->ep_square-color_dir(FORWARD, is_white))) & chess_board->pinned_pieces)){
        results[EPL] = pawnCapL & (1ULL << chess_board->ep_square);
        results[EPR] = pawnCapR & (1ULL << chess_board->ep_square);
    }else{
        results[EPL] = 0LL;
        results[EPR] = 0LL;
    }
}

Move* add_all_pawn_moves(Bitboard* results, Move* moves, bool color){
    moves = add_pawn_moves(results[PUSH1], moves, FORWARD, color);
    moves = add_pawn_moves(results[PUSH2], moves, DOUBLE_FORWARD, color);
    moves = add_pawn_moves(results[CAPL], moves, FORWARD_LEFT, color);
    moves = add_pawn_moves(results[CAPR], moves, FORWARD_RIGHT, color);
    moves = add_ep(results[EPL], moves, FORWARD_LEFT, color);
    moves = add_ep(results[EPR], moves, FORWARD_RIGHT, color);
    moves = add_prom(results[PROMO_PUSH], moves, FORWARD, color);
    moves = add_prom(results[PROMO_CAPL], moves, FORWARD_LEFT, color);
    moves = add_prom(results[PROMO_CAPR], moves, FORWARD_RIGHT, color);
    return moves;
}

Move* find_pawn_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy){
    Bitboard empty = ~chess_board->complete_board;
    Bitboard pawns = player->pawns;

    Bitboard pinned_pawns = pawns & chess_board->pinned_pieces;
    pawns &= ~pinned_pawns;
    
    Bitboard results[RESULT_COUNT] = {0LL};
    Bitboard results_pinned[RESULT_COUNT];
    find_different_pawn_moves(pawns, chess_board->whites_turn, empty, enemy->side_all, chess_board, results);
    if(!chess_board->attack_count){
        // TODO:: sadly cant be implemented like this since allowed squares is the whole line and that not efficiently fixable
        if(pinned_pawns){
            // Pinned pawns can only walk if king is not under attack
            while(pinned_pawns){
                square pinned_pawn_square = pop_lsb(pinned_pawns);
                Bitboard pinned_pawn = 1LL << pinned_pawn_square;
                Bitboard allowed_squares = SQUARES_ON_THE_LINE[pinned_pawn_square][__builtin_ctzll(player->king)];

                find_different_pawn_moves(pinned_pawn ,chess_board->whites_turn, empty, enemy->side_all, chess_board, results_pinned);
                
                for (int i = 0; i < RESULT_COUNT; ++i)
                    results[i] |= (allowed_squares & results_pinned[i]);
            }            
        }
    }else{
        for (int i = 0; i < RESULT_COUNT; ++i){
            results[i] &= chess_board->attack_defend_squares;
        }
    }
    
    moves =  add_all_pawn_moves(results, moves, chess_board->whites_turn);

    return moves;

}

Move* add_prom(Bitboard destinations, Move* moves, int8_t offset, bool color){
    while(destinations){
        square to = pop_lsb(destinations);
        square from = to - color_dir(offset, color);
        for(int8_t type = BISHOP; type <= QUEEN; type++){
            *moves++ = Move(from, to, PROMOTION, type);
        }
    }return moves;
}

Move* add_ep(Bitboard destinations, Move* moves, int8_t offset, bool color){
    if(destinations){
        square to = pop_lsb(destinations);
        *moves++ = Move(to-color_dir(offset, color), to, EN_PASSANT);
    }return moves;
}

Move* add_pawn_moves(Bitboard destinations, Move* moves, int8_t offset, bool color){
    while(destinations){
        square to = pop_lsb(destinations);
        *moves++ = Move(to-color_dir(offset, color), to);
    }return moves;
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

void find_pin_information(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind){
    // Finds Number of attacks on the king
    // Stores all pinned Pieces 
    // Stores all pieces attacking the King

    int8_t count_attacks = 0;

    // Straight block
    Bitboard relevant_squares = get_straight_attackers(enemy, pos_ind);
    Bitboard pinned_pieces_straight = get_straight_pins(enemy, player, pos_ind, relevant_squares);
    Bitboard straight_attackers = is_straight_attacked(chess_board, pos_ind, relevant_squares);
    
    // Diagonal block
    relevant_squares = get_diagonal_attackers(enemy, pos_ind);
    Bitboard pinned_pieces_diagonal = get_diagonal_pins(enemy, player, pos_ind, relevant_squares);
    Bitboard diagonal_attackers = is_diagonal_attacked(chess_board, pos_ind, relevant_squares);

    // Knights and Pawns
    Bitboard knight_attackers = KNIGHT_LOOKUP_TABLE[pos_ind] & enemy->knights;
    Bitboard pawn_attackers = PAWN_ATTACK_LOOKUP_TABLE[chess_board->whites_turn][pos_ind]&enemy->pawns;

    // store all pinned pieces: These can also be enemy pieces Pinned by there Team
    chess_board->pinned_pieces = pinned_pieces_diagonal | pinned_pieces_straight;
    //store all attacking pieces
    chess_board->attacking_pieces = straight_attackers | diagonal_attackers | knight_attackers | pawn_attackers;
    count_attacks = __builtin_popcountll(chess_board->attacking_pieces);

    chess_board->attack_count = count_attacks;
}

extern const CastlingRights CASTLING_FLAG[2][2] = {
    { BLACK_KING_SIDE,  BLACK_QUEEN_SIDE },
    { WHITE_KING_SIDE,  WHITE_QUEEN_SIDE }
};

Move* add_castling(Move* moves,chess_board* board, one_side* player, one_side* enemy, square king_pos, bool is_white) {
    if (board->attack_count)
        return moves;

    for (int cs = 0; cs < 2; ++cs) {

        // 1) has right?
        if (!(board->castling_rights & CASTLING_FLAG[is_white][cs]))
            continue;

        // 2) intermediate squares must be empty
        if (board->complete_board & CASTLE_FREE[is_white][cs])
            continue;

        // 3) check safety (only those not yet computed)
        Bitboard check = CASTLE_SAVE[is_white][cs] & ~player->save_king_squares;

        bool safe = true;
        while (check) {
            square sq = pop_lsb(check);
            if (!is_save_square(board, player, enemy, sq)) {
                safe = false;
                break;
            }
        }

        if (!safe)
            continue;

        // 4) add move
        
        *moves++ = Move(king_pos, CASTLE_TO[is_white][cs], CASTLING);
    }

    return moves;
}

Move* find_king_save_squares(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, square king_position){
    Bitboard possible_king_moves = KING_MOVES_MASK[king_position] & ~player->side_all;
    player->save_king_squares = 0LL;
    while(possible_king_moves){
        square to = pop_lsb(possible_king_moves);
        if(is_save_square(chess_board, player, enemy, to)){ // there can be a piece as long as the square is not under attack
            // this whole function could be split in only parallel moves and the rest so this is not done for every free square:
            player->save_king_squares |= (1ULL << to);
            *moves++ = Move(king_position, to);
        }
    }return moves;
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

