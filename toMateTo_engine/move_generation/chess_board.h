#ifndef CHESS_BOARD
#define CHESS_BOARD

#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <bit>

#include "toMateTo_engine/move_generation/move_stack.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/move_generation/types.h"


void set_index_zero(Bitboard* bitboard, Bitboard index); 
int msb_index(Bitboard bb);
void set_index_one(Bitboard* bitboard, Bitboard index);
extern int RESULT_COUNT;

static inline int8_t color_dir(int8_t magnitude, bool white) {
    return white ? magnitude : -magnitude;
}

struct one_side 
{
    Bitboard knights;
    Bitboard pawns;
    Bitboard rooks;
    Bitboard bishop;
    Bitboard king;
    Bitboard queen;
    Bitboard side_all;

    Bitboard save_king_squares;

    inline void update_side()
    {
        side_all = knights | pawns | rooks | bishop | king | queen;
    }

    void setup_side(bool is_white) {
    if (is_white) {
        pawns = 0x000000000000FF00 & ~((1ULL << 9)  |  // b2
                               (1ULL << 11) |  // d2
                               (1ULL << 12) |  // e2
                               (1ULL << 14));  // g2
        rooks  = 0x0000000000000081;  // a1, h1 (Bits 0, 7)
        knights= 0x0000000000000042;  // b1, g1 (Bits 1, 6)
        bishop = 0x0000000000000024;  // c1, f1 (Bits 2, 5)
        queen  = 0x0000000000000008;  // d1 (Bit 3)
        king   = 0x0000000000000010;  // e1 (Bit 4)
    } else {
        pawns  = 0x00FF000000000000;  // Bits 48–55
        rooks  = 0x8100000000000000;  // a8, h8 (Bits 56, 63)
        knights= 0x4200000000000000;  // b8, g8 (Bits 57, 62)
        bishop = 0x2400000000000000;  // c8, f8 (Bits 58, 61)
        queen  = 0x0800000000000000;  // d8 (Bit 59)
        king   = 0x1000000000000000;  // e8 (Bit 60)
    }

    update_side(); // setzt side_all = kombiniertes Bitboard
}


};


inline void remove_piece(one_side& s, PieceType pt, square sq) {
    Bitboard b = ~(1ULL << sq);
    switch (pt) {
        case PAWN:   s.pawns   &= b; break;
        case KNIGHT: s.knights &= b; break;
        case BISHOP: s.bishop  &= b; break;
        case ROOK:   s.rooks   &= b; break;
        case QUEEN:  s.queen   &= b; break;
        case KING:   s.king    &= b; break;
        default: break;
    }
}

inline void add_piece(one_side& s, PieceType pt, square sq) {
    Bitboard b = (1ULL << sq);
    switch (pt) {
        case PAWN:   s.pawns   |= b; break;
        case KNIGHT: s.knights |= b; break;
        case BISHOP: s.bishop  |= b; break;
        case ROOK:   s.rooks   |= b; break;
        case QUEEN:  s.queen   |= b; break;
        case KING:   s.king    |= b; break;
        default: break;
    }
}

inline PieceType piece_on(const one_side& s, square sq) {
    Bitboard b = 1ULL << sq;
    if (s.pawns   & b) return PAWN;
    if (s.knights & b) return KNIGHT;
    if (s.bishop  & b) return BISHOP;
    if (s.rooks   & b) return ROOK;
    if (s.queen   & b) return QUEEN;
    if (s.king    & b) return KING;
    return NO_PIECE_TYPE;
}


struct chess_board
{
    one_side white;
    one_side black;

    bool whites_turn;
    square ep_square;

    int16_t halve_move_counter;
    int16_t full_move_counter;

    void setup_chess_board(){
        white.setup_side(true);
        black.setup_side(false);

        update_board();
        whites_turn = true;
    };

    void print_board() {
        const char empty = '.';

        // Array für jedes Feld auf dem Board
        char board[64];
        for (int i = 0; i < 64; ++i) board[i] = empty;

        // writes the symbol of the Piece on the correct position of the String representation
        auto set_piece = [&](Bitboard bitboard, char piece_symbol) {
            Bitboard copy_bitboard = bitboard;
            while(copy_bitboard){
                int index = msb_index(copy_bitboard);
                board[index] = piece_symbol;
                set_index_zero(&copy_bitboard, index);
            }
        };

        // Weiße Figuren
        set_piece(white.pawns,  'P');
        set_piece(white.knights,'N');
        set_piece(white.bishop, 'B');
        set_piece(white.rooks,  'R');
        set_piece(white.queen,  'Q');
        set_piece(white.king,   'K');

        // Schwarze Figuren
        set_piece(black.pawns,  'p');
        set_piece(black.knights,'n');
        set_piece(black.bishop, 'b');
        set_piece(black.rooks,  'r');
        set_piece(black.queen,  'q');
        set_piece(black.king,   'k');

        // Ausgabe des Boards: von oben nach unten (Rank 8 bis 1)
        for (int rank = 7; rank >= 0; --rank) {
            std::cout << rank + 1 << " ";
            for (int file = 0; file < 8; ++file) {
                std::cout << board[rank * 8 + file] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "  a b c d e f g h" << std::endl;
    }

    inline void update_white()
    {
        white.update_side();
    };

    inline void update_black()
    {
        black.update_side();
    };

    Bitboard complete_board;
    Bitboard pinned_pieces;
    // Pieces currently attacking the King
    Bitboard attacking_pieces;
    // Can be used if king is attacked only once to filter where pieces can go!
    Bitboard attack_defend_squares;

    int8_t attack_count;

    CastlingRights castling_rights;  


    inline void update_board()
    {
        update_black();
        update_white();
        complete_board =  white.side_all | black.side_all;
    }
};

int try_all_moves(chess_board* cb, int depth);

void make_move(chess_board* cb, Move m, StateInfo& st);

void undo_move(chess_board* cb, Move m, const StateInfo& st);

Move* find_pawn_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy);

Move* add_all_pawn_moves(Bitboard* results, Move* moves, bool color);

Move* add_prom(Bitboard destinations, Move* moves, int8_t offset, bool color);

Move* add_ep(Bitboard destinations, Move* moves, int8_t offset, bool color);

Move* add_pawn_moves(Bitboard destinations, Move* moves, int8_t offset, bool color);

Move* add_normal_moves(square from, Bitboard destinations, Move* moves);

Move* find_all_moves(Move* moves, chess_board* chess_board);

Move* find_knight_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy);

Move* find_bishop_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* bishops);

Move* find_rook_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* rooks);

Move* find_king_save_squares(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, square king_position);

bool is_save_square(Move* moves, one_side* player, one_side* enemy, square pos);

int8_t attack_amounts_to_square(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind);

void find_pin_information(chess_board* chess_board, one_side* player, one_side* enemy, Bitboard pos_ind);

inline square pop_lsb(Bitboard &board){
    square bishop_index = __builtin_ctzll(board);        // Get index of least significant bit
    board &= board - 1;
    return bishop_index;
};

inline Bitboard magic_lookup(Bitboard occ, const MagicTableEntry& m) {
    int index = (int)((occ * m.magic_number) >> (64 - m.relevant_bits));
    return m.attack_list[index];
}

inline Bitboard pins_magic(int square, Bitboard blocked, const MagicTableEntry table[]){
    return magic_lookup(blocked, table[square]);
}

inline Bitboard is_diagonal_attacked(chess_board* chess_board,  int pos_ind, Bitboard relevant_squares){
    Bitboard relevant_attackers_and_defenders = relevant_squares & chess_board->complete_board;
    return pins_magic(pos_ind, relevant_attackers_and_defenders, PINNED_PIECES_BISHOP_MAGIC);
}

inline Bitboard is_straight_attacked(chess_board* chess_board, int pos_ind, Bitboard relevant_squares){
    // für die attacks muss hier wirklich gefragt werden ob die figur eine gegnereischer Rook oder Queen
    Bitboard relevant_attackers_and_defenders = relevant_squares & chess_board->complete_board;
    return pins_magic(pos_ind, relevant_attackers_and_defenders, PINNED_PIECES_ROOK_MAGIC);
}

inline Bitboard attackers_magic(int square, Bitboard attackers,const MagicTableEntry table[],const MagicTableEntry pattern[]){
    Bitboard masked = attackers & table[square].mask;
    return magic_lookup(masked, pattern[square]);
}

inline Bitboard get_straight_pins(one_side* enemy, one_side* player,int pos_ind, Bitboard attack_mask){
    Bitboard blocked = attack_mask & (enemy->knights | enemy->bishop | enemy->pawns | player->side_all);
    return pins_magic(pos_ind, blocked, PINNED_PIECES_ROOK_MAGIC);
}

inline Bitboard get_diagonal_pins(one_side* enemy, one_side* player, int pos_ind, Bitboard attack_mask){
    Bitboard blocked = attack_mask & (enemy->knights | enemy->rooks | enemy->pawns | player->side_all);
    return pins_magic(pos_ind, blocked, PINNED_PIECES_BISHOP_MAGIC);
}

inline Bitboard get_diagonal_attackers(one_side* enemy, int pos_ind) {
    return attackers_magic(pos_ind, enemy->bishop | enemy->queen, BISHOP_MAGIC, ATTACK_PATTERN_BISHOP_MAGIC);
}

inline Bitboard sliding_magic(int square, Bitboard occ, const MagicTableEntry table[], Bitboard blockers_mask = ~0ULL){
    return magic_lookup(occ & table[square].mask, table[square]) & blockers_mask;
}

Move* add_castling(Move* moves, chess_board* board, one_side* player, one_side* enemy, square king_pos, bool is_white);

inline Bitboard get_straight_attackers(one_side* enemy, square pos_ind) {
    return attackers_magic(pos_ind, enemy->rooks | enemy->queen, ROOK_MAGIC, ATTACK_PATTERN_ROOK_MAGIC);
}

inline Bitboard get_straight_attackers_pluss_side(one_side* enemy, square pos_ind) {
    return attackers_magic(pos_ind, enemy->rooks | enemy->queen, ROOK_MAGIC, ROOK_MAGIC);
}

inline Bitboard get_diagonal_attackers_pluss_side(one_side* enemy, int pos_ind) {
    return attackers_magic(pos_ind, enemy->bishop | enemy->queen, BISHOP_MAGIC, BISHOP_MAGIC);
}


inline Bitboard bishop_magic(int square, const chess_board* board, const one_side* player){
    return sliding_magic(square, board->complete_board, BISHOP_MAGIC, ~player->side_all);
}

inline Bitboard rook_magic(int square,const chess_board* board, const one_side* player){
    return sliding_magic(square, board->complete_board, ROOK_MAGIC, ~player->side_all);
}

void setup_fen_position(chess_board& board, const std::string& fen);

std::string board_to_fen(const chess_board& board);

#endif 