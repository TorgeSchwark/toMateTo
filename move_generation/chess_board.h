#ifndef CHESS_BOARD
#define CHESS_BOARD

#include <cstdint>
#include "move_stack.h"
#include "../generation/knight_tables.h"
#include <iostream>


void set_index_zero(int64_t* bitboard, int64_t index); 
int msb_index(int64_t bb);
void set_index_one(int64_t* bitboard, int64_t index);

struct one_side 
{
    int64_t knights;
    int64_t pawns;
    int64_t rooks;
    int64_t bishop;
    int64_t king;
    int64_t queen;
    int64_t side_all;

    void update_side()
    {
        side_all = knights | pawns | rooks | bishop | king | queen;
    }

    void setup_side(bool is_white) {
    if (is_white) {
        pawns  = 0x000000000000FF00;  // Bits 8–15
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


struct chess_board
{
    one_side white;
    one_side black;
    bool whites_turn;


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

        // Hilfsfunktion zum Setzen eines Felds, falls Bit gesetzt
        auto set_piece = [&](int64_t bitboard, char symbol) {
            int64_t copy_bitboard = bitboard;
            while(copy_bitboard){
                int index = msb_index(copy_bitboard);
                board[index] = symbol;
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

    void update_white()
    {
        white.update_side();
    };

    void update_black()
    {
        black.update_side();
    };

    int64_t complete_board;

    void update_board()
    {
        update_black();
        update_white();
        complete_board =  white.side_all | black.side_all;
    }
};

void find_all_moves(move_stack* move_stack, chess_board* chess_board);

void find_knight_moves(move_stack* move_stack, one_side* player, one_side* enemy);


void find_bishop_moves(move_stack* move_stack, one_side* player, one_side* enemy);

#endif 