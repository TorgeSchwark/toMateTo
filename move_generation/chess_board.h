#ifndef CHESS_BOARD
#define CHESS_BOARD

#include <cstdint>

struct chess_board
{
    int64_t white_knights;
    int64_t white_pawns;
    int64_t white_rooks;
    int64_t white_bishop;
    int64_t white_king;
    int64_t white_queen;
    int64_t white;

    void update_white()
    {
        white = white_bishop | white_king | white_knights | white_rooks | white_pawns | white_queen;
    }

    int64_t black_knights;
    int64_t black_pawns;
    int64_t black_rooks;
    int64_t black_bishop;
    int64_t black_king;
    int64_t black_queen;
    int64_t black;

    void update_black()
    {
        black = black_queen | black_king | black_bishop | black_rooks | black_pawns | black_knights;
    }

    int64_t complete_board;

    void update_board()
    {
        update_black();
        update_white();
        complete_board =  white | black;
    }

    int last_move_type;
};


#endif 